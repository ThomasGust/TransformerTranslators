import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, embedding_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embedding_size
        self.heads = heads

        self.head_dim = self.embed_size // self.heads

        assert (self.head_dim * heads == embedding_size), "Embedding size must be divisible by the number of heads!"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.out = nn.Linear(self.heads*self.head_dim, embedding_size)
    
    def forward(self, v, k, q, mask):

        N = q.shape[0]
        vl, kl, ql = v.shape[1], k.shape[1], q.shape[1]
        print(v.shape)
        print(k.shape)
        print(q.shape)
        v = v.reshape(N, vl, self.heads, self.head_dim)
        k = k.reshape(N, kl, self.heads, self.head_dim)
        q = q.reshape(N, ql, self.heads, self.head_dim)

        v = self.values(v)
        k = self.keys(k)
        q = self.queries(q)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy/(self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(N, ql, self.heads*self.head_dim)
        out = self.out(out)
        return out
    
class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.n1 = nn.LayerNorm(embed_size)
        self.n2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask):
        attention = self.attention(v, k, q, mask)

        x= self.dropout(self.n1(attention + q))

        forward = self.ff(x)
        return self.dropout(self.n2(forward + x))

class Encoder(nn.Module):

    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.src_vocab_size = src_vocab_size
        self.num_layers = num_layers
        self.heads = heads
        self.device = device
        self.forward_expansion = forward_expansion
        self.max_length = max_length
        self.d = dropout

        self.word_embedding = nn.Embedding(self.src_vocab_size, self.embed_size)
        self.pos_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=self.d,
                forward_expansion=forward_expansion) for layer in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.pos_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device
    
    def forward(self, x, v, k, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)

        q = self.dropout(self.norm(attention + x))

        out = self.transformer_block(v, k, q, src_mask)

        return out

class Decoder(nn.Module):

    def __init__(self, target_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()

        self.device = device
        self.target_vocab_size = target_vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.forward_expansion = forward_expansion
        self.d = dropout
        self.max_length = max_length

        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.heads, self.forward_expansion, self.d, self.device) for layer in range(num_layers)])

        self.fc_out = nn.Linear(self.embed_size, self.target_vocab_size)

        self.dropout = nn.Dropout(self.d)

    
    def forward(self, x, enc_out, src_mask, target_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask)
        
        out = self.fc_out(x)

        return out
    

class Transformer(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512, num_layers = 12, forward_expansion=4, heads=8, dropout=0.1, device="cuda", max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder =Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 6, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    
    model = Transformer(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size, src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)
