import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.datasets import Multi30k
from torchtext.data.metrics import bleu_score
from torchtext.data import Field, BucketIterator


def translate_sentence(model, sentence, german, english, device, max_length=50):
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)
    text_to_indices = [german.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, validation_data, testing_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10_000, min_freq=2)


class Transformer(nn.Module):

    def __init__(self, embedding_size, src_vocab_size, trg_vocab_size,
                 src_pad_idx, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout,
                 max_len,
                 device):
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout)

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    
    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask
    
    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device))
        trg_positions = (torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device))

        embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
        embed_trg = self.dropout((self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)))

        src_padding_mask = self.make_src_mask(src).to(self.device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask = src_padding_mask, tgt_mask=trg_mask)
        out = self.fc_out(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

num_epochs = 300
learning_rate = 3e-4
batch_size = 32

src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 32
num_heads = 16
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.1
max_length = 100
forward_expansion = 5
src_pad_idx = english.vocab.stoi["<pad>"]

training_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, testing_data),
    batch_size = batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device = device
)

model = Transformer(embedding_size=embedding_size, src_vocab_size=src_vocab_size,
                    trg_vocab_size=trg_vocab_size, src_pad_idx=src_pad_idx, num_heads=num_heads,
                    num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                    forward_expansion=forward_expansion, dropout=dropout, max_len=max_length,
                    device = device).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

sentence = "ein pferd geht unter einer bruck neben enem boot."

for epoch in range(num_epochs):
    losses = []
    print(f"[Epoch]: {epoch} / {num_epochs}")
 
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename="ger_eng.tar")
    model.eval()
    print(translate_sentence(model=model, sentence=sentence, german=german, english=english, device=device, max_length=max_length))

    model.train()

    step = 0

    for batch_idx, batch in enumerate(training_iterator):
        inp_data = batch.src
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        step += 1
        print(f"{step}/{len(training_iterator)}")
    print(f"Bleu: {bleu(testing_data, model, german, english, device)*100:.4f}")
    print(f"Loss: {sum(losses)/len(losses)}")
