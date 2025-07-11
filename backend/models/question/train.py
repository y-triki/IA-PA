import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from ...pipelines.model import EncoderRNN, DecoderRNN, Seq2Seq

# ---------- CONFIG ----------
EMBED_SIZE = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 15
TEACHER_FORCING_RATIO = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 1. Tokenisation simple ----------
class Vocab:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.index = 4

    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            if word not in self.word2idx:
                self.word2idx[word] = self.index
                self.idx2word[self.index] = word
                self.index += 1

    def sentence_to_indices(self, sentence, max_len):
        tokens = sentence.lower().split()
        ids = [self.word2idx.get(w, self.word2idx["<unk>"]) for w in tokens[:max_len - 2]]
        return [self.word2idx["<sos>"]] + ids + [self.word2idx["<eos>"]]

    def from_dict(self, d):
        self.word2idx = d["word2idx"]
        self.idx2word = {int(k): v for k, v in d["idx2word"].items()}
        self.index = d["index"]

    def __len__(self):
        return len(self.word2idx)


# ---------- 2. Dataset ----------
class SquadDataset(Dataset):
    def __init__(self, path, vocab, max_input_len=100, max_target_len=30):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.pairs = data
        self.vocab = vocab
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        for item in self.pairs:
            vocab.add_sentence(item["input"])
            vocab.add_sentence(item["target"])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text = self.pairs[idx]["input"]
        target_text = self.pairs[idx]["target"]

        src = self.vocab.sentence_to_indices(input_text, self.max_input_len)
        trg = self.vocab.sentence_to_indices(target_text, self.max_target_len)

        src += [0] * (self.max_input_len - len(src))
        trg += [0] * (self.max_target_len - len(trg))

        return torch.tensor(src), torch.tensor(trg)


# ---------- 3. Entraînement ----------
def train():
    print("Chargement des données...")

    vocab = Vocab()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "..", "data", "squad_pairs.json")

    dataset = SquadDataset(data_path, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Taille du vocabulaire : {len(vocab)}")
    print(f"Initialisation du modèle...")

    encoder = EncoderRNN(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    decoder = DecoderRNN(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Début de l'entraînement...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in dataloader:
            src, trg = batch
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Époque {epoch+1}/{EPOCHS} - Perte moyenne : {total_loss / len(dataloader):.4f}")

    # Sauvegarde
    os.makedirs("models/question", exist_ok=True)
    torch.save(model.state_dict(), "models/question/question_generator.pt")
    torch.save(vocab.to_dict(), "models/question/vocab.pt")
    print("Modèle sauvegardé dans models/question/")


if __name__ == "__main__":
    train()
