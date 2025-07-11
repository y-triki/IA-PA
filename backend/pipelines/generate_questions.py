import os
import sys
import re
import torch
import fitz  # PyMuPDF
from docx import Document

from backend.pipelines.translation import translate_en_to_fr, translate_fr_to_en, detect_lang

# Ajouter les chemins pour accéder aux modules du backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.pipelines.model import EncoderRNN, DecoderRNN, Seq2Seq
from backend.utils.translation_utils import detect_language, translate_to_en, translate_to_fr
from backend.utils.model_loader import initialize_translation_models

# Initialiser les modèles de traduction
import os
import torch
from backend.pipelines.model import EncoderRNN, DecoderRNN, Seq2Seq
from backend.pipelines.utils import lire_fichier

"""

# Initialiser les modèles de traduction
initialize_translation_models()

class Vocab:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.index = 4

    def from_dict(self, d):
        self.word2idx = d["word2idx"]
        self.idx2word = {int(k): v for k, v in d["idx2word"].items()}
        self.index = d["index"]

    def __len__(self):
        return len(self.word2idx)

    def encode(self, sentence, max_len=100):
        tokens = sentence.lower().split()
        ids = [self.word2idx.get(w, self.word2idx["<unk>"]) for w in tokens[:max_len - 2]]
        return [self.word2idx["<sos>"]] + ids + [self.word2idx["<eos>"]]

    def decode(self, indices):
        words = [self.idx2word.get(idx, "<unk>") for idx in indices]
        return " ".join([w for w in words if w not in ["<sos>", "<eos>", "<pad>"]])

def generer_question(texte, model, vocab, device, max_len=30):
    model.eval()
    src = torch.tensor(vocab.encode(texte)).unsqueeze(0).to(device)
    encoder_outputs = model.encoder(src)
    hidden = encoder_outputs[:, -1, :].unsqueeze(0)
    input = torch.tensor([[vocab.word2idx["<sos>"]]]).to(device)
    output_indices = []

    for _ in range(max_len):
        output, hidden = model.decoder(input, hidden, encoder_outputs)
        top1 = output.argmax(1)
        output_indices.append(top1.item())
        if top1.item() == vocab.word2idx["<eos>"]:
            break
        input = top1.unsqueeze(1)

    return vocab.decode(output_indices)

def decouper_en_phrases(text):
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 30]

def generate_open_questions(file_path):
    texte = lire_fichier(file_path)
    print("[DEBUG] Texte extrait :", texte[:500])

    if not texte.strip():
        return []

    langue = detect_language(texte)
    texte_en = translate_to_en(texte) if langue == "fr" else texte

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_data = torch.load("backend/models/question/vocab.pt", map_location=device)
    vocab = Vocab()
    vocab.from_dict(vocab_data)

    encoder = EncoderRNN(len(vocab), 256, 512)
    decoder = DecoderRNN(len(vocab), 256, 512)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load("backend/models/question/question_generator.pt", map_location=device))

    # Découper en phrases longues cohérentes
    phrases = decouper_en_phrases(texte_en)
    chunks = phrases[:5]  # max 5 questions

    questions = []
    for i, chunk in enumerate(chunks):
        question_en = generer_question(chunk, model, vocab, device)
        question_finale = translate_to_fr(question_en) if langue == "fr" else question_en
        if question_finale.strip():
            questions.append({"question": question_finale})
        print(f"[DEBUG] Q{i+1}: {question_finale}")

    return questions
"""
#---------

import os
import torch
import fitz  # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langdetect import detect

# Modèle multilingue
MODEL_NAME = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Pipeline de traduction de l'anglais vers le français
translate_en_to_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr", device=0 if torch.cuda.is_available() else -1)

def lire_fichier(path):
    """Extrait le texte brut depuis un fichier PDF ou TXT"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            doc = fitz.open(path)
            return "\n".join([page.get_text() for page in doc])
        except Exception as e:
            print(f"[ERREUR] PDF : {e}")
            return ""
    elif ext == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[ERREUR] TXT : {e}")
            return ""
    else:
        print(f"[ERREUR] Format non supporté : {ext}")
        return ""

def split_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_open_questions(path, max_questions=5):
    texte = lire_fichier(path)
    if not texte.strip():
        print("[DEBUG] Aucun texte extrait.")
        return []

    print("[DEBUG] Texte extrait :", texte[:400])
    chunks = split_text(texte, chunk_size=200)[:max_questions]
    questions = []

    for i, chunk in enumerate(chunks):
        prompt = f"Generate a question for a student based on the following passage: {chunk}"
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

        outputs = model.generate(
            inputs["input_ids"],
            max_length=64,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9
        )

        question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Si la question est en anglais, traduire en français
        try:
            if detect(question) == "en":
                question_fr = translate_en_to_fr(question)[0]["translation_text"]
            else:
                question_fr = question
        except Exception as e:
            print(f"[ERREUR traduction] : {e}")
            question_fr = question

        print(f"[DEBUG] Q{i+1}: {question_fr}")
        if question_fr:
            questions.append({"question": question_fr})

    return questions
