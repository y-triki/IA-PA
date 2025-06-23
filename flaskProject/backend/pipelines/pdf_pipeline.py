# pdf_pipeline.py
import os
import os
import fitz
from tokenizers import ByteLevelBPETokenizer
import torch


class PDFProcessor:
    def __init__(self, tokenizer_path, max_len=1024, max_tokens_total=8192):
        # Construit un chemin ABSOLU à partir du fichier pdf_pipeline.py
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        tokenizer_dir = os.path.join(BASE_DIR, '..', 'models')
        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_dir, "vocab.json"),
            os.path.join(tokenizer_dir, "merges.txt")
        )
        self.tokenizer.add_special_tokens(["[CITE]", "[MATH]", "[EQUATION]"])
        self.max_len = max_len
        self.max_tokens_total = max_tokens_total

    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text

    def chunk_and_tokenize(self, text, force_chunk=False):
        tokens = self.tokenizer.encode(text).ids
        if not force_chunk and len(tokens) <= self.max_len:
            return [tokens]  # Pas besoin de découper
        return [tokens[i:i + self.max_len] for i in range(0, len(tokens), self.max_len)]


