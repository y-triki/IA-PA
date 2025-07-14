# pdf_pipeline.py
import os
import fitz
from tokenizers import ByteLevelBPETokenizer
from typing import Union
import torch
import re


class PDFProcessor:
    def __init__(self, tokenizer_dir , max_len=1024, max_tokens_total=8192):
        # Construit un chemin ABSOLU à partir du fichier pdf_pipeline.py
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        tokenizer_dir = os.path.join(BASE_DIR, '..', 'models', 'student_model')
        self.tokenizer = ByteLevelBPETokenizer(
            os.path.join(tokenizer_dir, "vocab.json"),
            os.path.join(tokenizer_dir, "merges.txt")
        )
        self.tokenizer.add_special_tokens(["[CITE]", "[MATH]", "[EQUATION]"])
        self.max_len = max_len
        self.max_tokens_total = max_tokens_total

    def extract_text(self, pdf_file: Union[str, bytes]) -> str:
        """Extrait le texte du PDF à partir d'un chemin de fichier (str) OU de bytes."""
        if isinstance(pdf_file, str):  # Si chemin du fichier
            doc = fitz.open(pdf_file)

        elif isinstance(pdf_file, bytes):  # Si contenu du fichier en octets
            doc = fitz.open(stream=pdf_file, filetype="pdf")

        else:
            raise ValueError("pdf_file doit être un chemin de fichier (str) ou des octets (bytes).")

        text = ""
        for page in doc:
            text += page.get_text("text", sort=True) + "\n\n"

        # Add cleaning from our new clean_pdf_text function
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        text = re.sub(r'�', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def chunk_and_tokenize(self, text, force_chunk=False):
        """Tokenization du texte extrait du PDF, avec ou sans découpage."""
        tokens = self.tokenizer.encode(text).ids
        if not force_chunk and len(tokens) <= self.max_len:
            return [tokens]  # Pas besoin de découper

        return [tokens[i:i + self.max_len] for i in range(0, len(tokens), self.max_len)]

    def extract_text_with_metadata(self, pdf_file: Union[str, bytes]) -> dict:
        """Extract text with layout information"""
        if isinstance(pdf_file, str):
            doc = fitz.open(pdf_file)
        elif isinstance(pdf_file, bytes):
            doc = fitz.open(stream=pdf_file, filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text("text", sort=True) + "\n\n"

        return {
            "full_text": text,
            "pages": len(doc)
        }