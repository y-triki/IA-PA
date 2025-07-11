<<<<<<< HEAD
# utils/translation_utils.py

import sys
import os
import fasttext

# Ajouter le dossier parent au path pour trouver model_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.utils.model_loader import initialize_translation_models


# -------- INITIALISATION DES MODÈLES --------
models = initialize_translation_models()

try:
    _tokenizer_en2fr, _model_en2fr = models["en2fr"]
    _tokenizer_fr2en, _model_fr2en = models["fr2en"]
except KeyError as e:
    raise RuntimeError(f" Clé manquante dans initialize_translation_models() : {e}")

# -------- CHARGEMENT DU MODÈLE DE LANGUE --------
lid_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "lid.176.bin"))

if not os.path.exists(lid_model_path):
    raise FileNotFoundError(f" Fichier de langue FastText introuvable : {lid_model_path}")

_lang_detector = fasttext.load_model(lid_model_path)

def detect_language(text: str) -> str:
    """Détecte la langue d’un texte"""
    clean_text = text.replace("\n", " ").replace("\r", " ").strip()
    lang = _lang_detector.predict(clean_text)
    return lang[0][0].replace("__label__", "")


def translate_to_en(text: str) -> str:
    """Traduit du français vers l’anglais"""
    tokens = _tokenizer_fr2en.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    output = _model_fr2en.generate(**tokens)
    return _tokenizer_fr2en.batch_decode(output, skip_special_tokens=True)[0]


def translate_to_fr(text: str) -> str:
    """Traduit de l’anglais vers le français"""
    tokens = _tokenizer_en2fr.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    output = _model_en2fr.generate(**tokens)
    return _tokenizer_en2fr.batch_decode(output, skip_special_tokens=True)[0]
=======
# utils/translation_utils.py

import sys
import os
import fasttext

# Ajouter le dossier parent au path pour trouver model_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.utils.model_loader import initialize_translation_models


# -------- INITIALISATION DES MODÈLES --------
models = initialize_translation_models()

try:
    _tokenizer_en2fr, _model_en2fr = models["en2fr"]
    _tokenizer_fr2en, _model_fr2en = models["fr2en"]
except KeyError as e:
    raise RuntimeError(f" Clé manquante dans initialize_translation_models() : {e}")

# -------- CHARGEMENT DU MODÈLE DE LANGUE --------
lid_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "lid.176.bin"))

if not os.path.exists(lid_model_path):
    raise FileNotFoundError(f" Fichier de langue FastText introuvable : {lid_model_path}")

_lang_detector = fasttext.load_model(lid_model_path)

def detect_language(text: str) -> str:
    """Détecte la langue d’un texte"""
    clean_text = text.replace("\n", " ").replace("\r", " ").strip()
    lang = _lang_detector.predict(clean_text)
    return lang[0][0].replace("__label__", "")


def translate_to_en(text: str) -> str:
    """Traduit du français vers l’anglais"""
    tokens = _tokenizer_fr2en.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    output = _model_fr2en.generate(**tokens)
    return _tokenizer_fr2en.batch_decode(output, skip_special_tokens=True)[0]


def translate_to_fr(text: str) -> str:
    """Traduit de l’anglais vers le français"""
    tokens = _tokenizer_en2fr.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    output = _model_en2fr.generate(**tokens)
    return _tokenizer_en2fr.batch_decode(output, skip_special_tokens=True)[0]
>>>>>>> ecbe693 (Mise à jour depuis EC2 : dernières modifs locales)
