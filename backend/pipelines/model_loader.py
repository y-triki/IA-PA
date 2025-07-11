# model_loader.py
import os
import fasttext
from transformers import MarianMTModel, MarianTokenizer

_lang_detector = None
_tokenizer_en2fr = None
_model_en2fr = None
_tokenizer_fr2en = None
_model_fr2en = None

def initialize_translation_models():
    global _lang_detector, _tokenizer_en2fr, _model_en2fr, _tokenizer_fr2en, _model_fr2en

    if _lang_detector is None:
        # Chemin absolu vers lid.176.bin
        lid_path = os.path.join(os.path.dirname(__file__), "lid.176.bin")
        _lang_detector = fasttext.load_model(lid_path)

    if _tokenizer_en2fr is None:
        _tokenizer_en2fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        _model_en2fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    if _tokenizer_fr2en is None:
        _tokenizer_fr2en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        _model_fr2en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    return {
        "lang_detector": _lang_detector,
        "en2fr": (_tokenizer_en2fr, _model_en2fr),
        "fr2en": (_tokenizer_fr2en, _model_fr2en)
    }
