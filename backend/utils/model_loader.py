# model_loader.py
import os
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

_tokenizer_en2fr = None
_model_en2fr = None
_tokenizer_fr2en = None
_model_fr2en = None

def initialize_translation_models():
    global _tokenizer_en2fr, _model_en2fr, _tokenizer_fr2en, _model_fr2en

    if _tokenizer_en2fr is None:
        _tokenizer_en2fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        _model_en2fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    if _tokenizer_fr2en is None:
        _tokenizer_fr2en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        _model_fr2en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    return {
        "en2fr": (_tokenizer_en2fr, _model_en2fr),
        "fr2en": (_tokenizer_fr2en, _model_fr2en)
    }