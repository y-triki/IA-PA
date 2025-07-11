<<<<<<< HEAD
import argostranslate.package
import argostranslate.translate
from langdetect import detect

def detect_lang(text: str) -> str:
    """
    Détecte automatiquement la langue d’un texte. Retourne "fr", "en", etc.
    """
    try:
        return detect(text)
    except:
        return "unknown"

def translate_fr_to_en(text):
    if detect(text) != "fr":
        return text  # Ne traduit pas si ce n'est pas du français
    translator = argostranslate.translate.get_installed_languages()
    from_lang = list(filter(lambda x: x.code == "fr", translator))[0]
    to_lang = list(filter(lambda x: x.code == "en", translator))[0]
    translation = from_lang.get_translation(to_lang)
    return translation.translate(text)

def translate_en_to_fr(text):
    if detect(text) != "en":
        return text  # Ne traduit pas si ce n'est pas de l'anglais
    translator = argostranslate.translate.get_installed_languages()
    from_lang = list(filter(lambda x: x.code == "en", translator))[0]
    to_lang = list(filter(lambda x: x.code == "fr", translator))[0]
    translation = from_lang.get_translation(to_lang)
    return translation.translate(text)
=======
import argostranslate.package
import argostranslate.translate
from langdetect import detect

def detect_lang(text: str) -> str:
    """
    Détecte automatiquement la langue d’un texte. Retourne "fr", "en", etc.
    """
    try:
        return detect(text)
    except:
        return "unknown"

def translate_fr_to_en(text):
    if detect(text) != "fr":
        return text  # Ne traduit pas si ce n'est pas du français
    translator = argostranslate.translate.get_installed_languages()
    from_lang = list(filter(lambda x: x.code == "fr", translator))[0]
    to_lang = list(filter(lambda x: x.code == "en", translator))[0]
    translation = from_lang.get_translation(to_lang)
    return translation.translate(text)

def translate_en_to_fr(text):
    if detect(text) != "en":
        return text  # Ne traduit pas si ce n'est pas de l'anglais
    translator = argostranslate.translate.get_installed_languages()
    from_lang = list(filter(lambda x: x.code == "en", translator))[0]
    to_lang = list(filter(lambda x: x.code == "fr", translator))[0]
    translation = from_lang.get_translation(to_lang)
    return translation.translate(text)
>>>>>>> ecbe693 (Mise à jour depuis EC2 : dernières modifs locales)
