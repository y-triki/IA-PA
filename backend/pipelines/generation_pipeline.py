# generation_pipeline.py
import os
import torch
import re
import nltk
import numpy as np
from backend.pipelines.nucleus_decoder import nucleus_sampling_decode
from backend.pipelines.pdf_pipeline import PDFProcessor
from backend.pipelines.translation import translate_fr_to_en, translate_en_to_fr, detect_lang
from backend.pipelines.utils import clean_and_format_summary
from backend.pipelines.model import TransformerSummarizer
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

print("[DEBUG] >>> IMPORT RE OK")
print(">>> DEBUG: generation_pipeline.py chargé depuis", __file__)

# ---- Paramètres ---- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# ---- Initialisation ---- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Student model paths
student_model_path = os.path.join(BASE_DIR, '..', 'models', 'student_model', 'final_student_model.pth')
student_tokenizer_dir = os.path.join(BASE_DIR, '..', 'models', 'student_model')

# Teacher model path (if needed)
teacher_checkpoint_dir = os.path.join(BASE_DIR, '..', 'models', 'teacher_model')

# ---- Chargement du tokenizer étudiant ---- #
student_processor = PDFProcessor(tokenizer_dir=student_tokenizer_dir)
student_tokenizer = student_processor.tokenizer
vocab_size = student_tokenizer.get_vocab_size()
pad_id = student_tokenizer.token_to_id("<pad>") if "<pad>" in student_tokenizer.get_vocab() else 0

# ---- Chargement du modèle étudiant ---- #
student_model = TransformerSummarizer(
    vocab_size,
    pad_id=pad_id,
    d_model=768,
    nhead=12,
    num_layers=6,
    dropout=0.2,
    use_checkpointing=False
).to(device)

try:
    checkpoint = torch.load(student_model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        student_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        student_model.load_state_dict(checkpoint)
    print(" Student model loaded successfully")
except Exception as e:
    print(f" Error loading student model: {e}")
    raise

# ---- Chargement du modèle d'embedding ---- #
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Préparation du modèle enseignant (chargé seulement si nécessaire) ---- #
teacher_model = None
teacher_tokenizer = None


def load_teacher_model():
    """Charge le modèle enseignant seulement si nécessaire"""
    global teacher_model, teacher_tokenizer

    if teacher_model is None:
        from transformers import BartForConditionalGeneration, BartTokenizer

        print("Loading teacher model...")
        teacher_tokenizer = BartTokenizer.from_pretrained(teacher_checkpoint_dir)
        teacher_model = BartForConditionalGeneration.from_pretrained(teacher_checkpoint_dir)
        teacher_model.to(device)
        teacher_model.eval()
        print("Teacher model loaded")


def summarize_with_teacher(text, max_length=256):
    """Utilise le modèle enseignant pour générer un résumé"""
    load_teacher_model()

    inputs = teacher_tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        summary_ids = teacher_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.5
        )

    return teacher_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def calculate_coherence(original_text, summary):
    """Calcule la cohérence entre le texte original et le résumé"""
    try:
        # Embeddings pour les phrases clés
        orig_embedding = embedder.encode(original_text[:1000], convert_to_tensor=True)
        summary_embedding = embedder.encode(summary, convert_to_tensor=True)

        # Similarité cosinus
        similarity = util.pytorch_cos_sim(orig_embedding, summary_embedding).item()
        print(f" Similarity score: {similarity:.2f}")
        return similarity
    except Exception as e:
        print(f" Coherence calculation failed: {e}")
        return 0  # Assume incoherent if calculation fails


def is_coherent(summary, original_text, threshold=0.4):
    """Détermine si un résumé est cohérent avec le texte original"""
    if not summary.strip():
        return False

    # Vérifications simples
    if len(summary.split()) < 5:  # Résumé trop court
        return False

    if re.search(r'\b(\w+)( \1\b){2,}', summary):  # Répétitions excessives
        return False

    # Calcul de similarité sémantique
    return calculate_coherence(original_text, summary) >= threshold


def summarize_pdf(pdf_file, max_chunk_len=1024, p=0.9, threshold=0.6):
    """Résumé détaillé du contenu d'un PDF donné en entrée (bytes)."""
    original_text = ""
    lang_detected = "fr"

    try:
        print("[summarize_pdf] Début du processus...")

        # Extraction du texte
        original_text = student_processor.extract_text(pdf_file)
        print(f"[summarize_pdf] Longueur du texte extrait : {len(original_text)} caractères.")

        # Détection de la langue
        lang_detected = detect_lang(original_text[:500]) if len(original_text) > 100 else "fr"
        print(f"[INFO] Langue détectée : {lang_detected}")

        # Traduction si nécessaire
        try:
            if lang_detected == "fr":
                print("[INFO] Traduction FR → EN...")
                text_for_model = translate_fr_to_en(original_text)
                print("[INFO] Traduction terminée.")
            else:
                text_for_model = original_text
        except Exception as e:
            print(f"[ERREUR] Problème pendant la traduction : {e}")
            raise

        # Déterminer la longueur du résumé basée sur la taille du document
        summary_length_factor = min(1.0, len(text_for_model) / 10000)  # 0-1 basé sur la longueur
        base_max_len = 100
        dynamic_max_len = base_max_len + int(100 * summary_length_factor)
        print(f"[INFO] Dynamic max length: {dynamic_max_len} tokens")

        # Découpage intelligent avec contexte
        chunks = chunk_text_with_context(
            text_for_model,
            student_tokenizer,
            max_tokens=max_chunk_len,
            overlap=200
        )

        print("[DEBUG] Premier chunk (si dispo) :", chunks[0] if chunks else "Aucun chunk")
        print(f"[INFO] Nombre de chunks créés: {len(chunks)}")

        # Limite raisonnable pour les chunks
        chunks = chunks[:10]

        # Génération des résumés avec le modèle étudiant
        student_summaries = []
        for idx, chunk in enumerate(chunks):
            if not chunk:
                print(f"[WARNING] Chunk {idx} vide, ignoré.")
                continue

        for idx, chunk in enumerate(chunks):
            print(f"[summarize_pdf] Traitement du chunk {idx + 1}/{len(chunks)} avec le modèle étudiant...")
            src_ids = torch.tensor([chunk], device=device)
            src_mask = (src_ids != pad_id).long()

            # Génération avec paramètres optimisés
            summary = nucleus_sampling_decode(
                student_model,
                src_ids,
                src_mask,
                student_tokenizer,
                max_len=dynamic_max_len,
                p=p,
                temperature=0.8,
                repetition_penalty=1.5
            )

            # Nettoyage du résumé
            cleaned_summary = clean_summary_text(summary)
            student_summaries.append(cleaned_summary)

        # Combiner les résumés
        combined_summary = " ".join(student_summaries)

        # Vérifier la cohérence
        if is_coherent(combined_summary, original_text):
            print("[INFO] Student summary is coherent")
            final_summary = combined_summary
        else:
            print("[WARNING] Student summary incoherent - using teacher model")
            # Utiliser le modèle enseignant sur l'ensemble du texte
            final_summary = summarize_with_teacher(text_for_model, max_length=dynamic_max_len * 2)

        # Post-traitement et structuration
        structured_summary = structure_summary(final_summary)

        # Traduction de retour si nécessaire
        if lang_detected == "fr":
            print("[INFO] Traduction EN → FR...")
            french_summary = translate_en_to_fr(structured_summary)
            final_output = clean_french_text(french_summary)
        else:
            final_output = structured_summary

        return {
            "summary": final_output,
            "chunks_count": len(chunks),
            "model_used": "student" if "student" in locals() else "teacher"
        }

    except Exception as e:
        print(f"[summarize_pdf][ERREUR] {e}")
        # Fallback au modèle enseignant en cas d'erreur
        try:
            if original_text:
                print("[FALLBACK] Using teacher model as fallback")
                text_for_model = translate_fr_to_en(original_text) if lang_detected == "fr" else original_text
                summary = summarize_with_teacher(text_for_model)
                return {
                    "summary": translate_en_to_fr(summary) if lang_detected == "fr" else summary,
                    "chunks_count": 1,
                    "model_used": "teacher (fallback)"
                }
        except Exception as fallback_error:
            print(f"[CRITICAL] Fallback failed: {fallback_error}")

        return {
            "summary": "",
            "chunks_count": 0,
            "model_used": "error"
        }


def clean_summary_text(text):
    import re
    """Nettoyage du résumé généré"""
    print("[DEBUG] clean_summary_text called")
    try:
        print("[DEBUG] module re type:", type(re))
    except Exception as e:
        print("[DEBUG ERROR] re is not defined:", e)

    text = re.sub(r"<s>|</s>|<pad>", "", text)
    return text.strip()
    """
    # Suppression des tokens spéciaux
    text = re.sub(r"<s>|</s>|<pad>", "", text)
    # Suppression des répétitions de mots
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # Correction des espaces avant ponctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Suppression des caractères isolés
    text = re.sub(r'\b\w\b', '', text)
    return text.strip()
    """


def clean_french_text(text):
    """Nettoyage spécifique pour le français"""
    # Correction des espaces insécables
    text = re.sub(r'\s+([;:!?])', r'\1', text)
    text = re.sub(r'([«»])\s+', r'\1', text)
    text = re.sub(r'\s+([«»])', r'\1', text)
    # Capitalisation des phrases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(s.capitalize() for s in sentences if s)


teacher_model = None
teacher_tokenizer = None


def load_teacher_model():
    """Lazy-load teacher model when needed"""
    global teacher_model, teacher_tokenizer
    if teacher_model is None:
        from transformers import BartForConditionalGeneration, BartTokenizer

        # Configure paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        teacher_checkpoint_dir = os.path.join(BASE_DIR, '..', 'models', 'teacher_model')

        print(" Loading teacher model for QA...")
        teacher_tokenizer = BartTokenizer.from_pretrained(teacher_checkpoint_dir)
        teacher_model = BartForConditionalGeneration.from_pretrained(teacher_checkpoint_dir)
        teacher_model.to(device)
        teacher_model.eval()
        print(" Teacher model loaded for QA")
def chunk_text_with_context(text, tokenizer, max_tokens=1024, overlap=200):
    """Découpe le texte en chunks avec chevauchement pour maintenir le contexte"""
    # Tokenisation du texte
    if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
        tokens = tokenizer.encode(text).ids
    else:
        tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return [tokens]

    # Création de chunks avec overlap
    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunks.append(tokens[start_idx:end_idx])
        start_idx = end_idx - overlap

    return chunks


def structure_summary(summary_text):
    """Structure le résumé en paragraphes"""
    # Découpage en phrases
    sentences = re.split(r'(?<=[.!?])\s+', summary_text)
    paragraphs = []
    current_para = []
    max_sentences = 4

    for sentence in sentences:
        if len(current_para) < max_sentences:
            current_para.append(sentence)
        else:
            paragraphs.append(" ".join(current_para))
            current_para = [sentence]

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)


# generation_pipeline.py (additions)

def answer_question_with_teacher_french(raw_text: str, question: str, max_length: int = 256) -> str:
    """Use teacher model for French prompt-based question answering"""
    # Load teacher model if not already loaded
    load_teacher_model()

    # Translate only the question to English
    question_en = translate_fr_to_en(question)

    # Keep context in French
    context = raw_text[:15000]  # Limit context

    # Create French-oriented prompt
    prompt = f"""
    Voici une question sur un document en français. Répondez en français.

    Question: {question}

    Contexte du document:
    {context}

    Réponse:
    """

    # Tokenize input
    inputs = teacher_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding="max_length"
    ).to(device)

    # Generate answer
    with torch.no_grad():
        output_ids = teacher_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            temperature=0.7
        )

    # Decode and clean answer
    answer = teacher_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract just the answer part
    if "Réponse:" in answer:
        answer = answer.split("Réponse:")[-1].strip()

    return clean_french_text(answer)