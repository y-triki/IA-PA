<<<<<<< HEAD
# generation_pipeline.py
import os
import torch
from backend.pipelines.nucleus_decoder import nucleus_sampling_decode
from backend.pipelines.pdf_pipeline import PDFProcessor
from backend.pipelines.translation import translate_fr_to_en, translate_en_to_fr, detect_lang
from backend.pipelines.utils import clean_and_format_summary
from backend.pipelines.model import TransformerSummarizer
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# ---- Paramètres ---- #
device = torch.device("cpu")

# Chemin du fichier où est définie la variable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'models', 'summary', 'best_model.pth')

# ---- Initialisation ---- #
BASE_DIR = os.path.dirname(__file__)
processor = PDFProcessor(tokenizer_dir="../backend/models/summary")
tokenizer = processor.tokenizer
vocab_size = tokenizer.get_vocab_size()
pad_id = tokenizer.token_to_id("<pad>") if "<pad>" in tokenizer.get_vocab() else 0

# ---- Chargement du modèle ---- #
model = TransformerSummarizer(
    vocab_size,
    pad_id=pad_id,
    d_model=768,
    nhead=12,
    num_layers=6,
    dropout=0.2,
    use_checkpointing=False
).to(device)

checkpoint = torch.load(model_path, map_location=device)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)  # Au cas où c'était déjà uniquement le state_dict

# ---- Chargement du modèle MiniLM ---- #
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def summarize_pdf(pdf_file, max_chunk_len=30, p=0.9, threshold=0.6):
    """Résumé détaillé du contenu d'un PDF donné en entrée (bytes)."""
    try:
        print("[summarize_pdf] Début du processus...")
        raw_text = processor.extract_text(pdf_file)
        print(f"[summarize_pdf] Extrait brut (1000 chars max): {raw_text[:1000]}")
        print(f"[summarize_pdf] Longueur du texte extrait : {len(raw_text)} caractères.")

        # Détection de la langue
        lang_detected = detect(raw_text)
        print(f"[INFO] Langue du texte détectée : {lang_detected}")

        # Si français, traduction en anglais pour le modèle
        if lang_detected == "fr":
            print("[INFO] Texte en français, traduction en anglais pour le modèle...")
            text_for_model = translate_fr_to_en(raw_text)
        else:
            text_for_model = raw_text
            print(f"[INFO] Texte en {lang_detected}, pas de traduction nécessaire pour le modèle.")

        # Découpe du texte
        max_chunks_for_test = 7
        chunks = processor.chunk_and_tokenize(text_for_model, max_chunk_len)[:max_chunks_for_test]
        summaries = []
        for idx, chunk in enumerate(chunks):
            print(f"[summarize_pdf] Traitement du chunk {idx + 1}/{len(chunks)}...")
            src_ids = torch.tensor([chunk], device=device)
            src_mask = (src_ids != pad_id).long()
            summary_en = nucleus_sampling_decode(model, src_ids, src_mask, tokenizer, p=p)
            summaries.append(clean_and_format_summary(summary_en))

        clustered = cluster_summaries(summaries, embedder, threshold)
        structured_en = "\n\n".join([" ".join(group) for group in clustered])

        # Si français à la base, on retraduit le résumé final
        if lang_detected == "fr":
            structured_output = translate_en_to_fr(structured_en)
            print("[summarize_pdf] Résumé traduit en français.")
        else:
            structured_output = structured_en
            print("[summarize_pdf] Résumé final construit.")

        return {
            "summary": structured_output,
            "chunks_count": len(chunks),
        }

    except Exception as e:
        print(f"[summarize_pdf][ERREUR] {e}")
        return {
            "summary": "",
            "chunks_count": 0,
        }


def cluster_summaries(sentences, model, threshold=0.6):
    """Regroupe les résumés par proximité sémantique."""
    embeddings = model.encode(sentences, convert_to_tensor=True)
    clusters = []
    assigned = [False] * len(sentences)

    for i, emb in enumerate(embeddings):
        if assigned[i]:
            continue
        group = [sentences[i]]
        assigned[i] = True
        for j in range(i + 1, len(sentences)):
            if not assigned[j]:
                sim = util.pytorch_cos_sim(emb, embeddings[j]).item()
                if sim > threshold:
                    group.append(sentences[j])
                    assigned[j] = True
        clusters.append(group)

    return clusters

=======
# generation_pipeline.py
import os
import torch
from backend.pipelines.nucleus_decoder import nucleus_sampling_decode
from backend.pipelines.pdf_pipeline import PDFProcessor
from backend.pipelines.translation import translate_fr_to_en, translate_en_to_fr, detect_lang
from backend.pipelines.utils import clean_and_format_summary
from backend.pipelines.model import TransformerSummarizer
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# ---- Paramètres ---- #
device = torch.device("cpu")

# Chemin du fichier où est définie la variable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, '..', 'models', 'summary', 'best_model.pth')

# ---- Initialisation ---- #
BASE_DIR = os.path.dirname(__file__)
processor = PDFProcessor(tokenizer_dir="../backend/models/summary")
tokenizer = processor.tokenizer
vocab_size = tokenizer.get_vocab_size()
pad_id = tokenizer.token_to_id("<pad>") if "<pad>" in tokenizer.get_vocab() else 0

# ---- Chargement du modèle ---- #
model = TransformerSummarizer(
    vocab_size,
    pad_id=pad_id,
    d_model=768,
    nhead=12,
    num_layers=6,
    dropout=0.2,
    use_checkpointing=False
).to(device)

checkpoint = torch.load(model_path, map_location=device)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)  # Au cas où c'était déjà uniquement le state_dict

# ---- Chargement du modèle MiniLM ---- #
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def summarize_pdf(pdf_file, max_chunk_len=30, p=0.9, threshold=0.6):
    """Résumé détaillé du contenu d'un PDF donné en entrée (bytes)."""
    try:
        print("[summarize_pdf] Début du processus...")
        raw_text = processor.extract_text(pdf_file)
        print(f"[summarize_pdf] Extrait brut (1000 chars max): {raw_text[:1000]}")
        print(f"[summarize_pdf] Longueur du texte extrait : {len(raw_text)} caractères.")

        # Détection de la langue
        lang_detected = detect(raw_text)
        print(f"[INFO] Langue du texte détectée : {lang_detected}")

        # Si français, traduction en anglais pour le modèle
        if lang_detected == "fr":
            print("[INFO] Texte en français, traduction en anglais pour le modèle...")
            text_for_model = translate_fr_to_en(raw_text)
        else:
            text_for_model = raw_text
            print(f"[INFO] Texte en {lang_detected}, pas de traduction nécessaire pour le modèle.")

        # Découpe du texte
        max_chunks_for_test = 7
        chunks = processor.chunk_and_tokenize(text_for_model, max_chunk_len)[:max_chunks_for_test]
        summaries = []
        for idx, chunk in enumerate(chunks):
            print(f"[summarize_pdf] Traitement du chunk {idx + 1}/{len(chunks)}...")
            src_ids = torch.tensor([chunk], device=device)
            src_mask = (src_ids != pad_id).long()
            summary_en = nucleus_sampling_decode(model, src_ids, src_mask, tokenizer, p=p)
            summaries.append(clean_and_format_summary(summary_en))

        clustered = cluster_summaries(summaries, embedder, threshold)
        structured_en = "\n\n".join([" ".join(group) for group in clustered])

        # Si français à la base, on retraduit le résumé final
        if lang_detected == "fr":
            structured_output = translate_en_to_fr(structured_en)
            print("[summarize_pdf] Résumé traduit en français.")
        else:
            structured_output = structured_en
            print("[summarize_pdf] Résumé final construit.")

        return {
            "summary": structured_output,
            "chunks_count": len(chunks),
        }

    except Exception as e:
        print(f"[summarize_pdf][ERREUR] {e}")
        return {
            "summary": "",
            "chunks_count": 0,
        }


def cluster_summaries(sentences, model, threshold=0.6):
    """Regroupe les résumés par proximité sémantique."""
    embeddings = model.encode(sentences, convert_to_tensor=True)
    clusters = []
    assigned = [False] * len(sentences)

    for i, emb in enumerate(embeddings):
        if assigned[i]:
            continue
        group = [sentences[i]]
        assigned[i] = True
        for j in range(i + 1, len(sentences)):
            if not assigned[j]:
                sim = util.pytorch_cos_sim(emb, embeddings[j]).item()
                if sim > threshold:
                    group.append(sentences[j])
                    assigned[j] = True
        clusters.append(group)

    return clusters

>>>>>>> ecbe693 (Mise à jour depuis EC2 : dernières modifs locales)
