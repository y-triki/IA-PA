# generation_pipeline.py
import os
# Avoid unnecessary tokenizer threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import re
import nltk
import numpy as np
import gc
from backend.pipelines.nucleus_decoder import nucleus_sampling_decode
from backend.pipelines.pdf_pipeline import PDFProcessor
from backend.pipelines.translation import translate_fr_to_en, translate_en_to_fr, detect_lang
from backend.pipelines.utils import clean_and_format_summary
from backend.pipelines.model import TransformerSummarizer
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Reduce PyTorch thread usage to avoid memory spikes under heavy CPU load
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

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
# Force CPU for sentence-transformers to reduce GPU/VRAM pressure; disable progress bars
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

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

        # --- ✅ Tronquage préventif pour éviter des OOM ---
        MAX_CHARS = 10000  # Ajuste à 8000 ou 12000 si besoin
        if len(original_text) > MAX_CHARS:
            print(f"[INFO] Texte trop long, tronqué à {MAX_CHARS} caractères.")
            original_text = original_text[:MAX_CHARS]
        
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
        print(f"[INFO] Dynamic max length (teacher fallback): {dynamic_max_len} tokens")
        
        # --- ✅ Libération mémoire avant le découpage ---
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 5️⃣ Découpage intelligent avec contexte
        chunks = chunk_text_with_context(
            text_for_model,
            student_tokenizer,
            max_tokens=max_chunk_len,
            overlap=200
        )

        print("[DEBUG] Premier chunk (si dispo) :", chunks[0] if chunks else "Aucun chunk")
        print(f"[INFO] Nombre de chunks créés: {len(chunks)}")

        # Limite raisonnable pour les chunks (réduite pour éviter OOM)
        max_chunks = min(5, len(chunks))
        chunks = chunks[:max_chunks]

        # ---- Remplace la boucle "Génération des résumés" dans summarize_pdf ----
        student_summaries = []
        with torch.no_grad():
            for idx, chunk in enumerate(chunks):
                if not chunk:
                    print(f"[WARNING] Chunk {idx} vide, ignoré.")
                    continue

                print(f"[summarize_pdf] Traitement du chunk {idx + 1}/{len(chunks)} avec le modèle étudiant...")

                src_ids = torch.tensor([chunk], device=device)
                src_mask = (src_ids != pad_id).long()

                # 👉 max_length dynamique par chunk
                in_len = src_ids.shape[-1]
                dynamic_max_len_chunk = min(150, max(60, int(in_len * 0.4)))
                print(f"[DEBUG] dynamic_max_len pour chunk {idx+1}: {dynamic_max_len_chunk}")

                summary = nucleus_sampling_decode(
                    student_model,
                    src_ids,
                    src_mask,
                    student_tokenizer,
                    max_len=dynamic_max_len_chunk,
                    p=p,
                    temperature=0.8,
                    repetition_penalty=1.5
                )

                cleaned_summary = clean_summary_text(summary)
                student_summaries.append(cleaned_summary)

                # ✅ Libération mémoire immédiate
                del src_ids, src_mask
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Combiner les résumés
        combined_summary = " ".join(student_summaries)

        # Vérifier la cohérence
        if is_coherent(combined_summary, original_text, threshold=threshold):
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

def _split_into_passages(text: str, max_chars: int = 600, min_chars: int = 200) -> list:
    """Split text into rough passages (paragraphs/sentences) with size bounds."""
    # Normalize whitespace
    txt = re.sub(r"\s+", " ", text).strip()
    # First split by double newlines or periods
    parts = re.split(r"\n\n+|(?<=[.!?])\s+", txt)
    passages = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 < max_chars:
            buf = (buf + " " + p).strip()
        else:
            if len(buf) >= min_chars:
                passages.append(buf)
            buf = p
    if len(buf) >= min_chars:
        passages.append(buf)
    # Cap total passages to avoid heavy encoding
    return passages[:200]


def _retrieve_relevant_passages(question_fr: str, raw_text_fr: str, top_k: int = 5) -> list:
    """Use embeddings (if available) to retrieve top-k relevant passages; fallback to keyword filter."""
    passages = _split_into_passages(raw_text_fr)
    if not passages:
        return [raw_text_fr[:1200]]

    try:
        # Embed question and passages (embedder is global, CPU)
        q_emb = embedder.encode(question_fr, convert_to_tensor=True)
        p_embs = embedder.encode(passages, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, p_embs).cpu().numpy().flatten()
        idxs = np.argsort(scores)[::-1][:top_k]
        return [passages[i] for i in idxs]
    except Exception:
        # Fallback: simple keyword filter
        q_tokens = [t for t in re.findall(r"\w+", question_fr.lower()) if len(t) > 2]
        scored = []
        for p in passages:
            text_l = p.lower()
            hit = sum(1 for t in q_tokens if t in text_l)
            scored.append((hit, len(p), p))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [p for _,__,p in scored[:top_k] if _ > 0] or passages[:top_k]


def _clean_generated_answer(ans: str) -> str:
    """Remove artifacts and keep a concise French answer."""
    a = ans
    # Remove prompt echoes
    a = re.sub(r"(?is)^.*?réponse\s*:\s*", "", a).strip()
    # Remove artifacts like @xmath, code-ish tokens
    a = re.sub(r"@xmath\d+", "", a)
    a = re.sub(r"\(.*?\)\s*\*?", lambda m: "" if len(m.group(0)) > 80 else m.group(0), a)
    a = re.sub(r"[\[\]{}<>]{1,}|_{2,}|\*{2,}|`{1,}", " ", a)
    # Collapse repeats and spaces
    a = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", r"\1", a)
    a = re.sub(r"\s+", " ", a).strip()
    # Keep it within a reasonable size
    if len(a) > 600:
        a = a[:600].rsplit('.', 1)[0].strip() or a[:300]
    return a


def _answer_confidence(question_fr: str, answer_fr: str) -> float:
    """Rough confidence using semantic similarity and simple heuristics."""
    try:
        q = question_fr.strip()
        a = answer_fr.strip()
        if not a:
            return 0.0
        # Embedding similarity
        q_emb = embedder.encode(q[:400], convert_to_tensor=True)
        a_emb = embedder.encode(a[:600], convert_to_tensor=True)
        sim = float(util.pytorch_cos_sim(q_emb, a_emb).item())
    except Exception:
        sim = 0.0
    # Penalize gibberish patterns
    gibberish_penalty = 0.0
    if re.search(r"[A-Za-z]\d{3,}|\d{3,}[A-Za-z]", answer_fr):
        gibberish_penalty += 0.1
    if answer_fr.count("?") > 1:
        gibberish_penalty += 0.1
    if len(answer_fr.split()) < 4:
        gibberish_penalty += 0.2
    return max(0.0, sim - gibberish_penalty)


def answer_question_with_teacher_french(raw_text: str, question: str, max_length: int = 256) -> str:
    """Answer a French question using teacher model with lightweight retrieval and safer decoding."""
    load_teacher_model()

    # Build compact, relevant context (French)
    top_passages = _retrieve_relevant_passages(question, raw_text, top_k=5)
    context = "\n\n".join(top_passages)
    # Hard cap context length to avoid long inputs
    context = context[:4000]

    instruction = (
        "Vous êtes un assistant concis et factuel. Répondez STRICTEMENT en français, en une ou deux phrases, "
        "uniquement à partir du contenu fourni. Si l'information n'est pas présente dans le contexte, dites : "
        "\"Je ne trouve pas d’information dans le document.\""
    )

    prompt = (
        f"{instruction}\n\nQuestion: {question}\n\nContexte:\n{context}\n\nRéponse:" 
    )

    # Tokenize without unnecessary padding
    inputs = teacher_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=False
    ).to(device)

    # Safer generation settings
    gen_kwargs = dict(
        max_new_tokens=min(180, max_length),
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_beams=3,
        no_repeat_ngram_size=4,
        repetition_penalty=1.25,
        length_penalty=1.0,
        early_stopping=True
    )

    with torch.no_grad():
        output_ids = teacher_model.generate(**inputs, **gen_kwargs)

    answer = teacher_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Keep only the part after "Réponse:" if present
    if "Réponse:" in answer:
        answer = answer.split("Réponse:")[-1]

    answer = _clean_generated_answer(answer)

    # Confidence check and fallback
    conf = _answer_confidence(question, answer)
    if conf < 0.25:
        return "Je ne trouve pas d’information dans le document."

    return clean_french_text(answer)