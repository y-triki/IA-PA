<<<<<<< HEAD
from sentence_transformers import SentenceTransformer, util

# Charge le modèle une fois
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_relevant(text_chunks, question, max_chunks=15):
    """
    Retourne le chunk le plus pertinent parmi les max_chunks premiers du texte.
    max_chunks réduit le temps de calcul.
    """
    # Limite le nombre de chunks à analyser
    text_chunks = text_chunks[:max_chunks]

    # Encodage
    embeddings = model.encode(text_chunks, convert_to_tensor=True, show_progress_bar=False)
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Similarité cosinus
    scores = util.cos_sim(question_embedding, embeddings)
    best_idx = scores.argmax()
    return text_chunks[best_idx]
=======
from sentence_transformers import SentenceTransformer, util

# Charge le modèle une fois
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_relevant(text_chunks, question, max_chunks=15):
    """
    Retourne le chunk le plus pertinent parmi les max_chunks premiers du texte.
    max_chunks réduit le temps de calcul.
    """
    # Limite le nombre de chunks à analyser
    text_chunks = text_chunks[:max_chunks]

    # Encodage
    embeddings = model.encode(text_chunks, convert_to_tensor=True, show_progress_bar=False)
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Similarité cosinus
    scores = util.cos_sim(question_embedding, embeddings)
    best_idx = scores.argmax()
    return text_chunks[best_idx]
>>>>>>> ecbe693 (Mise à jour depuis EC2 : dernières modifs locales)
