<<<<<<< HEAD
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def structure_summary_by_theme(summary_text, n_clusters=5):
    sentences = sent_tokenize(summary_text)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    # Clustering (simple KMeans, ajustable)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Rassemble les phrases par thème
    clustered = {}
    for label, sentence in zip(labels, sentences):
        clustered.setdefault(label, []).append(sentence)

    # Génère le résumé structuré
    structured_summary = ""
    for cluster_id, cluster_sentences in sorted(clustered.items()):
        structured_summary += f"\n Thème {cluster_id + 1}:\n"
        structured_summary += " ".join(cluster_sentences) + "\n"

    return structured_summary.strip()
=======
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def structure_summary_by_theme(summary_text, n_clusters=5):
    sentences = sent_tokenize(summary_text)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    # Clustering (simple KMeans, ajustable)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Rassemble les phrases par thème
    clustered = {}
    for label, sentence in zip(labels, sentences):
        clustered.setdefault(label, []).append(sentence)

    # Génère le résumé structuré
    structured_summary = ""
    for cluster_id, cluster_sentences in sorted(clustered.items()):
        structured_summary += f"\n Thème {cluster_id + 1}:\n"
        structured_summary += " ".join(cluster_sentences) + "\n"

    return structured_summary.strip()
>>>>>>> ecbe693 (Mise à jour depuis EC2 : dernières modifs locales)
