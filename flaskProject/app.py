from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import sys

# Ajout du chemin du backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.pipelines.generation_pipeline import generate_quiz, summarize_pdf
from backend.pipelines.rag_pipeline import retrieve_relevant
from backend.pipelines.save_json import save_results
from werkzeug.utils import safe_join

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Dossier des uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    """Page d'accueil."""
    return render_template("index.html")



@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload du fichier, traitement du PDF, génération du résumé."""
    file = request.files.get("file")
    question = request.form.get("question", "")

    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # ⚡️ Appel DIRECT à summarize_pdf
        with open(path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            result = summarize_pdf(pdf_bytes)

        summary = result.get("summary", "")
        answer = ""  # À implémenter si nécessaire
        quiz = []    # Idem si nécessaire

        # Sauvegarde des résultats
        save_results({
            "filename": file.filename,
            "summary": summary,
            "question": question,
            "answer": answer,
            "quiz": quiz
        }, output_dir="shared/exports")  # Adapter le chemin si nécessaire

        return render_template("result.html",
                               summary=summary,
                               answer=answer,
                               quiz=quiz,
                               question=question,
                               filename=file.filename)

    return redirect(url_for("index"))


@app.route("/generate_quiz", methods=["POST"])
def generate_quiz_route():
    """Génération du QCM à partir du résumé."""
    summary = request.form.get("summary", "")
    question = request.form.get("question", "")
    quiz = generate_quiz(summary) if summary else []
    answer = retrieve_relevant([summary], question) if question else ""
    session["summary"] = summary
    session["quiz"] = quiz
    return render_template("quiz.html",
                           summary=summary,
                           quiz=quiz,
                           answer=answer,
                           question=question)


@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    """Récupération des réponses du quiz soumis."""
    submitted_answers = {
        key: value for key, value in request.form.items() if key.startswith("question_")
    }
    return render_template("qcm.html",
                           summary=session.get("summary", ""),
                           quiz=session.get("quiz", []),
                           answers=submitted_answers)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Accès direct aux fichiers uploadés."""
    file_path = safe_join(UPLOAD_FOLDER, filename)
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(file_path))


@app.route("/ask_question", methods=["POST"])
def ask_question():
    """Répond à une question à propos du document déjà traité."""
    filename = request.form.get("filename")
    question = request.form.get("question", "")
    if not filename or not question:
        return redirect(url_for("index"))

    path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(path):
        return redirect(url_for("index"))

    # Appel à la NOUVELLE méthode
    with open(path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
        summary = summarize_pdf(pdf_bytes)

    chunks = [summary[i:i + 300] for i in range(0, len(summary), 300)]
    answer = retrieve_relevant(chunks, question)

    return render_template(
        "result.html",
        summary=summary,
        answer=answer,
        quiz=[],  # Si pas nécessaire, laisser vide
        question=question,
        filename=filename
    )


if __name__ == "__main__":
    app.run(debug=True)

