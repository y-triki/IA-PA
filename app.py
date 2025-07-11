from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import sys

# Ajout du chemin du backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.pipelines.generation_pipeline import summarize_pdf
from backend.pipelines.generate_questions import generate_open_questions
from backend.pipelines.rag_pipeline import retrieve_relevant
from backend.pipelines.save_json import save_results
from werkzeug.utils import safe_join

from backend.utils.load_from_s3 import download_all_models
download_all_models()

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    question = request.form.get("question", "")

    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        try:
            with open(path, "rb") as pdf_file:
                result = summarize_pdf(pdf_file.read())
        except Exception as e:
            return render_template("error.html", error_message=f"Erreur pendant le traitement du fichier : {e}")

        if not result or not isinstance(result, dict) or "summary" not in result:
            return render_template("error.html", error_message="Échec du traitement du fichier.")

        summary = result.get("summary", "")
        session["filename"] = file.filename
        session["summary"] = summary

        save_results({
            "filename": file.filename,
            "summary": summary,
            "question": question,
            "answer": "",
            "quiz": []
        }, output_dir="shared/exports")

        return render_template("result.html",
                               summary=summary,
                               answer="",
                               quiz=[],
                               question=question,
                               filename=file.filename)

    return redirect(url_for("index"))

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz_route():
    filename = session.get("filename")
    path = os.path.join(UPLOAD_FOLDER, filename) if filename else None

    if not filename or not os.path.exists(path):
        return "Aucun fichier trouvé."

    try:
        # Génére les questions ouvertes à partir du fichier
        questions = generate_open_questions(path)

        if not questions:
            return "Aucune question générée. Veuillez vérifier le contenu du fichier."

        # Format compatible avec HTML : liste de questions simples
        quiz = [{"question": q["question"] if isinstance(q, dict) and "question" in q else q} for q in questions]

        session["quiz"] = quiz

        return render_template("quiz.html",
                               summary=session.get("summary", ""),
                               quiz=quiz,
                               filename=filename)

    except Exception as e:
        # En cas d'erreur, afficher dans le navigateur (temporairement)
        return f"Erreur lors de la génération des questions : {e}"


@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    submitted_answers = {
        key: value for key, value in request.form.items() if key.startswith("question_")
    }
    return render_template("qcm.html",
                           summary=session.get("summary", ""),
                           quiz=session.get("quiz", []),
                           answers=submitted_answers)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    file_path = safe_join(UPLOAD_FOLDER, filename)
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(file_path))

@app.route("/ask_question", methods=["POST"])
def ask_question():
    filename = request.form.get("filename")
    question = request.form.get("question", "")
    if not filename or not question:
        return redirect(url_for("index"))

    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return redirect(url_for("index"))

    with open(path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
        summary_result = summarize_pdf(pdf_bytes)

    summary = summary_result.get("summary", "") if isinstance(summary_result, dict) else ""
    chunks = [summary[i:i + 500] for i in range(0, len(summary), 500)]
    answer = retrieve_relevant(chunks, question)

    return render_template(
        "result.html",
        summary=summary,
        answer=answer,
        quiz=[],
        question=question,
        filename=filename
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)