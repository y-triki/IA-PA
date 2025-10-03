from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import sys
import re

# Limit parallelism to avoid high memory/sem_wait leaks on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from backend.pipelines.pdf_pipeline import PDFProcessor

# Ajout du chemin du backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.pipelines.generation_pipeline import summarize_pdf
from backend.pipelines.generate_questions import generate_open_questions
from backend.pipelines.rag_pipeline import retrieve_relevant
from backend.pipelines.save_json import save_results
from werkzeug.utils import safe_join

from backend.utils.load_from_s3 import download_all_models
download_all_models()
from backend.utils.feedback_manager import FeedbackManager
from backend.utils.finetune_from_feedback import build_all_feedback_datasets

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

            summary = result.get("summary", "")
            chunks_count = result.get("chunks_count", 0)
            model_used = result.get("model_used", "student")

            if not summary:
                return render_template("error.html", error_message="Échec du traitement du fichier.")

            # Stocker toutes les infos dans la session
            session["filename"] = file.filename
            session["summary"] = summary
            session["question"] = question
            session["answer"] = ""
            session["quiz"] = []
            session["chunks_count"] = chunks_count
            session["model_used"] = model_used

            # Sauvegarde
            save_results({
                "filename": file.filename,
                "summary": summary,
                "question": question,
                "answer": "",
                "quiz": [],
                "chunks_count": chunks_count,
                "model_used": model_used
            }, output_dir="shared/exports")

            # REDIRECTION au lieu de render_template
            return redirect(url_for("result"))

        except Exception as e:
            return render_template("error.html", error_message=f"Erreur pendant le traitement du fichier : {e}")

    return redirect(url_for("index"))


@app.route("/result")
def result():
    # Retrieve one-time feedback message if available
    feedback_message = session.pop("feedback_message", None)
    return render_template(
        "result.html",
        summary=session.get("summary", ""),
        question=session.get("question", ""),
        answer=session.get("answer", ""),
        quiz=session.get("quiz", []),
        filename=session.get("filename", ""),
        feedback_message=feedback_message
    )


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
    return render_template("quiz.html",
                           summary=session.get("summary", ""),
                           quiz=session.get("quiz", []),
                           answers=submitted_answers)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    file_path = safe_join(UPLOAD_FOLDER, filename)
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(file_path))


# app.py (updated /ask_question route)
@app.route("/ask_question", methods=["POST"])
def ask_question():
    filename = request.form.get("filename")
    question = request.form.get("question", "")
    if not filename or not question:
        return redirect(url_for("index"))

    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return redirect(url_for("index"))

    try:
        # Create text extractor (no tokenizer needed)
        text_extractor = PDFProcessor(tokenizer_dir=None)

        # Extract text from PDF
        with open(path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            raw_text = text_extractor.extract_text(pdf_bytes)

        # Use teacher model for French prompt-based QA
        from backend.pipelines.generation_pipeline import answer_question_with_teacher_french
        answer = answer_question_with_teacher_french(
            raw_text=raw_text,
            question=question
        )

        # Persist latest Q&A in session so result page and feedback flow can reuse it
        session["question"] = question
        session["answer"] = answer
        session["filename"] = filename

        return render_template(
            "result.html",
            summary=session.get("summary", ""),
            answer=answer,
            quiz=[],
            question=question,
            filename=filename
        )

    except Exception as e:
        return render_template("error.html",
                               error_message=f"Erreur lors du traitement de la question: {str(e)}")

@app.route("/feedback", methods=["POST"])
def feedback():
    """Collect user feedback on Q&A/Summary/Quiz and persist to JSON, then return to result page."""
    try:
        feedback_value = request.form.get("feedback", "").strip()
        item_type = request.form.get("item_type", "") or "qa"
        filename = request.form.get("filename", "")

        # Accept multiple payload types
        question = request.form.get("question", "")
        answer = request.form.get("answer", "")
        summary = request.form.get("summary", "")
        quiz_payload = request.form.get("quiz", "")

        # Prefer explicit answer; else use summary; else use quiz payload
        payload_answer = answer or summary or quiz_payload or ""

        fm = FeedbackManager()
        fm.append_feedback(
            question=question,
            answer=payload_answer,
            feedback=feedback_value,
            meta={
                "filename": filename,
                "path": os.path.join(UPLOAD_FOLDER, filename) if filename else None,
                "item_type": item_type,
            }
        )
        print(f"[FEEDBACK] Saved: type={item_type} feedback={feedback_value} filename={filename}")
        session["feedback_message"] = "Merci pour votre retour !"
    except Exception as e:
        print(f"[FEEDBACK][ERROR] {e}")
        session["feedback_message"] = f"Erreur lors de l'enregistrement du feedback : {e}"

    # Ensure result page has data even if session was not populated earlier
    return redirect(url_for("result"))


@app.route("/export_feedback_datasets", methods=["GET"]) 
def export_feedback_datasets():
    try:
        info = build_all_feedback_datasets()
        # Persist a short confirmation for the UI
        session["feedback_message"] = (
            f"Datasets exportés: summarization={info.get('summarization_count')} items, "
            f"qa={info.get('qa_count')} items. Dossier: {info.get('output_dir')}"
        )
        # Redirect to result page to show the banner with message
        return redirect(url_for('result'))
    except Exception as e:
        return render_template("error.html", error_message=f"Erreur export dataset: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)