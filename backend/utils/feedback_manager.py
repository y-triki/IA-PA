import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional


DEFAULT_FEEDBACK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'shared', 'feedback.json')


class FeedbackManager:
    """
    Manage user feedback stored in a JSON file.
    Each feedback record contains: question, answer, feedback ("up"|"down"), and metadata.
    """

    def __init__(self, json_path: str = DEFAULT_FEEDBACK_PATH):
        self.json_path = json_path
        self._ensure_file()

    def _ensure_file(self) -> None:
        directory = os.path.dirname(self.json_path)
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def read_all(self) -> List[Dict]:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def append_feedback(self, question: str, answer: str, feedback: str, meta: Dict = None) -> Dict:
        if feedback not in {"up", "down"}:
            raise ValueError("feedback must be either 'up' or 'down'")

        record = {
            "question": (question or '').strip(),
            "answer": (answer or '').strip(),
            "feedback": feedback,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
        if meta:
            record["meta"] = meta

        all_items = self.read_all()
        all_items.append(record)
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        return record


def prepare_dataset_from_feedback(json_path: str = DEFAULT_FEEDBACK_PATH) -> List[Tuple[str, str]]:
    """
    Prepare a simple (input, output) dataset from feedback JSON for Q&A fine-tuning.
    By default, only 'up' (useful) feedback items are included.
    Returns: List of (question, answer) pairs.
    """
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pairs: List[Tuple[str, str]] = []
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            if item.get('feedback') != 'up':
                # skip negative feedback by default
                continue
            q = (item.get('question') or '').strip()
            a = (item.get('answer') or '').strip()
            if q and a:
                pairs.append((q, a))
        return pairs
    except (json.JSONDecodeError, OSError):
        return []


def prepare_summarization_dataset_from_feedback(
    json_path: str = DEFAULT_FEEDBACK_PATH,
    output_path: Optional[str] = None,
    max_chars: int = 12000
) -> List[Dict]:
    """
    Build a supervised summarization dataset from positive summary feedback entries.
    For each item with meta.item_type == 'summary' and feedback == 'up', we pair the
    extracted PDF text (truncated) with the approved summary.

    Returns the list of dicts and optionally writes JSONL at output_path with fields:
    {"source_text": ..., "target_summary": ..., "filename": ..., "timestamp": ...}
    """
    try:
        if not os.path.exists(json_path):
            return []
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    items: List[Dict] = []

    # Lazy import to avoid heavy deps at import time
    try:
        from backend.pipelines.pdf_pipeline import PDFProcessor
    except Exception:
        PDFProcessor = None  # type: ignore

    processor = PDFProcessor(tokenizer_dir=None) if PDFProcessor else None  # extractor only

    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get('feedback') != 'up':
            continue
        meta = entry.get('meta') or {}
        if meta.get('item_type') != 'summary':
            continue
        summary_text = (entry.get('answer') or '').strip()
        pdf_path = (meta.get('path') or '').strip()
        filename = meta.get('filename') or os.path.basename(pdf_path) or None
        if not summary_text or not pdf_path or not os.path.exists(pdf_path):
            continue
        try:
            if processor:
                with open(pdf_path, 'rb') as fpdf:
                    full_text = processor.extract_text(fpdf.read())
            else:
                full_text = ''
        except Exception:
            full_text = ''
        source_text = (full_text or '')[:max_chars]
        if not source_text:
            continue
        items.append({
            "source_text": source_text,
            "target_summary": summary_text,
            "filename": filename,
            "timestamp": entry.get('timestamp')
        })

    if output_path and items:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as fout:
            for it in items:
                fout.write(json.dumps(it, ensure_ascii=False) + "\n")

    return items


def prepare_qa_dataset_from_feedback(
    json_path: str = DEFAULT_FEEDBACK_PATH,
    output_path: Optional[str] = None
) -> List[Dict]:
    """
    Export positive Q&A feedback as a JSONL dataset with fields:
    {"question": ..., "answer": ..., "filename": ..., "timestamp": ...}
    """
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    items: List[Dict] = []
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get('feedback') != 'up':
            continue
        q = (entry.get('question') or '').strip()
        a = (entry.get('answer') or '').strip()
        if not q or not a:
            continue
        meta = entry.get('meta') or {}
        items.append({
            "question": q,
            "answer": a,
            "filename": meta.get('filename'),
            "timestamp": entry.get('timestamp')
        })

    if output_path and items:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as fout:
            for it in items:
                fout.write(json.dumps(it, ensure_ascii=False) + "\n")

    return items
