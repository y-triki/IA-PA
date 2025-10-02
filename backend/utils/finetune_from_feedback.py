"""
Utilities to turn user feedback into training datasets and (optionally) kick off fine-tuning.

We keep in-app functionality light-weight: building JSONL datasets that you can use
with your preferred training scripts (PyTorch/Transformers).
"""
from __future__ import annotations
import os
import json
from typing import Dict, Any

from backend.utils.feedback_manager import (
    DEFAULT_FEEDBACK_PATH,
    prepare_summarization_dataset_from_feedback,
    prepare_qa_dataset_from_feedback,
)


def build_all_feedback_datasets(
    feedback_path: str = DEFAULT_FEEDBACK_PATH,
    out_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "shared", "exports", "datasets"
    ),
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """
    Build both summarization and QA datasets from feedback.json and write as JSONL files.
    Returns a dict with output paths and item counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    sum_out = os.path.join(out_dir, "feedback_summarization.jsonl")
    qa_out = os.path.join(out_dir, "feedback_qa.jsonl")

    sum_items = prepare_summarization_dataset_from_feedback(
        json_path=feedback_path,
        output_path=sum_out,
        max_chars=max_chars,
    )
    qa_items = prepare_qa_dataset_from_feedback(
        json_path=feedback_path,
        output_path=qa_out,
    )

    return {
        "summarization_dataset": sum_out if sum_items else None,
        "summarization_count": len(sum_items),
        "qa_dataset": qa_out if qa_items else None,
        "qa_count": len(qa_items),
        "output_dir": out_dir,
    }


def train_student_from_feedback(
    feedback_path: str = DEFAULT_FEEDBACK_PATH,
    epochs: int = 1,
    lr: float = 5e-6,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Placeholder for fine-tuning student model using feedback-derived datasets.
    For safety and to avoid heavy CPU/GPU in the web process, we only prepare datasets here.

    You can consume the exported JSONL files with your offline training scripts.
    Returns the dataset export info.
    """
    info = build_all_feedback_datasets(feedback_path)
    info.update({
        "note": (
            "Datasets ready. Use a separate training script to fine-tune your models. "
            "Example: load JSONL, tokenize with the student tokenizer, and train."
        ),
        "suggested_hparams": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
        },
    })
    return info
