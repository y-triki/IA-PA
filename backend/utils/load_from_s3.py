# backend/utils/load_from_s3.py

from __future__ import annotations
import os
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Allow overriding via environment variables without breaking current behavior
DEFAULT_BUCKET = os.getenv("S3_MODELS_BUCKET", "assistant-smart-ai-models")
MODELS_MANIFEST = os.getenv("MODELS_MANIFEST_JSON")  # JSON content or a path to a JSON file
PREFIX_MANIFEST = os.getenv("MODELS_PREFIXES_JSON")  # JSON content or a path to a JSON file
FORCE_DOWNLOAD = os.getenv("FORCE_DOWNLOAD", "0") in {"1", "true", "True"}


def _load_json_config(value: str | None):
    """Load JSON either from a file path or from a JSON string. Returns parsed object or None."""
    if not value:
        return None
    try:
        if os.path.exists(value):
            with open(value, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(value)
    except Exception as e:
        print(f"[WARN] Impossible de charger la configuration JSON: {e}")
        return None


def _get_s3_client():
    return boto3.client("s3")


def download_from_s3(bucket: str, key: str, local_path: str):
    """Download a single S3 object to a local file path, creating parent dirs as needed."""
    try:
        if os.path.isdir(local_path):
            raise ValueError(f"local_path est un dossier, chemin de fichier requis: {local_path}")
        if not os.path.exists(local_path) or FORCE_DOWNLOAD:
            print(f"[INFO] Téléchargement de s3://{bucket}/{key} vers {local_path}...")
            s3 = _get_s3_client()
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            s3.download_file(bucket, key, local_path)
        else:
            print(f"[INFO] Fichier déjà présent : {local_path}")
    except NoCredentialsError:
        print("[ERREUR] Identifiants AWS introuvables. Vérifiez vos variables d'environnement/credentials.")
        raise
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        print(f"[ERREUR] Échec du téléchargement {key} ({code}): {e}")
        raise


def download_prefix(bucket: str, prefix: str, dest_dir: str):
    """Recursively download all objects under an S3 prefix into a local directory, preserving the suffix layout."""
    print(f"[INFO] Synchronisation du préfixe s3://{bucket}/{prefix} -> {dest_dir}")
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    os.makedirs(dest_dir, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):]
            local_path = os.path.join(dest_dir, rel)
            download_from_s3(bucket, key, local_path)


def _default_files_to_download():
    return [
        # Question
        ("models/question/vocab.pt", "backend/models/question/vocab.pt"),
        ("models/question/question_generator.pt", "backend/models/question/question_generator.pt"),

        # Student model
        ("models/student_model/final_student_model.pth", "backend/models/student_model/final_student_model.pth"),
        ("models/student_model/merges.txt", "backend/models/student_model/merges.txt"),
        ("models/student_model/vocab.json", "backend/models/student_model/vocab.json"),

        # Teacher model
        ("models/teacher_model/config.json", "backend/models/teacher_model/config.json"),
        ("models/teacher_model/merges.txt", "backend/models/teacher_model/merges.txt"),
        ("models/teacher_model/rng_state.pth", "backend/models/teacher_model/rng_state.pth"),
        ("models/teacher_model/scheduler.pt", "backend/models/teacher_model/scheduler.pt"),
        ("models/teacher_model/vocab.json", "backend/models/teacher_model/vocab.json"),
        ("models/teacher_model/training_args.bin", "backend/models/teacher_model/training_args.bin"),
        ("models/teacher_model/trainer_state.json", "backend/models/teacher_model/trainer_state.json"),
        ("models/teacher_model/tokenizer_config.json", "backend/models/teacher_model/tokenizer_config.json"),
        ("models/teacher_model/special_tokens_map.json", "backend/models/teacher_model/special_tokens_map.json"),
        ("models/teacher_model/model.safetensors", "backend/models/teacher_model/model.safetensors"),
        ("models/teacher_model/generation_config.json", "backend/models/teacher_model/generation_config.json"),

        # Summary
        ("models/summary/best_model.pth", "backend/models/summary/best_model.pth"),
        ("models/summary/merges.txt", "backend/models/summary/merges.txt"),
        ("models/summary/vocab.json", "backend/models/summary/vocab.json"),
        ("models/lid.176.bin", "backend/models/summary/lid.176.bin"),
    ]


def download_all_models():
    bucket = DEFAULT_BUCKET

    # 1) Explicit files via manifest or default list
    manifest = _load_json_config(MODELS_MANIFEST)
    if manifest and isinstance(manifest, dict) and "files" in manifest:
        files_to_download = [(f["s3_key"], f["local_path"]) for f in manifest["files"]]
    else:
        files_to_download = _default_files_to_download()

    for s3_key, local_path in files_to_download:
        download_from_s3(bucket, s3_key, local_path)

    # 2) Optional prefix syncs for training/retraining checkpoints
    prefixes_cfg = _load_json_config(PREFIX_MANIFEST)
    if prefixes_cfg and isinstance(prefixes_cfg, dict) and "prefixes" in prefixes_cfg:
        for entry in prefixes_cfg["prefixes"]:
            prefix = entry["prefix"]
            dest_dir = entry["dest_dir"]
            download_prefix(bucket, prefix, dest_dir)
