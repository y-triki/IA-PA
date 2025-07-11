# backend/utils/load_from_s3.py

import os
import boto3

def download_from_s3(bucket: str, key: str, local_path: str):
    if not os.path.exists(local_path):
        print(f"[INFO] Téléchargement de {key} depuis S3...")
        s3 = boto3.client("s3")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
    else:
        print(f"[INFO] Fichier déjà présent : {local_path}")

def download_all_models():
    bucket = "assistant-smart-ai-models"
    files_to_download = [
        # Question
        ("models/question/vocab.pt", "backend/models/question/vocab.pt"),
        ("models/question/question_generator.pt", "backend/models/question/question_generator.pt"),

        # Summary
        ("models/summary/best_model.pth", "backend/models/summary/best_model.pth"),
        ("models/summary/merges.txt", "backend/models/summary/merges.txt"),
        ("models/summary/vocab.json", "backend/models/summary/vocab.json"),
        ("models/lid.176.bin", "backend/models/summary/lid.176.bin")
    ]

    for s3_key, local_path in files_to_download:
        download_from_s3(bucket, s3_key, local_path)
