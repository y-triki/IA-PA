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
        ("models/lid.176.bin", "backend/models/summary/lid.176.bin")
    ]

    for s3_key, local_path in files_to_download:
        download_from_s3(bucket, s3_key, local_path)
