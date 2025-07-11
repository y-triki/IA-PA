<<<<<<< HEAD
import json
from datetime import datetime
"""
def save_results(data, output_dir="data"):
    filename = f"{output_dir}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
"""
import os
import json
from datetime import datetime

def save_results(data, output_dir="shared/exports"):
    # Obtenir le chemin absolu du dossier output à partir du script courant
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "..", output_dir)

    # Créer le dossier s'il n'existe pas
    os.makedirs(output_path, exist_ok=True)

    # Nom de fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_path, f"session_{timestamp}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
=======
import json
from datetime import datetime
"""
def save_results(data, output_dir="data"):
    filename = f"{output_dir}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
"""
import os
import json
from datetime import datetime

def save_results(data, output_dir="shared/exports"):
    # Obtenir le chemin absolu du dossier output à partir du script courant
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "..", output_dir)

    # Créer le dossier s'il n'existe pas
    os.makedirs(output_path, exist_ok=True)

    # Nom de fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_path, f"session_{timestamp}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
>>>>>>> ecbe693 (Mise à jour depuis EC2 : dernières modifs locales)
