from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from typing import Optional
import numpy as np
import os
import shutil
import requests
import zipfile
import tempfile

# Constantes
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
MODELS_DIR = "./models"
DEFAULT_MODEL_NAME = os.path.join(MODELS_DIR, "esti-rag-ft")

app = FastAPI()

# Singleton model
default_model: SentenceTransformer = None
current_model_path: str = DEFAULT_MODEL_NAME

def load_model(model_path: str) -> SentenceTransformer:
    print(f"🧠 Chargement SentenceTransformer depuis : {model_path}")
    return SentenceTransformer(model_path, device=DEVICE)

# Constantes
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
MODELS_DIR = "./models"
DEFAULT_MODEL_NAME = os.path.join(MODELS_DIR, "esti-rag-ft")

# Globales
default_model: SentenceTransformer = None
current_model_path: str = DEFAULT_MODEL_NAME

def load_model_by_url(url: str, version: str) -> SentenceTransformer:
    """
    Télécharge un modèle zippé depuis une URL, l'extrait dans ./models/{version},
    puis le charge avec SentenceTransformer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "model.zip")

        # Téléchargement
        print(f"⬇️ Téléchargement du modèle depuis {url}")
        r = requests.get(url)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)

        # Extraction
        target_dir = os.path.join(MODELS_DIR, version)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        print(f"✅ Modèle extrait dans {target_dir}")

        # Chargement
        model = SentenceTransformer(target_dir, device=DEVICE)
        return model

@app.get("/")
def greet_json():
    return {"Hello": "World!"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global default_model, current_model_path
    if os.path.exists(DEFAULT_MODEL_NAME):
        default_model = load_model(DEFAULT_MODEL_NAME)
        current_model_path = DEFAULT_MODEL_NAME
    else:
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"⚠️ Modèle {DEFAULT_MODEL_NAME} introuvable. Utilisation du modèle par défaut HuggingFace : {fallback_model}")
        default_model = load_model(fallback_model)
        current_model_path = fallback_model
    yield

app = FastAPI(lifespan=lifespan)

class TextRequest(BaseModel):
    texts: list[str]
    model: Optional[str] = ""

@app.post("/embed")
def get_embedding(req: TextRequest):
    try:
        if req.model:
            model = load_model(req.model)
            embeddings = model.encode(req.texts, convert_to_numpy=True, normalize_embeddings=True)
        else:
            embeddings = default_model.encode(req.texts, convert_to_numpy=True, normalize_embeddings=True)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reload-model")
def reload_model(version: Optional[str] = Query(default=None), url: Optional[str] = Query(default=None)):
    """
    Recharger dynamiquement le modèle par défaut.
    - Si 'version' est fourni, copie models/{version} → models/esti-rag-ft
    - Si 'url' est une archive zip → télécharge et extrait
    - Si 'url' est une URL Hugging Face → télécharge le modèle via snapshot_download
    """
    global default_model, current_model_path

    try:
        if url and version:
            print(f"🔁 Téléchargement du modèle depuis URL : {url} → models/esti-rag-ft")

            extract_dir = os.path.join(MODELS_DIR, "esti-rag-ft")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)

            if "huggingface.co" in url:
                # ✅ Cas Hugging Face → utiliser snapshot_download
                parts = url.split("huggingface.co/")[1].split("/")
                if len(parts) >= 2:
                    repo_id = f"{parts[0]}/{parts[1]}"
                    subfolder = "/".join(parts[4:]) if "tree" in parts else ""

                    with tempfile.TemporaryDirectory() as tmp_download_dir:
                        model_path = snapshot_download(
                            repo_id=repo_id,
                            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
                            local_dir=tmp_download_dir,
                            local_dir_use_symlinks=False
                        )

                        sub_model_path = os.path.join(model_path, subfolder) if subfolder else model_path

                        for item in os.listdir(sub_model_path):
                            s = os.path.join(sub_model_path, item)
                            d = os.path.join(extract_dir, item)
                            if os.path.isdir(s):
                                shutil.copytree(s, d)
                            else:
                                shutil.copy2(s, d)
                else:
                    raise HTTPException(status_code=400, detail="❌ URL Hugging Face invalide.")
            else:
                # 📦 Cas archive ZIP classique
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "model.zip")
                    r = requests.get(url)
                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        f.write(r.content)

                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                    except zipfile.BadZipFile:
                        raise HTTPException(status_code=400, detail="❌ Le fichier téléchargé n’est pas une archive ZIP valide.")

            # ✅ Charger le modèle
            default_model = load_model(extract_dir)
            current_model_path = extract_dir
            return {"status": "success", "model_path": extract_dir}

        elif version:
            source_path = os.path.join(MODELS_DIR, version)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Modèle {version} introuvable à {source_path}")

            if os.path.exists(DEFAULT_MODEL_NAME):
                shutil.rmtree(DEFAULT_MODEL_NAME)

            shutil.copytree(src=source_path, dst=DEFAULT_MODEL_NAME)
            default_model = load_model(DEFAULT_MODEL_NAME)
            current_model_path = DEFAULT_MODEL_NAME
            return {"status": "success", "model_path": DEFAULT_MODEL_NAME}

        else:
            raise ValueError("Il faut fournir à la fois 'version' et 'url' pour téléchargement, ou juste 'version'.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur rechargement modèle: {str(e)}")

@app.get("/status")
def status():
    return {
        "device": DEVICE,
        "model_path": current_model_path
    }


