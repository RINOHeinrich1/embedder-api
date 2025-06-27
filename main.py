from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
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
    - Si 'url' est fourni, télécharge l'archive zip et extrait dans models/esti-rag-ft
    """
    global default_model, current_model_path

    try:
        if url and version:
            print(f"🔁 Téléchargement et extraction depuis URL : {url} dans models/esti-rag-ft")

            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "model.zip")

                r = requests.get(url)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    f.write(r.content)

                extract_dir = os.path.join(MODELS_DIR, "esti-rag-ft")
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)

                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

            # Charger le nouveau modèle
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)
