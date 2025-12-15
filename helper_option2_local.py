"""
Helper functions for HealthMate-AI
OPTION 2: Using Local Model Files (Bundled with App)
"""
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from pathlib import Path


class CustomEnsembleEmbeddings(Embeddings):
    """
    Custom embedding class using your fine-tuned ensemble model from local path.
    This is the 3-fold ensemble model with +18.31% improvement over baseline.
    Compatible with LangChain.
    """

    def __init__(self, model_path: str):
        """
        Initialize with local model path
        
        Args:
            model_path: Path to your ensemble model directory
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        print(f"📥 Loading fine-tuned embedding model from: {model_path}")
        self.model = SentenceTransformer(model_path)
        print("✅ Custom embedding model loaded successfully")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed search docs (used for documents in RAG systems).
        """
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query text (used for user queries).
        """
        embedding = self.model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False)[0]
        return embedding.tolist()


def download_hugging_face_embeddings():
    """
    Load your custom fine-tuned embeddings model from local directory.
    
    SETUP REQUIRED:
    1. Copy your model to: models/l6v2_3fold_ensemble/
    2. Ensure the model directory contains:
       - config.json
       - pytorch_model.bin (or model.safetensors)
       - tokenizer files
       - sentence_bert_config.json
    
    Directory structure:
        deployment/
        ├── app.py
        ├── models/
        │   └── l6v2_3fold_ensemble/
        │       ├── config.json
        │       ├── pytorch_model.bin
        │       ├── sentence_bert_config.json
        │       └── ...
        └── src/
            └── helper.py (this file)
    
    Falls back to base model if custom model is not found.
    """

    # Get the directory where this file is located
    current_dir = Path(__file__).parent.parent  # Go up to deployment root

    # Define possible model paths
    model_paths = [
        current_dir / "models" / "l6v2_3fold_ensemble",
        current_dir / "models" / "l6v2_best2_ensemble",
        current_dir / "models" / "custom_embeddings",
    ]

    # Try to find and load custom model
    for model_path in model_paths:
        if model_path.exists():
            print(f"🎯 Found custom fine-tuned model at: {model_path}")
            try:
                embeddings = CustomEnsembleEmbeddings(
                    model_path=str(model_path))
                return embeddings
            except Exception as e:
                print(f"⚠️ Failed to load model from {model_path}: {e}")
                continue

    # Fallback to base model
    print("ℹ️ Custom model not found. Using base model: sentence-transformers/all-MiniLM-L6-v2")
    print("💡 To use your fine-tuned model, copy it to: models/l6v2_3fold_ensemble/")
    print("📖 See CUSTOM_EMBEDDING_GUIDE.md for instructions")

    # Use base model as fallback
    from sentence_transformers import SentenceTransformer
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Wrap in our custom class for consistency
    class BaseEmbeddings(Embeddings):
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            embeddings = self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()

        def embed_query(self, text: str) -> list[float]:
            embedding = self.model.encode(
                [text], convert_to_numpy=True, show_progress_bar=False)[0]
            return embedding.tolist()

    return BaseEmbeddings(base_model)
