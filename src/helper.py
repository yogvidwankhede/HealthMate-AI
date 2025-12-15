"""
Helper functions for HealthMate-AI
OPTION 1: Using Hugging Face Hub (Recommended)
"""
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import os


class CustomEnsembleEmbeddings(Embeddings):
    """
    Custom embedding class using your fine-tuned ensemble model from Hugging Face Hub.
    This is the 3-fold ensemble model with +18.31% improvement over baseline.
    Compatible with LangChain.
    """

    def __init__(self, model_name: str):
        """
        Initialize with model from Hugging Face Hub
        
        Args:
            model_name: Hugging Face model name (e.g., 'username/model-name')
        """
        print(f"📥 Loading fine-tuned embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
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
    Download and return your custom fine-tuned embeddings model from Hugging Face Hub.
    
    SETUP REQUIRED:
    1. Upload your model to Hugging Face Hub (see CUSTOM_EMBEDDING_GUIDE.md)
    2. Set environment variable: HF_MODEL_NAME="your-username/your-model-name"
    
    Falls back to base model if custom model is not available.
    """

    # Try to get custom model name from environment variable
    custom_model = os.environ.get("HF_MODEL_NAME")

    if custom_model:
        print(f"🎯 Using custom fine-tuned model: {custom_model}")
        try:
            embeddings = CustomEnsembleEmbeddings(model_name=custom_model)
            return embeddings
        except Exception as e:
            print(f"⚠️ Failed to load custom model: {e}")
            print("⚠️ Falling back to base model...")

    # Fallback to base model
    print("ℹ️ Using base model: sentence-transformers/all-MiniLM-L6-v2")
    print("💡 To use your fine-tuned model, set HF_MODEL_NAME environment variable")
    embeddings = CustomEnsembleEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
