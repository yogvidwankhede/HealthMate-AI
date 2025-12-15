from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# ==================================================================
#          OPTION 1: Using Local Model Path (Recommended)
# ==================================================================


class CustomEnsembleEmbeddings(Embeddings):
    """
    Custom embedding class that uses your fine-tuned ensemble model.
    Compatible with LangChain.
    """

    def __init__(self, model_path: str):
        """
        Initialize with your custom model path.
        
        Args:
            model_path: Path to your ensemble model directory
                       e.g., './l6v2_3fold_models/l6v2_3fold_ensemble'
        """
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed search docs (used for documents in RAG systems).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query text (used for user queries).
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


# ==================================================================
#                   USAGE EXAMPLES
# ==================================================================

def download_embeddings():
    """
    Download and return your custom ensemble embeddings model.
    """
    # Point to your ensemble model
    model_path = "Embedding/l6v2_3fold_models/l6v2_3fold_ensemble"

    embeddings = CustomEnsembleEmbeddings(model_path=model_path)
    return embeddings


# Initialize
embeddings = download_embeddings()

# Test it works
print("Testing custom embeddings...")
test_text = "This is a medical text"
test_embedding = embeddings.embed_query(test_text)
print(f"Embedding shape: {len(test_embedding)}")
print(f"First 5 dimensions: {test_embedding[:5]}")
print()

# ==================================================================
#    OPTION 2: If You Want to Use Different Ensemble Models
# ==================================================================


def download_embeddings_choice(model_choice: str = "best_3_fold"):
    """
    Download embeddings with choice of model.
    
    Args:
        model_choice: 'best_3_fold', 'best_2', or 'finetuned'
    """

    model_paths = {
        "best_3_fold": "Embedding/l6v2_3fold_models/l6v2_3fold_ensemble",
        "best_2": "Embedding/l6v2_3fold_models/l6v2_best2_ensemble",
        "finetuned": "Embedding/basic_ft",
    }

    if model_choice not in model_paths:
        raise ValueError(f"Invalid choice. Use: {list(model_paths.keys())}")

    model_path = model_paths[model_choice]
    embeddings = CustomEmbeddingsEmbeddings(model_path=model_path)

    print(f"Loaded {model_choice} model from {model_path}")
    return embeddings


# ==================================================================
#           OPTION 3: For LangChain RAG Pipeline
# ==================================================================

# Example: Using with LangChain's Chroma vector store
def setup_rag_with_custom_embeddings():
    """
    Setup RAG pipeline with your custom embeddings.
    """
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter

    # Initialize your embeddings
    embeddings = download_embeddings()

    # Load documents
    loader = TextLoader("your_medical_data.txt")
    documents = loader.load()

    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create vector store with your embeddings
    vector_store = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store, embeddings


# ==================================================================
#           BATCH EMBEDDING EXAMPLE
# ==================================================================

def embed_medical_data(texts: list[str]):
    """
    Embed multiple medical texts using your model.
    """
    embeddings = download_embeddings()

    # Embed all at once (faster)
    embedded_list = embeddings.embed_documents(texts)

    return embedded_list


# Example usage
medical_texts = [
    "Patient has hypertension",
    "High blood pressure detected",
    "Patient is diabetic",
]

embedded_vectors = embed_medical_data(medical_texts)
print(f"Embedded {len(embedded_vectors)} texts")
print(f"Each embedding dimension: {len(embedded_vectors[0])}")
print()

# ==================================================================
#     COMPARISON: Old vs New
# ==================================================================

# print("="*70)
# print("COMPARISON")
# print("="*70)
# print()
# print("OLD CODE (Default HuggingFace):")
# print("  from langchain.embeddings import HuggingFaceEmbeddings")
# print("  embeddings = HuggingFaceEmbeddings(")
# print('      model_name="sentence-transformers/all-MiniLM-L6-v2"')
# print("  )")
# print()
# print("NEW CODE (Your Ensemble):")
# print("  embeddings = CustomEnsembleEmbeddings(")
# print('      model_path="./l6v2_3fold_models/l6v2_3fold_ensemble"')
# print("  )")
# print()
# print("PERFORMANCE: Your model should give better results")
# print("            because it's fine-tuned on your medical data!")
# print("="*70)
