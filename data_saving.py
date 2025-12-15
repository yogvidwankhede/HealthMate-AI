# ============================================================================
# PDF Loader with Caching - Avoid Reloading PDFs Every Time
# ============================================================================

import pickle
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from datetime import datetime


def load_pdf_files(data_folder, cache_file='pdf_cache.pkl'):
    """
    Extract text from PDF files and cache the results.
    
    First run: Loads PDFs from folder and saves to cache file (slower)
    Next runs: Loads from cache file (very fast)
    
    Args:
        data_folder: Folder path containing PDF files (e.g., "data")
        cache_file: Where to save cached data (default: 'pdf_cache.pkl')
    
    Returns:
        documents: List of extracted PDF documents
    """

    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"📦 Loading from cache: {cache_file}")
        print("⏱️  This should be instant!\n")

        try:
            with open(cache_file, 'rb') as f:
                documents = pickle.load(f)

            print(
                f"✅ Successfully loaded {len(documents)} documents from cache")
            print(
                f"📁 Cache file size: {os.path.getsize(cache_file) / (1024*1024):.2f} MB\n")

            return documents

        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            print("🔄 Reloading from PDFs...\n")

    # If no cache or cache failed, load from PDFs
    print(f"📂 Loading PDFs from '{data_folder}'...")
    print("⏳ This will take a few minutes on first run...\n")

    loader = DirectoryLoader(
        data_folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    print(f"✅ Successfully loaded {len(documents)} documents from PDFs")

    # Save to cache for future use
    print(f"\n💾 Saving to cache: {cache_file}")

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(documents, f)

        cache_size = os.path.getsize(cache_file) / (1024*1024)
        print(f"✅ Cache saved successfully ({cache_size:.2f} MB)")
        # print(f"⚡ Next time, loading will be instant!\n")

    except Exception as e:
        print(f"⚠️  Warning: Could not save cache: {e}\n")

    return documents


# ============================================================================
# OPTIONAL: Function to clear cache if you update your PDFs
# ============================================================================

def clear_pdf_cache(cache_file='pdf_cache.pkl'):
    """
    Delete the cache file (use this if you add new PDFs to the folder)
    
    Usage:
        clear_pdf_cache()
    """
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"🗑️  Cache file deleted: {cache_file}")
        print("📝 Next load will rebuild cache from updated PDFs\n")
    else:
        print(f"ℹ️  Cache file not found: {cache_file}\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
FIRST TIME (Loads from PDFs and creates cache):
─────────────────────────────────────────────────
>>> extracted_data = load_pdf_files("data")

📂 Loading PDFs from 'data'...
⏳ This will take a few minutes on first run...

✅ Successfully loaded 4505 documents from PDFs

💾 Saving to cache: pdf_cache.pkl
✅ Cache saved successfully (156.23 MB)
⚡ Next time, loading will be instant!


SECOND TIME (Loads from cache - instant):
──────────────────────────────────────────
>>> extracted_data = load_pdf_files("data")

📦 Loading from cache: pdf_cache.pkl
⏱️  This should be instant!

✅ Successfully loaded 4505 documents from cache
📁 Cache file size: 156.23 MB


IF YOU ADD NEW PDFs TO THE FOLDER:
──────────────────────────────────
>>> clear_pdf_cache()

🗑️  Cache file deleted: pdf_cache.pkl

>>> extracted_data = load_pdf_files("data")  # This will rebuild cache


CUSTOM CACHE FILE NAME:
──────────────────────
>>> extracted_data = load_pdf_files("data", cache_file='my_pdfs_backup.pkl')
"""
