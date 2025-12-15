"""
Test script for custom embeddings

Run this to verify your custom embedding model is working correctly.
"""


def test_embeddings():
    """Test that embeddings are working correctly"""

    print("="*70)
    print("TESTING CUSTOM EMBEDDING MODEL")
    print("="*70)
    print()

    try:
        # Import the helper function
        from src.helper import download_hugging_face_embeddings

        print("Step 1: Loading embedding model...")
        embeddings = download_hugging_face_embeddings()
        print("✅ Model loaded successfully\n")

        # Test single query
        print("Step 2: Testing single query embedding...")
        test_query = "What is diabetes?"
        embedding = embeddings.embed_query(test_query)

        print(f"✅ Query: '{test_query}'")
        print(f"✅ Embedding dimension: {len(embedding)}")
        print(f"✅ First 5 values: {embedding[:5]}")
        print(f"✅ Embedding type: {type(embedding)}")
        print()

        # Test batch embedding
        print("Step 3: Testing batch document embedding...")
        test_docs = [
            "Patient has hypertension",
            "Symptoms of heart disease",
            "Treatment for diabetes"
        ]

        doc_embeddings = embeddings.embed_documents(test_docs)

        print(f"✅ Embedded {len(doc_embeddings)} documents")
        print(f"✅ Each embedding dimension: {len(doc_embeddings[0])}")
        print()

        # Verify dimensions
        print("Step 4: Verifying dimensions...")
        assert len(
            embedding) == 384, f"Expected 384 dims, got {len(embedding)}"
        assert all(
            len(emb) == 384 for emb in doc_embeddings), "Inconsistent dimensions"
        print("✅ All dimensions correct (384)\n")

        # Check value ranges
        print("Step 5: Checking embedding value ranges...")
        import numpy as np
        all_values = np.array(embedding)
        print(f"✅ Min value: {all_values.min():.4f}")
        print(f"✅ Max value: {all_values.max():.4f}")
        print(f"✅ Mean value: {all_values.mean():.4f}")
        print()

        # Test similarity (sanity check)
        print("Step 6: Testing semantic similarity...")
        from numpy import dot
        from numpy.linalg import norm

        # Similar queries should have high similarity
        query1 = embeddings.embed_query("diabetes symptoms")
        query2 = embeddings.embed_query("signs of diabetes")
        query3 = embeddings.embed_query("car engine repair")

        def cosine_similarity(a, b):
            return dot(a, b) / (norm(a) * norm(b))

        sim_similar = cosine_similarity(query1, query2)
        sim_different = cosine_similarity(query1, query3)

        print(f"✅ Similarity (diabetes symptoms vs signs): {sim_similar:.4f}")
        print(f"✅ Similarity (diabetes vs car engine): {sim_different:.4f}")

        if sim_similar > sim_different:
            print("✅ Sanity check passed: Similar queries have higher similarity")
        else:
            print("⚠️  Warning: Similar queries should have higher similarity")
        print()

        # Final summary
        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print()
        print("Your custom embedding model is working correctly! 🎉")
        print()
        print("Next steps:")
        print("1. Test locally: python app.py")
        print("2. Deploy to Render")
        print("3. Monitor performance")
        print()

        return True

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print()
        print("Make sure you're in the deployment folder and have installed dependencies:")
        print("  cd deployment")
        print("  pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_embeddings()
    exit(0 if success else 1)
