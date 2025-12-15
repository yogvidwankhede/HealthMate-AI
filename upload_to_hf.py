"""
Script to Upload Your Custom Embedding Model to Hugging Face Hub

Run this script ONCE locally to upload your fine-tuned model to Hugging Face.
After uploading, you can use it in production deployment.

Prerequisites:
1. Install: pip install huggingface_hub
2. Login: huggingface-cli login
3. Have your model path ready
"""

from sentence_transformers import SentenceTransformer
import os
from pathlib import Path


def upload_model_to_hf(
    local_model_path: str,
    hf_model_name: str,
    private: bool = True,
    organization: str = None
):
    """
    Upload your fine-tuned embedding model to Hugging Face Hub.
    
    Args:
        local_model_path: Path to your local model (e.g., './Embedding/l6v2_3fold_models/l6v2_3fold_ensemble')
        hf_model_name: Name for your model on HF (e.g., 'username/healthmate-medical-embeddings')
        private: Whether to make the model private (recommended for production)
        organization: Optional organization name (if uploading to an org)
    
    Example:
        upload_model_to_hf(
            local_model_path='./Embedding/l6v2_3fold_models/l6v2_3fold_ensemble',
            hf_model_name='yogvid/healthmate-medical-embeddings',
            private=True
        )
    """

    print("="*70)
    print("UPLOADING CUSTOM EMBEDDING MODEL TO HUGGING FACE HUB")
    print("="*70)
    print()

    # Check if model exists
    if not os.path.exists(local_model_path):
        print(f"❌ Error: Model not found at {local_model_path}")
        return False

    print(f"📂 Loading model from: {local_model_path}")

    try:
        # Load the model
        model = SentenceTransformer(local_model_path)
        print("✅ Model loaded successfully")
        print(
            f"   Model dimension: {model.get_sentence_embedding_dimension()}")

        # Test the model
        print("\n🧪 Testing model...")
        test_embedding = model.encode(["This is a test sentence"])
        print(f"✅ Test successful. Embedding shape: {test_embedding.shape}")

        # Upload to Hugging Face
        print(f"\n📤 Uploading to Hugging Face Hub as: {hf_model_name}")
        print(f"   Private: {private}")
        if organization:
            print(f"   Organization: {organization}")

        model.save_to_hub(
            repo_id=hf_model_name,
            organization=organization,
            private=private,
            commit_message="Upload HealthMate-AI fine-tuned medical embeddings (3-fold ensemble)",
            exist_ok=True  # Overwrite if exists
        )

        print("\n✅ SUCCESS! Model uploaded to Hugging Face Hub")
        print(f"\n📍 Your model URL: https://huggingface.co/{hf_model_name}")

        # Instructions for deployment
        print("\n" + "="*70)
        print("NEXT STEPS FOR DEPLOYMENT")
        print("="*70)
        print()
        print("1. In your Render dashboard, add this environment variable:")
        print(f"   HF_MODEL_NAME = {hf_model_name}")
        print()
        print("2. Use the helper_option1_hf.py file (rename to helper.py)")
        print()
        print("3. Deploy your app - it will automatically use your custom model!")
        print()
        print("="*70)

        return True

    except Exception as e:
        print(f"\n❌ Error uploading model: {e}")
        return False


def main():
    """Main function with interactive prompts"""

    print("🏥 HealthMate-AI - Model Upload Wizard")
    print("="*70)
    print()

    # Get model path
    print("📂 Enter the path to your local model:")
    print("   Example: ./Embedding/l6v2_3fold_models/l6v2_3fold_ensemble")
    print("   Example: D:/1A Medical Projecy/l6v2_3fold_models/l6v2_3fold_ensemble")
    local_path = input("   Path: ").strip()

    if not local_path:
        print("❌ No path provided. Exiting.")
        return

    # Get Hugging Face username
    print("\n👤 Enter your Hugging Face username:")
    print("   (Find it at: https://huggingface.co/settings/account)")
    username = input("   Username: ").strip()

    if not username:
        print("❌ No username provided. Exiting.")
        return

    # Get model name
    print("\n📝 Enter a name for your model:")
    print("   Example: healthmate-medical-embeddings")
    print("   (Will be uploaded as: {username}/{model-name})")
    model_name = input("   Model name: ").strip()

    if not model_name:
        print("❌ No model name provided. Exiting.")
        return

    full_name = f"{username}/{model_name}"

    # Ask about privacy
    print("\n🔒 Should this model be private?")
    print("   Private: Only you can access (recommended for production)")
    print("   Public: Anyone can access")
    private_input = input("   Private? (y/n): ").strip().lower()
    private = private_input in ['y', 'yes', '']

    # Confirm
    print("\n" + "="*70)
    print("CONFIRMATION")
    print("="*70)
    print(f"Local path:  {local_path}")
    print(f"HF repo:     {full_name}")
    print(f"Private:     {private}")
    print()
    confirm = input("Proceed with upload? (y/n): ").strip().lower()

    if confirm not in ['y', 'yes']:
        print("❌ Upload cancelled.")
        return

    # Upload
    success = upload_model_to_hf(
        local_model_path=local_path,
        hf_model_name=full_name,
        private=private
    )

    if success:
        print("\n🎉 All done! Your model is now on Hugging Face Hub!")
    else:
        print("\n❌ Upload failed. Please check the error messages above.")


if __name__ == "__main__":
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
        print("✅ huggingface_hub is installed")
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("📦 Install it with: pip install huggingface_hub")
        print("🔑 Then login with: huggingface-cli login")
        exit(1)

    # Run the wizard
    main()


# ============================================================================
# ALTERNATIVE: Direct usage without wizard
# ============================================================================

# If you want to run this directly in your code:
"""
from upload_to_hf import upload_model_to_hf

upload_model_to_hf(
    local_model_path='./Embedding/l6v2_3fold_models/l6v2_3fold_ensemble',
    hf_model_name='your-username/healthmate-medical-embeddings',
    private=True
)
"""
