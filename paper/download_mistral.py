from huggingface_hub import snapshot_download
for repo, pat in [("mistralai/Mistral-7B-Instruct-v0.2", ["*.safetensors", "*.json", "tokenizer.model"]),
                  ("yogvidwankhede/healthmate-mistral-7b-medical-lora", ["*.json", "*.safetensors", "*.jinja"])]:
    print(snapshot_download(repo, allow_patterns=pat))
