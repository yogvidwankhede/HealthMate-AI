import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.cuda.empty_cache()

print("="*70)
print("STEP 1: TEST BASE MODEL (NO ADAPTER)")
print("="*70)

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    use_fast=False
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("✓ Base model loaded")

# Test base model
test_text = "[INST] What is diabetes? [/INST]"
inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Input IDs: {inputs['input_ids'][0].tolist()}")

try:
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\n✅ BASE MODEL WORKS!")
    print(f"Response: {response}\n")

    base_works = True

except Exception as e:
    print(f"\n❌ BASE MODEL FAILED: {e}")
    base_works = False

if base_works:
    print("="*70)
    print("STEP 2: TEST WITH ADAPTER")
    print("="*70)

    from peft import PeftModel

    adapter_model = PeftModel.from_pretrained(
        base_model,
        "LLM/mistral_medical_lora_models/adapter_seed_42"
    )
    adapter_model.eval()

    print("✓ Adapter loaded")

    # Test adapter model
    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

    try:
        with torch.no_grad():
            outputs = adapter_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\n✅ ADAPTER MODEL WORKS!")
        print(f"Response: {response}")

    except Exception as e:
        print(f"\n❌ ADAPTER MODEL FAILED: {e}")
        print("\n🔍 This means your adapter has an issue!")
        print("Possible causes:")
        print("  1. Adapter was trained with different transformers version")
        print("  2. Adapter config incompatibility")
        print("  3. Corrupted adapter weights")
        print("\nSolution: Re-download adapter from Kaggle or use base model")
