import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain.llms.base import LLM
from typing import Optional, List, Any
import warnings
warnings.filterwarnings('ignore')


class FineTunedMistralLLM(LLM):
    """Custom LangChain LLM for fine-tuned Mistral-7B"""

    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    def __init__(self, adapter_path: str, use_4bit: bool = False, **kwargs):
        super().__init__(**kwargs)

        print(f"Loading fine-tuned Mistral from {adapter_path}...")

        # Load tokenizer
        print("   Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                use_fast=False,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"   Error loading tokenizer: {e}")
            print("   Trying alternative method...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                adapter_path,
                use_fast=False,
                trust_remote_code=True
            )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        print("   Tokenizer loaded")

        # Load base model
        print("   Loading base model (1-2 minutes)...")
        if use_4bit:
            print("   Using 4-bit quantization...")
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                base = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except ImportError:
                print("   bitsandbytes not available, using FP16")
                base = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                torch_dtype=torch.float16,
                device_map="auto",
            )
        print("   Base model loaded")

        # Load LoRA adapter
        print("   Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        print("   LoRA adapter loaded")

        device = next(self.model.parameters()).device
        print(f"\nModel loaded successfully!")
        print(f"   Device: {device}")
        if torch.cuda.is_available():
            print(
                f"   GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    @property
    def _llm_type(self) -> str:
        return "fine_tuned_mistral"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate response for given prompt"""

        # Format for Mistral
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_name": "fine_tuned_mistral_7b",
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


def create_medical_llm(adapter_path: str, use_4bit: bool = True):
    """
    Create a fine-tuned Mistral LLM for medical question answering.
    
    Args:
        adapter_path: Path to trained LoRA adapter folder
        use_4bit: Enable 4-bit quantization for GTX 1650 Ti
    
    Returns:
        FineTunedMistralLLM instance
    """
    return FineTunedMistralLLM(adapter_path=adapter_path, use_4bit=use_4bit)
