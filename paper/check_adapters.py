"""Check every tensor of the three published LoRA adapters for NaN/inf. Writes paper/results/adapter_nan.json."""
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
out = {}
for sub in ["", "adapter_seed_123/", "adapter_seed_999/"]:
    p = hf_hub_download("yogvidwankhede/healthmate-mistral-7b-medical-lora", sub + "adapter_model.safetensors")
    n = nan = inf = 0
    with safe_open(p, "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k).float(); n += 1; nan += int(torch.isnan(t).any()); inf += int(torch.isinf(t).any())
    out[sub.strip("/") or "seed42_root"] = {"tensors": n, "tensors_with_nan": nan, "tensors_with_inf": inf}
json.dump(out, open("paper/results/adapter_nan.json", "w"), indent=1); print(out)
