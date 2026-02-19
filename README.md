<div align="center">

# LlamaTron RS1 Nemesis

A compact 1B-parameter medical reasoning model fine-tuned from  
meta-llama/Llama-3.2-1B-Instruct on a large-scale clinical chain-of-thought dataset

</div>

LlamaTron RS1 Nemesis is designed to support structured clinical reasoning tasks such as differential diagnosis, treatment planning, pharmacological reasoning, and clinical case analysis. The model was created through supervised fine-tuning (SFT) using QLoRA on a high-quality, openly available dataset of 204,773 clinical conversations containing complete chain-of-thought traces.

## Model Overview

- **Base Model**  
  meta-llama/Llama-3.2-1B-Instruct

- **Fine-tuning Dataset**  
  Medical-Reasoning-SFT-MiniMax-M2.1  
  (204,773 clinical reasoning conversations with full chain-of-thought annotations)

- **Training Method**  
  QLoRA (4-bit NormalFloat quantization with double quantization + LoRA adapters)

- **Trainable Parameters**  
  5.6 million (0.45% of total parameters)

- **Hardware**  
  Single NVIDIA H200 GPU

- **Training Duration**  
  3 hours 59 minutes

- **Training Steps**  
  6,271

- **Convergence**  
  Training loss decreased steadily from 1.57 to 1.42  
  Validation loss: 1.45 (stable convergence, no overfitting observed)

## Dataset Citation

```bibtex
@dataset{medical_reasoning_sft_minimax_m2_1,
  title       = {Medical-Reasoning-SFT-MiniMax-M2.1},
  author      = {OpenMed},
  year        = {2025},
  publisher   = {Hugging Face},
  url         = {https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-MiniMax-M2.1}
}
The dataset is licensed under Apache 2.0.
Training Configuration Highlights

Quantization
4-bit NF4 with double quantization, compute dtype bfloat16 (via bitsandbytes)
LoRA Settings
rank = 8, alpha = 16, dropout = 0.05, bias = none
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Key Hyperparameters
Per-device batch size: 8
Gradient accumulation steps: 4 (effective batch size 32)
Learning rate: 2×10⁻⁴ with cosine decay and 5% warmup ratio
Optimizer: paged AdamW 8-bit
Maximum sequence length: 512 tokens
Precision: bfloat16 + TF32 enabled
Gradient checkpointing: enabled
Packing: disabled
Length grouping: enabled (group_by_length)

Evaluation & Saving
Validation every 500 steps
Best model selected by lowest validation loss
Save total limit: 3 checkpoints

Usage Example (Inference)
Pythonfrom transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "your-username/LlamaTron-RS1-Nemesis"  # ← replace with your actual repository
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "user", "content": (
        "55-year-old male presenting with acute chest pain radiating to the left arm, "
        "diaphoresis, and nausea. Provide a structured differential diagnosis "
        "and recommended initial management."
    )}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
Training & Merging Instructions
The repository contains plain text files with the complete training and merging code. Users can copy-paste the content directly into Python scripts.

fine_tune_script.txt
Full QLoRA fine-tuning configuration and training loop
merge_script.txt
Script to load the LoRA adapter and merge it into the base model for efficient inference

Both files are self-contained and require only minor path adjustments before use.
Performance Summary
The training process showed smooth and consistent loss reduction across all 6,271 steps. The final validation loss of 1.45 reflects effective adaptation to the clinical reasoning domain without signs of overfitting.

Demo Screenshots

<img width="1451" height="498" alt="1" src="https://github.com/user-attachments/assets/20190d7b-858b-4ffb-a5d2-2f101b18a01c" />

<img width="1444" height="758" alt="2" src="https://github.com/user-attachments/assets/ca3aa3eb-6632-413c-b8bd-cbc1bbbc0be6" />


Repository Contents
text.
├── fine_tune_script.txt      # Complete fine-tuning code (copy-paste ready)
├── merge_script.txt          # LoRA merging code (copy-paste ready)
├── README.md                 # This document
└── requirements.txt          # List of required Python packages


License
Apache 2.0
The model weights, training code, and any derivative works are released under the Apache 2.0 license, in alignment with the base model and dataset.
This project aims to contribute to the development of transparent, efficient, and high-quality open-source medical reasoning systems.
text### Quick Tips for Adding Your Images
1. Create a folder called `images/` (or `screenshots/`) in your repo root.
2. Upload your two images there (e.g. `loss-curve.png` and `inference-example.png`).
3. Update the paths like this:
   ```markdown
   ![Training Loss Curve](images/loss-curve.png)
   ![Model Inference Example](images/inference-example.png)

Commit and push — GitHub will display them perfectly.
