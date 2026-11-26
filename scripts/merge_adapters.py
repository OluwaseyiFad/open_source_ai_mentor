import torch
from pathlib import Path
from unsloth import FastModel


chckpoint_path = Path("./outputs/final_checkpoint")

max_seq_len = 8192

model, tokenizer = FastModel.from_pretrained(
    model_name=str(chckpoint_path),
    max_seq_length=max_seq_len,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # Use False for lossless 16-bit merge
)

# Merge trained weights into based base model
model.save_pretrained_merged(
    "./outputs/merged_model",
    tokenizer,
    save_method="merged_16bit",
)


# Export in GGUF
model.save_pretrained_gguf(
    "./outputs/gguf_model",
    tokenizer,
    quantization_method="q4_k_m", # Determines model size and quality
)
