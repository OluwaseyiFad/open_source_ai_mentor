import torch

from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset


max_seq_len = 2048  # Tune for long context
model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
dataset = load_dataset()


# Load model with 4-bit quantization
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    dtype=None,
    max_seq_length=max_seq_len,
    load_in_4bit=True,  # QLORA activated
    full_finetuning=False,
)


# Attach LoRA adapters
model = FastModel.get_peft_model(
    model,
    r=16,  # Tune as needed
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Apply LoRA adapters to all layers
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth optimized checkpointing
)

training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=60,
    save_total_limit=3,  # Keeps only 3 most recent checkpoints
    output_dir="outputs",
    report_to="none",
),

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
    packing=True,
)

trainer_stats = trainer.train()