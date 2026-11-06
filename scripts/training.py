import torch

from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset


# Load model with 4-bit quantization
def load_model(model_name, max_seq_len):
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        max_seq_length=max_seq_len,
        load_in_4bit=True,  # QLORA activated
        full_finetuning=False,
    )
    return model, tokenizer


# Attach LoRA adapters
def attach_adapter(model, r=16):
    model = FastModel.get_peft_model(
        model,
        r=r,  # Tune as needed
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # Apply LoRA adapters to all layers
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized checkpointing
    )
    return model


def load_sft_trainer(model, tokenizer, dataset):
    training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=60, # smoke test
        save_total_limit=3,  # Keeps only 3 most recent checkpoints
        output_dir="outputs",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        packing=False,  # True not supported on MLX as of now
    )
    return trainer


if __name__ == "__main__":
    max_seq_len = 5120  # Tune for long context
    model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    model, tokenizer = load_model(model_name, max_seq_len)
    model = attach_adapter(model)
    dataset_dict = load_dataset("json", data_files={
            "train": "scripts/test_train.jsonl",
        })
    dataset = dataset_dict["train"]
    # print(dataset["train"][0])
    # dataset = load_dataset("json", data_files={
    #         "train": "./dataset/train.json",
    #         "validation": "./dataset/val.json",
    #     })
    trainer = load_sft_trainer(model, tokenizer, dataset)
    trainer_stats = trainer.train()