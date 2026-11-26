from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
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
        lora_dropout=0,  # Unsloth is fasters with 0 dropout
        random_state=3407,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # Apply LoRA adapters to all layers
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimized checkpointing
    )
    return model


def load_sft_trainer(model, tokenizer, train_dataset, val_dataset, max_seq_len):
    training_args = SFTConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,  # good default for QLORA with limited examples
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        # max_steps=60,  # smoke test
        eval_strategy="steps", eval_steps=25,
        save_strategy="steps", save_steps=25, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=1,
        seed=3407,
        max_length=max_seq_len,
        output_dir="outputs/run1",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        packing=False,  # True not supported on MLX as of now
    )
    return trainer


if __name__ == "__main__":
    max_seq_len = 8192  # Tune for long context
    model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    model, tokenizer = load_model(model_name, max_seq_len)
    model = attach_adapter(model)
    dataset_dict = load_dataset("json", data_files={
            "train": "datasets/train.jsonl",
            "validation": "datasets/val.jsonl",
        })
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]
    # print("train dataset: ", train_dataset[0])
    # print("val dataset: ", val_dataset[0])

    # trainer = load_sft_trainer(model, tokenizer, train_dataset.select(range(20)), val_dataset.select(range(5)), max_seq_len) # for smoke test
    trainer = load_sft_trainer(model, tokenizer, train_dataset, val_dataset, max_seq_len)  # for real run
    trainer_stats = trainer.train()
    trainer.save_model("outputs/final_checkpoint")  # Saves the best model
    tokenizer.save_pretrained("outputs/final_checkpoint")