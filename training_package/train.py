import os
# Set HF Mirror environment variable at the very beginning, BEFORE any other imports
# This ensures transformers/datasets libraries pick it up during initialization
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    TrainerCallback # Import TrainerCallback
)
import numpy as np
import gc # Import garbage collector

# Define a callback to clear CUDA cache
class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Clear cache every 50 steps to prevent OOM
        if state.global_step > 0 and state.global_step % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

# 1. Model Configuration
# Use "google/flan-t5-xl" (3B params) for better quality if you have >24GB VRAM
# Use "google/flan-t5-large" (780M params) for >12GB VRAM
# Use "google/flan-t5-base" (250M params) for standard testing
MODEL_NAME = "google/flan-t5-xl"
OUTPUT_DIR = "./output_model"
DATA_DIR = "./data"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128 # Increased for better questions
BATCH_SIZE = 4 # Conservative size to safely fit in busy GPUs
EPOCHS = 5
LEARNING_RATE = 1e-4

def main():
    # 1. Load Data
    data_files = {
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "validation.jsonl")
    }
    
    dataset = load_dataset("json", data_files=data_files)
    print(f"Loaded dataset: {dataset}")

    # Initialize Distributed Environment if using torchrun
    # We must do this BEFORE any distributed operations like barrier()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if local_rank != -1:
        # Explicitly set master address and port if not present, to ensure NCCL works
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500" # Default, usually overwritten by torchrun

        # Disable DeepSpeed JIT compilation to avoid CUDA version mismatch error
        # This forces DeepSpeed to use pure Python/PyTorch implementations where possible
        os.environ["DS_BUILD_OPS"] = "0"

        if not torch.distributed.is_initialized():
             torch.distributed.init_process_group(backend="nccl")

    # 2. Tokenizer
    # Ensure only the main process (rank 0) downloads the model
    # Others wait until it's cached
    
    tokenizer = None
    model = None
    
    if local_rank <= 0:
        print(f"Process {local_rank}: Downloading/Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Barrier to wait for rank 0
    if local_rank != -1:
        torch.distributed.barrier()

    # Now load for everyone (should use cache)
    if local_rank > 0:
         print(f"Process {local_rank}: Loading from cache...")
         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
         model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
    
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Use load_from_cache_file=True to leverage caching in distributed setting
    tokenized_datasets = dataset.map(preprocess_function, batched=True, load_from_cache_file=True)

    # 3. Model
    # Model already loaded above to handle distributed download
    # model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # 4. Training Arguments
    # Distributed training handling
    # If running with torchrun/accelerate, these will be set automatically
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps", # Change to steps to save progress more frequently
        eval_steps=200,        # Evaluate every 200 steps
        save_strategy="steps", # Save checkpoint every 200 steps
        save_steps=200,        # Save checkpoint every 200 steps
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,    # Keep last 3 checkpoints
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        fp16=False,
        bf16=True,
        gradient_accumulation_steps=8,
        ddp_find_unused_parameters=False,
        report_to="none",
        gradient_checkpointing=True,
        deepspeed="ds_config.json", # Enable DeepSpeed ZeRO-2 for memory optimization
        max_grad_norm=1.0, # Add gradient clipping to prevent explosion
        dataloader_num_workers=0, # Reduced to 0 (main process only) to prevent RAM OOM
        load_best_model_at_end=True, # Load best model at end
        metric_for_best_model="loss",
    )

    # 5. Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[ClearCacheCallback()], # Add the callback here
    )

    # 7. Train
    if local_rank <= 0:
        print("Starting training...")
    trainer.train()

    # 8. Save
    if local_rank <= 0:
        print(f"Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Training complete!")

if __name__ == "__main__":
    main()
