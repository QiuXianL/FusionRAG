import os
# Set HF Mirror environment variable to speed up downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback
)
import gc

# --- Configuration for Server (RTX 4090 24GB) ---
# NOTE: The "XL" (3B) model is just slightly too big for 24GB VRAM even with optimizations.
# We switch to "Large" (780M params) which is still VERY powerful but runs smoothly.
MODEL_NAME = "google/flan-t5-large" 
OUTPUT_DIR = "./output_model_server"
DATA_DIR = "./data"
# Explicitly set download location
MODEL_CACHE_DIR = "./models_cache" 

# Training parameters optimized for RTX 4090 (24GB VRAM) - FLAN-T5-LARGE
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
# "Large" with Checkpointing fits HUGE batches.
# We use Batch 8 + Checkpointing + Accumulation = High Throughput + Safety
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 10
LEARNING_RATE = 2e-4

# Add garbage collection callback
class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

def main():
    # 0. Environment Check
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 1. Load Data
    data_files = {
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "validation": os.path.join(DATA_DIR, "validation.jsonl")
    }
    
    # Check if data exists
    if not os.path.exists(data_files["train"]):
        print(f"Error: Data file not found at {data_files['train']}")
        print("Please run 'python prepare_data.py' first.")
        return

    dataset = load_dataset("json", data_files=data_files)
    print(f"Loaded dataset: {dataset}")

    # 2. Tokenizer & Model
    # We specify cache_dir so the model is downloaded to YOUR folder, not hidden system folders
    print(f"Loading model: {MODEL_NAME}...")
    print(f"Model will be downloaded to: {os.path.abspath(MODEL_CACHE_DIR)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    
    # 3. Preprocess Data
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 4. Training Arguments
    # Optimized for Single GPU (RTX 4090)
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 'evaluation_strategy' is deprecated in newer transformers versions, using 'eval_strategy'
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        
        # --- Memory Optimization Key Settings ---
        fp16=False,
        bf16=True, # RTX 4090 supports BF16, which is more stable than FP16
        gradient_checkpointing=True, # Enable this to save VRAM
        optim="adafactor", # Uses less memory than AdamW
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # --- Anti-Overfitting Strategy ---
        
        report_to="none",
        load_best_model_at_end=True,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ClearCacheCallback()],
    )

    # 7. Train
    print("Starting training on your RTX 4090 server...")
    trainer.train()

    # 8. Save
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete! You can now download the 'output_model_local' folder.")

if __name__ == "__main__":
    main()
