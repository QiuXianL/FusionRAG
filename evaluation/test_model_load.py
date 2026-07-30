
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "output_model/checkpoint-345"
print(f"Testing model loading from: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded.")
    
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        load_in_8bit=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
