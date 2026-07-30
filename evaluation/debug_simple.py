
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from src.document_manager import DocumentManager
import sys

# Force flush
sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

def test_deepseek_simple():
    print("\n=== Testing DeepSeek Simple ===")
    api_key = os.getenv("DEEPSEEK_API")
    print(f"API Key: {api_key[:4]}... if exists")
    
    if not api_key:
        print("No API Key!")
        return

    dm = DocumentManager()
    content = "DeepSeek is an AGI company."
    try:
        print("Generating...")
        qs = dm._generate_enhanced_questions(content, 1)
        print(f"Questions: {qs}")
    except Exception as e:
        print(f"Error: {e}")

def test_base_model_loading():
    print("\n=== Testing Base Model Loading ===")
    model_path = "google/flan-t5-xl"
    # model_path = "./models/e5-base-v2" # Test with small model if XL fails
    
    print(f"Loading {model_path}...")
    try:
        # Try simplified loading
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Try CPU load if GPU fails
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=False # Force false to debug
        )
        print("Loaded!")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_deepseek_simple()
    # test_base_model_loading() # Skip for now to save time/memory
