
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from src.document_manager import DocumentManager

load_dotenv()

def test_finetuned_loading():
    print("\n=== Testing Finetuned Model Loading ===")
    model_path = "output_model/checkpoint-345"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Loading model from {model_path} (forcing 8bit=False)...")
        # Force float32 and no device_map to be safe first
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        print("✅ Model loaded successfully!")
        
        # Try generation
        input_text = "generate questions: This is a test context."
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=20)
        print(f"Generation test: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

def test_deepseek_generation():
    print("\n=== Testing DeepSeek Generation ===")
    api_key = os.getenv("DEEPSEEK_API")
    print(f"API Key present: {bool(api_key)}")
    if api_key:
        print(f"API Key (masked): {api_key[:4]}...{api_key[-4:]}")
    
    doc_manager = DocumentManager(persist_directory="temp_debug_db")
    content = "DeepSeek is an artificial intelligence company. It focuses on AGI."
    
    try:
        print("Calling _generate_enhanced_questions...")
        questions = doc_manager._generate_enhanced_questions(content, num_questions=1)
        print(f"Generated questions: {questions}")
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_finetuned_loading()
    test_deepseek_generation()
