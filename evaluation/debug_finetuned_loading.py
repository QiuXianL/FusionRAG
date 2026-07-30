
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_loading():
    model_path = "output_model/output_model_server"
    print(f"Testing loading from: {model_path}")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Tokenizer loaded.")
        
        print("Loading model (CPU, float32)...")
        # Try simplest loading first to rule out quantization/device issues
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully on CPU!")
        
        print("Moving to GPU...")
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("Moved to GPU.")
            
        # Test Generation
        print("\nTesting Generation...")
        text = "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks."
        input_text = f"generate questions: {text}"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=64)
        print("Input:", text)
        print("Generated Question:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        
    except Exception as e:
        print(f"Loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
