import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

def generate_question(context, model_path="./output_model"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you run train.py first?")
        return

    input_text = f"generate questions: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=64, 
        num_beams=4, 
        early_stopping=True
    )
    
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Read from command line arg or file
        arg = sys.argv[1]
        if arg.endswith(".txt"):
            with open(arg, 'r', encoding='utf-8') as f:
                context = f.read()
        else:
            context = arg
    else:
        # Default test
        context = "DeepSeek is an artificial intelligence company that focuses on developing large language models."
        
    print(f"Context: {context}")
    print("-" * 30)
    q = generate_question(context)
    print(f"Generated Question: {q}")
