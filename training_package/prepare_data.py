import json
import random
import os

def prepare_data(output_dir="data"):
    print("Checking for data...")
    
    # 1. Load SQuAD v2 (Factual, single-hop questions)
    # Set HF Mirror environment variable for users in China or restricted networks
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        from datasets import load_dataset, concatenate_datasets
        print("Loading SQuAD v2 dataset (Base knowledge)...")
        squad = load_dataset("squad_v2", split="train[:20000]") # Use subset to speed up
    except ImportError:
        print("Error: 'datasets' library not found.")
        return

    # 2. Load HotpotQA (Complex, multi-hop reasoning questions)
    print("Loading HotpotQA dataset (Complex reasoning)...")
    hotpot = load_dataset("hotpot_qa", "distractor", split="train[:5000]") # Add some complex examples

    print("Processing mixed data...")
    
    formatted_data = []
    
    # Process SQuAD
    for item in squad:
        if len(item['answers']['text']) > 0:
            formatted_data.append({
                "input_text": f"generate questions: {item['context'].strip()}",
                "target_text": item['question'].strip()
            })

    # Process HotpotQA
    # HotpotQA context is a list of [title, sentences]. We need to join them.
    for item in hotpot:
        # Context is a list of [title, [sentences]]
        # We'll flatten the context to text
        context_text = ""
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
            context_text += f"{title}: " + "".join(sentences) + " "
        
        if context_text and item['question']:
            formatted_data.append({
                "input_text": f"generate questions: {context_text.strip()}",
                "target_text": item['question'].strip()
            })
    
    print(f"Total training samples: {len(formatted_data)} (SQuAD + HotpotQA)")
    random.shuffle(formatted_data) # Shuffle mixed data
    
    # We will use a subset for validation from the official validation set
    val_formatted_data = []
    
    # Simple validation split from our mixed data (since we mixed them)
    # Re-split formatted_data instead of loading separate validation sets
    split_idx = int(len(formatted_data) * 0.9)
    train_final = formatted_data[:split_idx]
    val_final = formatted_data[split_idx:]

    # Optional: Limit training data size if it's too huge, but SQuAD is fine (100k)
    # formatted_data = formatted_data[:50000] 

    os.makedirs(output_dir, exist_ok=True)
    
    def save_jsonl(data, filename):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} samples to {path}")

    save_jsonl(train_final, "train.jsonl")
    save_jsonl(val_final, "validation.jsonl")

if __name__ == "__main__":
    prepare_data()
