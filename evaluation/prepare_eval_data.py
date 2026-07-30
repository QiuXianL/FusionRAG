import json
import os

def convert_squad_to_jsonl(input_file, output_file):
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Converting {len(data)} items...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # Construct input/target pair
            # Input: generate questions: <document>
            # Target: <question>
            new_item = {
                "input_text": f"generate questions: {item['document']}",
                "target_text": item['question']
            }
            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    input_path = "evaluation/squad_data.json"
    output_path = "evaluation/validation.jsonl"
    
    if os.path.exists(input_path):
        convert_squad_to_jsonl(input_path, output_path)
    else:
        print(f"Error: {input_path} not found.")
