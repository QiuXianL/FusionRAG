import requests
import json
import os

def download_hotpot_samples(output_path, num_samples=10):
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    print(f"Downloading from {url}...")
    
    try:
        # Stream the response to avoid downloading the whole file
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            # Read a chunk that should contain enough data
            # 1MB should be enough for 10 samples
            chunk = r.raw.read(2 * 1024 * 1024) 
            
            # The file is a large JSON array [...], so we need to handle incomplete JSON
            content = chunk.decode('utf-8')
            
            # Find the start of the array
            start_idx = content.find('[')
            if start_idx == -1:
                print("Could not find start of JSON array")
                return
            
            # We will try to parse objects one by one
            # This is a bit hacky because it's a comma-separated list inside []
            # We'll split by "}, {" which is typical for JSON lists of objects, 
            # but we need to be careful.
            
            # A safer way is to find the first 10 closing braces that balance
            # But simpler: just use string manipulation since we know the structure roughly
            
            samples = []
            current_pos = start_idx + 1
            
            count = 0
            while count < num_samples:
                # Find the next object start and end
                # This is tricky with nested structures. 
                # Let's just dump the raw content and use a robust parser if possible, 
                # or just use the HuggingFace datasets library but fix the error.
                
                # Actually, let's try to fix the datasets library error first.
                # The error was "Feature type 'List' not found".
                # This often happens when 'datasets' doesn't know how to handle a specific type in the schema.
                # But since that failed, let's stick to manual download.
                
                # Let's look for "answer": "..." which is at the end of each object usually? 
                # No, structure is unpredictable.
                
                # Let's try to find "context": and matching braces.
                pass
                break # breaking because the while loop is pseudo-code here
                
    except Exception as e:
        print(f"Error: {e}")

# Better approach: 
# The file is valid JSON. We can read until we have enough closing brackets?
# Or just read a large chunk, append ']' and try to parse?
# Or better: use a library that streams JSON like `ijson`.
# But `ijson` might not be installed.

# Alternative: Use HuggingFace Hub directly to get a parquet file if available?
# HotpotQA on HF is usually a script.

# Let's try to just download the file fully? It's about 45MB for dev. 
# That's acceptable.

def download_full_and_extract():
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    local_filename = "evaluation/hotpot_dev_distractor_v1.json"
    
    if not os.path.exists(local_filename):
        print("Downloading full dev set (approx 45MB)...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    else:
        print("File already exists.")
        
    # Now parse it
    with open(local_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total samples: {len(data)}")
    return data

def save_to_json(samples, filename):
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} samples to {filename}")

if __name__ == "__main__":
    all_samples = download_full_and_extract()
    
    # Transform to our format
    transformed = []
    # Use all samples for transformation, but we only need first 150 really
    for i, item in enumerate(all_samples[:150]):
        context_texts = []
        # HotpotQA context structure: [ [title, [sent1, sent2]], ... ]
        # item['context'] is a list of [title, sentences] pairs
        # Actually item['context'] is often a list of lists: [ [title, sentences], ... ]
        for title, sentences in item['context']:
             text = "".join(sentences)
             context_texts.append(f"Title: {title}\n{text}")
             # Just take first few contexts to avoid too long
             if len(context_texts) >= 5:
                 break
        
        full_context = "\n\n".join(context_texts)
        
        transformed.append({
            "id": item['_id'],
            "question": item['question'],
            "answer": item['answer'],
            "content": full_context,
            "source": "hotpot_qa"
        })
        
    # Save 20 transformed samples
    save_to_json(transformed[:20], "evaluation/hotpot_20_samples.json")
    
    # Save 40 transformed samples
    save_to_json(transformed[:40], "evaluation/hotpot_40_samples.json")

    # Save 50 transformed samples
    save_to_json(transformed[:50], "evaluation/hotpot_50_samples.json")

    # Save 150 transformed samples
    save_to_json(transformed[:150], "evaluation/hotpot_150_samples.json")

