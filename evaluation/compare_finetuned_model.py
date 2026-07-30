import os
import sys
import random

# 1. 优先设置国内镜像源 (必须在 import transformers 之前设置)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import argparse
from tqdm import tqdm
import gc
# os.environ["HF_DATASETS_OFFLINE"] = "1" 
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

class ModelComparator:
    def __init__(self, base_model_name="google/flan-t5-xl", finetuned_model_path="./output_model", device=None):
        self.base_model_name = base_model_name
        self.finetuned_model_path = finetuned_model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_model_and_generate(self, model_path, samples, batch_size=4, is_baseline=False):
        """
        加载模型，对样本进行生成，然后释放显存。
        """
        print(f"\n[{'Baseline' if is_baseline else 'Finetuned'}] Loading model from {model_path}...")
        
        # 移除 try-except 以便在服务器上看到完整的报错堆栈
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 使用 bfloat16 加速并节省显存
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        model.eval()

        print(f"Generating responses for {len(samples)} samples...")
        predictions = []
        
        # Batch generation
        for i in tqdm(range(0, len(samples), batch_size)):
            batch_contexts = samples[i:i+batch_size]
            # 添加前缀 (训练时如果加了前缀，这里也要加)
            # 假设训练数据中已经是 "generate questions: context" 格式，或者我们在 prepare_data.py 里加了
            # 查看 prepare_data.py，输入已经是 "generate questions: ..."
            # 所以这里如果输入只有 context，需要手动加。
            # 我们假设传入的 samples 已经是纯 context，需要拼接。
            
            input_texts = [f"generate questions: {ctx}" for ctx in batch_contexts]
            
            inputs = tokenizer(
                input_texts, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=64,
                    num_beams=4,
                    early_stopping=True
                )
            
            batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_preds)
            
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return predictions

    def run_comparison(self, data_path, num_samples=10, output_file="model_comparison_report.md"):
        # 1. Load Data
        print(f"Loading data from {data_path}...")
        samples = []
        targets = []
        
        if not os.path.exists(data_path):
            print(f"Error: Data file {data_path} not found.")
            # Fallback to dummy data
            print("Using dummy data for testing...")
            dummy_contexts = [
                "DeepSeek is an artificial intelligence company that focuses on developing large language models.",
                "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states.",
                "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation."
            ]
            samples = dummy_contexts
            targets = ["N/A"] * len(samples)
        else:
            all_lines = []
            with open(data_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            # 随机抽取样本
            if len(all_lines) > num_samples:
                print(f"Randomly selecting {num_samples} samples from {len(all_lines)} total lines...")
                selected_lines = random.sample(all_lines, num_samples)
            else:
                selected_lines = all_lines

            for line in selected_lines:
                item = json.loads(line)
                # 提取纯 context，去掉 "generate questions: " 前缀
                input_text = item.get("input_text", "")
                if input_text.startswith("generate questions: "):
                    context = input_text[len("generate questions: "):]
                else:
                    context = input_text
                
                samples.append(context)
                targets.append(item.get("target_text", ""))

        # 2. Generate with Baseline
        base_preds = self.load_model_and_generate(self.base_model_name, samples, is_baseline=True)
        
        # 3. Generate with Finetuned
        finetuned_preds = self.load_model_and_generate(self.finetuned_model_path, samples, is_baseline=False)
        
        # 4. Generate Report
        self.save_report(samples, targets, base_preds, finetuned_preds, output_file)

    def save_report(self, contexts, targets, base_preds, finetuned_preds, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Model Comparison Report: Question Generation\n\n")
            f.write(f"- **Base Model**: {self.base_model_name}\n")
            f.write(f"- **Finetuned Model**: {self.finetuned_model_path}\n")
            f.write("\n---\n")
            
            for i, (ctx, target, base, ft) in enumerate(zip(contexts, targets, base_preds, finetuned_preds)):
                f.write(f"### Sample {i+1}\n\n")
                f.write(f"**Context**: \n> {ctx[:500]}..." if len(ctx) > 500 else f"**Context**: \n> {ctx}\n")
                f.write("\n")
                f.write(f"**Ground Truth**: `{target}`\n\n")
                f.write(f"**Base Model**: {base}\n\n")
                f.write(f"**Finetuned Model**: **{ft}**\n\n")
                f.write("---\n")
        
        print(f"\nReport saved to {filename}")
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Base vs Finetuned Model")
    parser.add_argument("--base_model", type=str, default="google/flan-t5-xl", help="HuggingFace model name for baseline")
    parser.add_argument("--finetuned_path", type=str, default="./output_model", help="Path to finetuned model directory")
    parser.add_argument("--data_path", type=str, default="./data/validation.jsonl", help="Path to validation data jsonl")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--output_file", type=str, default="comparison_report.md", help="Output markdown file")
    
    args = parser.parse_args()
    
    # 自动寻找数据路径
    data_path = args.data_path
    if not os.path.exists(data_path):
        # 尝试常见路径
        potential_paths = [
            "../training_package/data/validation.jsonl",
            "training_package/data/validation.jsonl",
            "data/validation.jsonl"
        ]
        for p in potential_paths:
            if os.path.exists(p):
                data_path = p
                break
    
    comparator = ModelComparator(
        base_model_name=args.base_model,
        finetuned_model_path=args.finetuned_path
    )
    
    comparator.run_comparison(
        data_path=data_path,
        num_samples=args.num_samples,
        output_file=args.output_file
    )
