# Question Generation Training Package

This package allows you to train a custom Question Generation model on a Linux server to replace external APIs (like DeepSeek) for generating questions from text.

## 1. Setup

1.  **Upload** this entire `training_package` folder to your Linux server.
2.  **Upload Data**: Copy your `squad_data.json` file into this folder.
    *   The file should be a JSON list of objects: `[{"document": "...", "question": "...", "answer": "..."}]`
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 2. Prepare Data

Run the preparation script to convert the JSON into training format (JSONL):

```bash
python prepare_data.py
```

This will create a `data/` directory with `train.jsonl` and `validation.jsonl`.

## 3. Train

Since you have an 8x 4090 server, we can leverage multi-GPU training for faster results and larger models.

### Option A: Single GPU (Simple)
If you just want to run it quickly on one card:
```bash
python train.py
```

### Option B: Multi-GPU (Recommended for 8x 4090)
To use all GPUs, use `torchrun`. 
**Note**: If you encounter "address already in use" error, specify a different master port.

```bash
torchrun --nproc_per_node=8 --master_port=29501 train.py
```

**Configuration Note**:
The `train.py` is currently configured for **`google/flan-t5-xl`** (3B parameters), which fits comfortably on a 4090 (24GB).
- If you want to train the massive **`google/flan-t5-xxl`** (11B parameters), you will need to use DeepSpeed or FSDP (Fully Sharded Data Parallel).

## 4. Test (Inference)

Test your new model:

```bash
python inference.py "DeepSeek is an AI company."
```

## 5. Deployment

Once trained, you can download the `output_model` folder back to your local machine (or use it on the server).

To use it in your RAG system, point the `QuestionGenerator` to load from this local path instead of calling the API.
