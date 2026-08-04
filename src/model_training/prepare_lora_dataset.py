import os
import json
import random

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

INPUT_JSONL = os.path.join(DATA_PROCESSED_DIR, "agentic_distillation_dataset.jsonl")
TRAIN_OUTPUT = os.path.join(DATA_PROCESSED_DIR, "train.jsonl")
VALID_OUTPUT = os.path.join(DATA_PROCESSED_DIR, "valid.jsonl")

def main():
    print("Starting dataset split for MLX training...")
    
    if not os.path.exists(INPUT_JSONL):
        print(f"Error: Input JSONL not found at {INPUT_JSONL}. Run generate_dataset.py first.")
        return

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    dataset = [json.loads(line) for line in lines]
    
    random.seed(42)
    random.shuffle(dataset)

    # 80/20 for training and validation
    split_idx = max(1, int(len(dataset) * 0.8))
    train_data = dataset[:split_idx]
    valid_data = dataset[split_idx:]
    
    print(f"Dataset split: {len(train_data)} train samples, {len(valid_data)} validation samples.")
    
    def save_split(data_split: list, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data_split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    save_split(train_data, TRAIN_OUTPUT)
    save_split(valid_data, VALID_OUTPUT)
    
    print(f"Data successfully split and saved to '{TRAIN_OUTPUT}' and '{VALID_OUTPUT}'.")

    # Sanity check
    if valid_data:
        print("\n--- Snippet of a validation sample ---")
        preview = valid_data[1]["text"]
        print(preview)
        print("--------------------------------------\n")

    print("You are now ready to run the MLX LoRA training command.")

if __name__ == "__main__":
    main()