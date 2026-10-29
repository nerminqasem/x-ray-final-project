import os
import json
from config import DATASET_DIR, DATASET_STATS, RESULTS_DIR

def load_dataset_info():
    """
    """
    return DATASET_STATS

def save_dataset_statistics():
    path = os.path.join(RESULTS_DIR, "dataset_statistics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DATASET_STATS, f, indent=2)
    return path

if __name__ == "__main__":
    print("Dataset Info:", load_dataset_info())
    print("Saved at:", save_dataset_statistics())