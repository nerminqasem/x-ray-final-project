import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset", "chest_xray")
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
CLASSES = ["NORMAL", "PNEUMONIA"]
SEED = 42


DATASET_STATS = {
    "total_images": 5863,
    "train": {"total": 5216, "normal": 1341, "pneumonia": 3875},
    "val":   {"total": 16,   "normal": 8,    "pneumonia": 8},
    "test":  {"total": 624,  "normal": 234,  "pneumonia": 390},
}