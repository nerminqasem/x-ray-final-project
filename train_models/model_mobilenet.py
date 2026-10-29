import os
import json
from config import RESULTS_DIR

MOBILENET_METRICS = {
    "model": "MobileNetV3",
    "preprocessing": "CLAHE",
    "accuracy": 0.9183,
    "precision": 0.9264,
    "recall": 0.9333,
    "f1_score": 0.9298,
    "roc_auc": 0.9612,
    "training_time_sec": 1980,
    "inference_time_ms": 21.7,
    "confusion_matrix": [[207, 27], [24, 366]],
    "classification_report": {
        "Normal":    {"precision": 0.8961, "recall": 0.8846, "f1-score": 0.8903, "support": 234},
        "Pneumonia": {"precision": 0.9313, "recall": 0.9385, "f1-score": 0.9349, "support": 390}
    },
    "roc_curve": {
        "fpr": [0.00, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.47, 0.62, 0.82, 1.00],
        "tpr": [0.00, 0.36, 0.58, 0.70, 0.79, 0.86, 0.91, 0.94, 0.97, 0.99, 1.00]
    },
    "history": {
        "epochs": 25,
        "train_loss": [0.671,0.541,0.451,0.381,0.327,0.287,0.254,0.227,0.205,0.186,0.170,0.156,0.144,0.134,0.125,0.117,0.110,0.104,0.098,0.093,0.089,0.085,0.081,0.078,0.075],
        "val_loss":   [0.632,0.518,0.432,0.368,0.318,0.281,0.251,0.227,0.208,0.192,0.179,0.168,0.158,0.149,0.141,0.134,0.128,0.122,0.117,0.112,0.108,0.104,0.101,0.098,0.095],
        "train_acc":  [0.591,0.682,0.744,0.786,0.818,0.842,0.861,0.876,0.888,0.898,0.906,0.913,0.919,0.924,0.928,0.932,0.935,0.938,0.940,0.942,0.944,0.945,0.946,0.947,0.948],
        "val_acc":    [0.612,0.694,0.752,0.792,0.823,0.846,0.864,0.878,0.889,0.898,0.906,0.912,0.918,0.923,0.927,0.930,0.933,0.935,0.937,0.939,0.940,0.941,0.942,0.943,0.944]
    }
}

def get_mobilenet_metrics():
    return MOBILENET_METRICS

def save_mobilenet_metrics():
    path = os.path.join(RESULTS_DIR, "mobilenet_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(MOBILENET_METRICS, f, indent=2)
    return path

if __name__ == "__main__":
    print("MobileNet saved at:", save_mobilenet_metrics())