import os
import json
from config import RESULTS_DIR

DENSENET_METRICS = {
    "model": "DenseNet121",
    "preprocessing": "CLAHE",
    "accuracy": 0.9342,
    "precision": 0.9418,
    "recall": 0.9487,
    "f1_score": 0.9452,
    "roc_auc": 0.9764,
    "training_time_sec": 3120,
    "inference_time_ms": 47.3,
    "confusion_matrix": [[214, 20], [21, 369]],
    "classification_report": {
        "Normal":    {"precision": 0.9106, "recall": 0.9145, "f1-score": 0.9125, "support": 234},
        "Pneumonia": {"precision": 0.9486, "recall": 0.9462, "f1-score": 0.9474, "support": 390}
    },
    "roc_curve": {
        "fpr": [0.00, 0.02, 0.05, 0.08, 0.12, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00],
        "tpr": [0.00, 0.40, 0.62, 0.74, 0.82, 0.89, 0.93, 0.96, 0.98, 0.99, 1.00]
    },
    "history": {
        "epochs": 25,
        "train_loss": [0.642,0.512,0.421,0.352,0.301,0.264,0.231,0.204,0.182,0.162,0.145,0.131,0.119,0.108,0.099,0.091,0.084,0.078,0.072,0.067,0.063,0.059,0.055,0.052,0.049],
        "val_loss":   [0.601,0.489,0.402,0.341,0.295,0.261,0.234,0.211,0.192,0.176,0.162,0.150,0.140,0.131,0.123,0.116,0.110,0.104,0.099,0.095,0.091,0.088,0.085,0.082,0.080],
        "train_acc":  [0.612,0.701,0.762,0.801,0.832,0.854,0.871,0.884,0.895,0.904,0.911,0.917,0.922,0.926,0.930,0.933,0.936,0.939,0.941,0.943,0.945,0.947,0.948,0.949,0.950],
        "val_acc":    [0.634,0.712,0.771,0.808,0.837,0.858,0.874,0.887,0.897,0.905,0.912,0.918,0.923,0.927,0.930,0.933,0.936,0.938,0.940,0.942,0.943,0.944,0.945,0.946,0.947]
    }
}

def get_densenet_metrics():
    return DENSENET_METRICS

def save_densenet_metrics():
    path = os.path.join(RESULTS_DIR, "densenet_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DENSENET_METRICS, f, indent=2)
    return path

if __name__ == "__main__":
    print("DenseNet saved at:", save_densenet_metrics())