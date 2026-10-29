import os
import json
from config import RESULTS_DIR, DATASET_STATS
from model_densenet import get_densenet_metrics
from model_mobilenet import get_mobilenet_metrics
from preprocessing import get_preprocessing_results

def save_all():
    d = get_densenet_metrics()
    m = get_mobilenet_metrics()

    final = {
        "dataset_statistics": DATASET_STATS,
        "models": {
            "DenseNet121": {
                "accuracy": d["accuracy"],
                "precision": d["precision"],
                "recall": d["recall"],
                "f1_score": d["f1_score"],
                "roc_auc": d["roc_auc"],
                "training_time_sec": d["training_time_sec"],
                "inference_time_ms": d["inference_time_ms"],
                "confusion_matrix": d["confusion_matrix"]
            },
            "MobileNetV3": {
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"],
                "training_time_sec": m["training_time_sec"],
                "inference_time_ms": m["inference_time_ms"],
                "confusion_matrix": m["confusion_matrix"]
            }
        },
        "preprocessing_comparison": get_preprocessing_results(),
        "best_pipeline": {
            "model": "DenseNet121",
            "preprocessing": "CLAHE",
            "accuracy": d["accuracy"],
            "roc_auc": d["roc_auc"]
        }
    }

    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"[SAVE] metrics.json saved -> {path}")
    return final

if __name__ == "__main__":
    save_all()