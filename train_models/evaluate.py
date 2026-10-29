import os
import json
from config import RESULTS_DIR
from model_densenet import get_densenet_metrics, save_densenet_metrics
from model_mobilenet import get_mobilenet_metrics, save_mobilenet_metrics

def evaluate_all():
    d = get_densenet_metrics()
    m = get_mobilenet_metrics()

    save_densenet_metrics()
    save_mobilenet_metrics()

    comparison = {
        "model_comparison": [
            {
                "model": d["model"],
                "accuracy": d["accuracy"],
                "precision": d["precision"],
                "recall": d["recall"],
                "f1_score": d["f1_score"],
                "roc_auc": d["roc_auc"],
                "training_time_sec": d["training_time_sec"],
                "inference_time_ms": d["inference_time_ms"]
            },
            {
                "model": m["model"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"],
                "training_time_sec": m["training_time_sec"],
                "inference_time_ms": m["inference_time_ms"]
            }
        ]
    }

    path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"[EVAL] Model comparison saved -> {path}")
    return comparison

if __name__ == "__main__":
    evaluate_all()