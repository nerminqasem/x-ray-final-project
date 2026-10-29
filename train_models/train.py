"""
ويحفظ histories وmetrics.
"""
import os
import json
from config import RESULTS_DIR
from model_densenet import get_densenet_metrics
from model_mobilenet import get_mobilenet_metrics

def fake_train(model_name):
    metrics = get_densenet_metrics() if model_name == "DenseNet121" else get_mobilenet_metrics()
    history_path = os.path.join(RESULTS_DIR, f"history_{model_name.lower().replace('121','').replace('v3','')}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(metrics["history"], f, indent=2)
    print(f"[TRAIN] {model_name} finished. History saved -> {history_path}")
    return history_path

if __name__ == "__main__":
    fake_train("DenseNet121")
    fake_train("MobileNetV3")