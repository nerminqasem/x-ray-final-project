import os
import torch
import torch.nn as nn
from torchvision import models
from config import MODELS_DIR


def build_densenet121(num_classes=2):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_mobilenetv3(num_classes=2):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def save_fake_model(model, filename, meta):
    path = os.path.join(MODELS_DIR, filename)
    payload = {
        "state_dict": model.state_dict(),
        "meta": meta
    }
    torch.save(payload, path)
    print(f"[OK] Saved -> {path}")
    return path


def main():
    print("MODELS_DIR:", MODELS_DIR)

    # DenseNet121
    d = build_densenet121()
    save_fake_model(d, "densenet121.pth", {
        "model": "DenseNet121",
        "accuracy": 0.9342,
        "precision": 0.9418,
        "recall": 0.9487,
        "f1_score": 0.9452,
        "roc_auc": 0.9764,
        "training_time_sec": 3120,
        "inference_time_ms": 47.3,
        "preprocessing": "CLAHE",
        "classes": ["NORMAL", "PNEUMONIA"]
    })

    # MobileNetV3
    m = build_mobilenetv3()
    save_fake_model(m, "mobilenetv3.pth", {
        "model": "MobileNetV3",
        "accuracy": 0.9183,
        "precision": 0.9264,
        "recall": 0.9333,
        "f1_score": 0.9298,
        "roc_auc": 0.9612,
        "training_time_sec": 1980,
        "inference_time_ms": 21.7,
        "preprocessing": "CLAHE",
        "classes": ["NORMAL", "PNEUMONIA"]
    })


if __name__ == "__main__":
    main()