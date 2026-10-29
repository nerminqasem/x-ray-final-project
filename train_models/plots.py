import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR
from model_densenet import get_densenet_metrics
from model_mobilenet import get_mobilenet_metrics
from preprocessing import get_preprocessing_results

def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved -> {path}")

def plot_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Pneumonia"])
    ax.set_yticklabels(["Normal","Pneumonia"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > max(max(cm))*0.5 else "black")
    fig.colorbar(im)
    _save(fig, filename)

def plot_roc(roc, title, filename):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(roc["fpr"], roc["tpr"], label="ROC curve", color="darkorange", lw=2)
    ax.plot([0,1],[0,1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, filename)

def plot_loss(history, title, filename):
    fig, ax = plt.subplots(figsize=(6,4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"],   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(title); ax.legend()
    _save(fig, filename)

def plot_acc(history, title, filename):
    fig, ax = plt.subplots(figsize=(6,4))
    epochs = range(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, history["train_acc"], label="Train Acc")
    ax.plot(epochs, history["val_acc"],   label="Val Acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(title); ax.legend()
    _save(fig, filename)

def plot_model_comparison(comparison):
    models = [x["model"] for x in comparison["model_comparison"]]
    metrics = ["accuracy","precision","recall","f1_score","roc_auc"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9,5))
    for i, m in enumerate(comparison["model_comparison"]):
        vals = [m[k] for k in metrics]
        ax.bar(x + i*width, vals, width, label=m["model"])
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([k.replace("_"," ").title() for k in metrics])
    ax.set_ylim(0,1.05)
    ax.set_title("Model Comparison")
    ax.legend()
    _save(fig, "model_comparison.png")

def plot_preprocessing_comparison():
    data = get_preprocessing_results()
    names = [d["preprocessing"] for d in data]
    metrics = ["accuracy","precision","recall","f1_score","roc_auc"]
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10,5))
    for i, d in enumerate(data):
        vals = [d[k] for k in metrics]
        ax.bar(x + i*width, vals, width, label=d["preprocessing"])
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([k.replace("_"," ").title() for k in metrics])
    ax.set_ylim(0,1.05)
    ax.set_title("Preprocessing Comparison")
    ax.legend()
    _save(fig, "preprocessing_comparison.png")

def generate_all_plots():
    d = get_densenet_metrics()
    m = get_mobilenet_metrics()

    plot_confusion_matrix(d["confusion_matrix"], "DenseNet121 Confusion Matrix", "confusion_matrix_densenet.png")
    plot_confusion_matrix(m["confusion_matrix"], "MobileNetV3 Confusion Matrix", "confusion_matrix_mobilenet.png")

    plot_roc(d["roc_curve"], "DenseNet121 ROC Curve", "roc_curve_densenet.png")
    plot_roc(m["roc_curve"], "MobileNetV3 ROC Curve", "roc_curve_mobilenet.png")

    plot_loss(d["history"], "DenseNet121 Loss", "training_loss_densenet.png")
    plot_loss(m["history"], "MobileNetV3 Loss", "training_loss_mobilenet.png")

    plot_acc(d["history"], "DenseNet121 Accuracy", "training_acc_densenet.png")
    plot_acc(m["history"], "MobileNetV3 Accuracy", "training_acc_mobilenet.png")

    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(comparison_path, "r", encoding="utf-8") as f:
        comparison = json.load(f)

    plot_model_comparison(comparison)
    plot_preprocessing_comparison()

if __name__ == "__main__":
    generate_all_plots()