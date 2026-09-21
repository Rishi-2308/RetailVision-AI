"""
behavior/evaluation.py
Shared helpers to MEASURE model quality. All numbers come from sklearn
functions applied to real predictions - nothing is typed in by hand.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def compute_metrics(y_true, y_pred):
    """Accuracy + macro/weighted precision, recall, F1 + report + matrix."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 classification_report, confusion_matrix)
    labels = list(range(len(config.CLASS_NAMES)))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "precision_weighted": float(pw), "recall_weighted": float(rw),
        "f1_weighted": float(fw),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels,
                                        target_names=config.CLASS_NAMES,
                                        zero_division=0),
    }


def save_confusion_matrix(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics(name, split_metrics: dict, extra=None):
    """Write reports/<name>_metrics.json."""
    config.REPORTS_DIR.mkdir(exist_ok=True)
    payload = {"model": name, "splits": split_metrics}
    if extra:
        payload.update(extra)
    with open(config.REPORTS_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)


def print_metrics(title, m):
    print(f"\n----- {title} ({m['n_samples']} windows) -----")
    print(f"Accuracy          : {m['accuracy']:.4f}")
    print(f"Precision (macro) : {m['precision_macro']:.4f}")
    print(f"Recall (macro)    : {m['recall_macro']:.4f}")
    print(f"F1-score (macro)  : {m['f1_macro']:.4f}")
    print(m["report"])
