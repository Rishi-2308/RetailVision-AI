"""
behavior/train_random_forest.py  -  PHASE 5 (baseline model).
Trains a Random Forest on features.csv (statistics of every action window),
evaluates on validation and test, saves the model and the real metrics.

Run:  python behavior/train_random_forest.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from behavior.evaluation import (compute_metrics, print_metrics,  # noqa: E402
                                 save_confusion_matrix, save_metrics)


def main():
    from sklearn.ensemble import RandomForestClassifier
    path = config.PROCESSED_DIR / "features.csv"
    if not path.exists():
        sys.exit("[ERROR] features.csv missing. Run  python behavior/feature_extraction.py")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in ("segment_id", "split", "class_id")]
    parts = {s: df[df["split"] == s] for s in ("train", "validation", "test")}
    for s, p in parts.items():
        if p.empty:
            sys.exit(f"[ERROR] split '{s}' has no rows.")
    X_train, y_train = parts["train"][feat_cols].values, parts["train"]["class_id"].values

    print(f"Training Random Forest on {len(X_train)} windows, {len(feat_cols)} features ...")
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_STATE)
    rf.fit(X_train, y_train)

    results = {}
    for name in ("validation", "test"):
        p = parts[name]
        pred = rf.predict(p[feat_cols].values)
        m = compute_metrics(p["class_id"].values, pred)
        print_metrics(f"Random Forest - {name}", m)
        results[name] = m
    config.REPORTS_DIR.mkdir(exist_ok=True)
    save_confusion_matrix(results["test"]["confusion_matrix"],
                          config.REPORTS_DIR / "confusion_matrix_rf.png",
                          "Random Forest - test set")

    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    imp.head(20).to_csv(config.REPORTS_DIR / "rf_top_features.csv", header=["importance"])
    print("\nTop 10 most important features:")
    print(imp.head(10).to_string())

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": rf, "feature_columns": feat_cols}, config.RF_MODEL_PATH)
    with open(config.LABELS_PATH, "w") as f:
        json.dump(config.CLASS_NAMES, f)
    save_metrics("random_forest", {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()})
    print(f"\nSaved model -> {config.RF_MODEL_PATH}")
    print("Saved confusion matrix -> reports/confusion_matrix_rf.png")
    print("Next: python behavior/train_lstm.py")


if __name__ == "__main__":
    main()
