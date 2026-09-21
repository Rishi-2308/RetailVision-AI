"""
behavior/compare_models.py
Builds the Random Forest vs LSTM table from the REAL saved metrics
(reports/random_forest_metrics.json and reports/lstm_metrics.json).

Run:  python behavior/compare_models.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main():
    rows = []
    for label, fname in (("Random Forest", "random_forest_metrics.json"),
                         ("LSTM", "lstm_metrics.json")):
        f = config.REPORTS_DIR / fname
        if not f.exists():
            print(f"[WARN] {fname} not found - train that model first.")
            continue
        t = json.load(open(f))["splits"]["test"]
        rows.append((label, t["accuracy"], t["precision_macro"], t["recall_macro"], t["f1_macro"]))
    if not rows:
        sys.exit("[ERROR] no metrics found.")
    lines = ["| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |",
             "|---|---:|---:|---:|---:|"]
    lines += [f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |" for r in rows]
    text = "\n".join(lines)
    print("TEST-SET COMPARISON (measured)\n")
    print(text)
    (config.REPORTS_DIR / "model_comparison.md").write_text(text + "\n")
    print("\nSaved reports/model_comparison.md")


if __name__ == "__main__":
    main()
