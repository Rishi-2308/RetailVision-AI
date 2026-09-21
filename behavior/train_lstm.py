"""
behavior/train_lstm.py  -  PHASE 6 (final deep-learning model).
LSTM over the time steps of each action window:
   sequence (SEQ_LEN x 26 features) -> LSTM -> Dropout -> Dense -> Softmax(5)
Uses the scaler fitted on TRAIN data only, early stopping on validation loss,
and evaluates once on the untouched test set.

Run:  python behavior/train_lstm.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from behavior.evaluation import (compute_metrics, print_metrics,  # noqa: E402
                                 save_confusion_matrix, save_metrics)


def scale(X, scaler):
    n, t, c = X.shape
    return scaler.transform(X.reshape(-1, c)).reshape(n, t, c).astype(np.float32)


def build_model(seq_len, n_features, n_classes):
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_curves(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, title in (("loss", "Loss"), ("accuracy", "Accuracy")):
        plt.figure(figsize=(6, 4))
        plt.plot(history.history[key], label="train")
        plt.plot(history.history[f"val_{key}"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"LSTM {title}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / f"lstm_{key}_curve.png", dpi=150)
        plt.close()


def main():
    path = config.PROCESSED_DIR / "sequences.npz"
    if not path.exists() or not config.SCALER_PATH.exists():
        sys.exit("[ERROR] Run  python behavior/feature_extraction.py  first.")
    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("[ERROR] TensorFlow is not installed: pip install -r requirements.txt")
    tf.keras.utils.set_random_seed(config.RANDOM_STATE)

    d = np.load(path)
    scaler = joblib.load(config.SCALER_PATH)
    X = {s: scale(d[f"X_{s}"], scaler) for s in ("train", "validation", "test")}
    y = {s: d[f"y_{s}"] for s in ("train", "validation", "test")}
    for s in X:
        if len(X[s]) == 0:
            sys.exit(f"[ERROR] split '{s}' is empty.")
    n_classes = len(config.CLASS_NAMES)
    print("Shapes:", {s: X[s].shape for s in X})

    # balance classes (some actions are much rarer than others)
    counts = np.bincount(y["train"], minlength=n_classes).astype(float)
    weights = {i: float(counts.sum() / (n_classes * c)) for i, c in enumerate(counts) if c > 0}
    print("Class weights:", {config.CLASS_NAMES[k]: round(v, 2) for k, v in weights.items()})

    model = build_model(X["train"].shape[1], X["train"].shape[2], n_classes)
    model.summary()
    from tensorflow import keras
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    history = model.fit(X["train"], y["train"], validation_data=(X["validation"], y["validation"]),
                        epochs=100, batch_size=32, class_weight=weights,
                        callbacks=callbacks, verbose=2)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    plot_curves(history)
    results = {}
    for name in ("validation", "test"):
        pred = np.argmax(model.predict(X[name], verbose=0), axis=1)
        m = compute_metrics(y[name], pred)
        print_metrics(f"LSTM - {name}", m)
        results[name] = m
    save_confusion_matrix(results["test"]["confusion_matrix"],
                          config.REPORTS_DIR / "confusion_matrix_lstm.png",
                          "LSTM - test set")
    model.save(config.LSTM_MODEL_PATH)
    save_metrics("lstm", results, {"epochs_run": len(history.history["loss"]),
                                   "seq_len": int(X["train"].shape[1])})
    print(f"\nSaved model -> {config.LSTM_MODEL_PATH}")
    print("Saved curves and confusion matrix in reports/")
    print("Next: python behavior/compare_models.py")


if __name__ == "__main__":
    main()
