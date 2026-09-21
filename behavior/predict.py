"""
behavior/predict.py
Uses the TRAINED model (LSTM, or Random Forest baseline) to recognise the
shopping behaviour inside a tracked shopper's feature sequence.

Main entry point:   BehaviorPredictor().predict_track(channels, times)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from behavior.feature_extraction import (N_CHANNELS, window_statistics)  # noqa: E402


class ModelNotReadyError(RuntimeError):
    """Raised when the trained model files are missing or broken."""


class BehaviorPredictor:
    def __init__(self, which=None):
        which = (which or config.APP_MODEL).lower()
        self.model_name = which
        self._load(which)

    # ---------------------------------------------------------- loading
    def _load(self, which):
        import joblib
        if not config.SCALER_PATH.exists() and which == "lstm":
            raise ModelNotReadyError(
                f"{config.SCALER_PATH} not found. Train the models first "
                "(see README, Phase 3-6).")
        try:
            if which == "lstm":
                if not config.LSTM_MODEL_PATH.exists():
                    raise ModelNotReadyError(
                        f"{config.LSTM_MODEL_PATH} not found. Run "
                        "python behavior/train_lstm.py")
                from tensorflow import keras
                self.model = keras.models.load_model(config.LSTM_MODEL_PATH)
                self.scaler = joblib.load(config.SCALER_PATH)
            elif which == "rf":
                if not config.RF_MODEL_PATH.exists():
                    raise ModelNotReadyError(
                        f"{config.RF_MODEL_PATH} not found. Run "
                        "python behavior/train_random_forest.py")
                bundle = joblib.load(config.RF_MODEL_PATH)
                self.model = bundle["model"]
            else:
                raise ModelNotReadyError(f"Unknown model '{which}' (use 'lstm' or 'rf')")
        except ModelNotReadyError:
            raise
        except Exception as e:  # noqa
            raise ModelNotReadyError(f"Could not load the {which} model: {e}") from e

    # -------------------------------------------------------- prediction
    def predict_sequences(self, seqs: np.ndarray):
        """seqs: (n, SEQ_LEN, N_CHANNELS) -> probabilities (n, n_classes)."""
        if seqs.ndim != 3 or seqs.shape[2] != N_CHANNELS:
            raise ValueError(f"Expected (n, {config.SEQ_LEN}, {N_CHANNELS}), got {seqs.shape}")
        if self.model_name == "lstm":
            n, t, c = seqs.shape
            scaled = self.scaler.transform(seqs.reshape(-1, c)).reshape(n, t, c)
            return self.model.predict(scaled.astype(np.float32), verbose=0)
        stats = np.stack([window_statistics(s) for s in seqs])
        return self.model.predict_proba(stats)

    def predict_track(self, channels, frame_times):
        """
        channels    : (N, N_CHANNELS) features of ONE shopper (sampled frames)
        frame_times : (N,) time in seconds of every row
        Returns a list of windows: dict(start, end, class_id, label, confidence)
        """
        channels = np.asarray(channels, dtype=np.float32)
        times = np.asarray(frame_times, dtype=np.float64)
        if len(times) < 4:
            return []
        t0, t1 = times[0], times[-1]
        win = min(config.PRED_WINDOW_SEC, max(t1 - t0, 0.5))
        starts = np.arange(t0, max(t1 - win, t0) + 1e-6, config.PRED_STRIDE_SEC)
        seqs, bounds = [], []
        for s in starts:
            e = s + win
            targets = np.linspace(s, e, config.SEQ_LEN)
            pos = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
            seqs.append(channels[pos])
            bounds.append((float(s), float(e)))
        if not seqs:
            return []
        probs = self.predict_sequences(np.stack(seqs))
        out = []
        for (s, e), p in zip(bounds, probs):
            k = int(np.argmax(p))
            out.append({"start": s, "end": e, "class_id": k,
                        "label": config.CLASS_NAMES[k], "confidence": float(p[k])})
        return out


def windows_to_events(windows):
    """
    Merge consecutive windows with the same behaviour into events.
    Every window "votes" only for the middle PRED_STRIDE_SEC of itself, so
    overlapping windows do not inflate the duration.
    Windows below MIN_CONFIDENCE are ignored ("no clear action").
    Returns list of dict(behavior, start, end, confidence).
    """
    events = []
    half = config.PRED_STRIDE_SEC / 2.0
    for w in windows:
        if w["confidence"] < config.MIN_CONFIDENCE:
            continue
        centre = (w["start"] + w["end"]) / 2.0
        s, e = centre - half, centre + half
        if events and events[-1]["behavior"] == w["label"] and s <= events[-1]["end"] + 1e-6:
            ev = events[-1]
            ev["end"] = max(ev["end"], e)
            ev["_conf"].append(w["confidence"])
        else:
            events.append({"behavior": w["label"], "start": s, "end": e,
                           "_conf": [w["confidence"]]})
    for ev in events:
        ev["confidence"] = float(np.mean(ev.pop("_conf")))
    return events
