"""
behavior/feature_extraction.py  -  PHASE 4.
Turns videos into numbers the models can learn from.

For every sampled frame (about 10 per second) we measure:
  * 9 body landmarks from MediaPipe Pose (head, shoulders, elbows, wrists,
    hips) -> 18 numbers (x, y, normalised 0-1)
  * whether the pose was found
  * motion energy (how many pixels changed compared with the previous frame)
and then derive:
  * body centre (x, y), wrist-to-shoulder distance (left/right),
    wrist speed, body speed
Everything is really computed from the video - nothing is invented.

Outputs (in dataset/processed/):
  pose/<video>.npz      cached per-frame features of every video (resumable)
  sequences.npz         LSTM data: (windows, SEQ_LEN, features) for each split
  features.csv          Random-Forest data: statistics of every window
  models/scaler.pkl     StandardScaler fitted on TRAIN frames only

Run:   python behavior/feature_extraction.py
       python behavior/feature_extraction.py --limit 3     (quick test)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from detection.pose import POSE_COLUMNS, LANDMARK_ORDER  # noqa: E402

RAW_COLUMNS = POSE_COLUMNS + ["pose_detected", "motion_energy"]        # 20
DERIVED_COLUMNS = ["body_cx", "body_cy", "wrist_sh_dist_l", "wrist_sh_dist_r",
                   "wrist_speed", "body_speed"]
CHANNEL_NAMES = POSE_COLUMNS + ["pose_detected", "motion_energy"] + DERIVED_COLUMNS  # 26
N_CHANNELS = len(CHANNEL_NAMES)
STATS = ["mean", "std", "min", "max", "delta"]
POSE_CACHE = config.PROCESSED_DIR / "pose"


def sample_step_for(fps: float) -> int:
    """Frames to skip so that we analyse about config.TARGET_FPS frames/second."""
    return max(1, int(round((fps or 30.0) / config.TARGET_FPS)))


# ------------------------------------------------------------ per video
def extract_raw_video(video_path, extractor=None, bbox_fn=None, max_seconds=0):
    """
    Read a video with OpenCV and measure the raw features of sampled frames.
    Returns dict(frame_idx, raw, fps, step).  raw has RAW_COLUMNS; landmark
    columns are NaN where no pose was found.
    """
    import cv2
    from detection.pose import PoseExtractor, motion_energy
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"OpenCV cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = sample_step_for(fps)
    own = extractor is None
    extractor = extractor or PoseExtractor(static_image_mode=False)

    idxs, rows, prev_gray, i = [], [], None, 0
    try:
        while True:
            if i % step != 0:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if max_seconds and i / fps > max_seconds:
                break
            h, w = frame.shape[:2]
            scale = config.FRAME_WIDTH / w
            small = cv2.resize(frame, (config.FRAME_WIDTH, int(h * scale)))
            gray = cv2.cvtColor(cv2.resize(small, (80, 60)), cv2.COLOR_BGR2GRAY)
            pose = extractor.extract(small)
            row = np.full(len(RAW_COLUMNS), np.nan, dtype=np.float32)
            if pose is not None:
                row[:18] = pose
                row[18] = 1.0
            else:
                row[18] = 0.0
            row[19] = motion_energy(prev_gray, gray)
            prev_gray = gray
            idxs.append(i)
            rows.append(row)
            i += 1
    finally:
        cap.release()
        if own:
            extractor.close()
    if not rows:
        raise IOError(f"No frames could be read from {video_path}")
    return {"frame_idx": np.array(idxs), "raw": np.vstack(rows),
            "fps": float(fps), "step": int(step)}


def fill_missing(raw: np.ndarray) -> np.ndarray:
    """Fill frames without a pose: linear interpolation, then nearest value,
    and 0.5 (frame centre) if a column has no value at all."""
    df = pd.DataFrame(raw[:, :18])
    df = df.interpolate(limit_direction="both")
    df = df.fillna(0.5)
    out = raw.copy()
    out[:, :18] = df.values
    return out


def add_derived(raw_filled: np.ndarray, dt: float) -> np.ndarray:
    """Add the derived movement features. dt = seconds between samples."""
    col = {n: i for i, n in enumerate(RAW_COLUMNS)}
    g = lambda name: raw_filled[:, col[name]]  # noqa: E731
    body_cx = np.mean([g("l_sh_x"), g("r_sh_x"), g("l_hip_x"), g("r_hip_x")], axis=0)
    body_cy = np.mean([g("l_sh_y"), g("r_sh_y"), g("l_hip_y"), g("r_hip_y")], axis=0)
    dist_l = np.hypot(g("l_wr_x") - g("l_sh_x"), g("l_wr_y") - g("l_sh_y"))
    dist_r = np.hypot(g("r_wr_x") - g("r_sh_x"), g("r_wr_y") - g("r_sh_y"))

    def speed(x, y):
        s = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) / max(dt, 1e-6)
        return s

    wrist_speed = np.maximum(speed(g("l_wr_x"), g("l_wr_y")), speed(g("r_wr_x"), g("r_wr_y")))
    body_speed = speed(body_cx, body_cy)
    derived = np.column_stack([body_cx, body_cy, dist_l, dist_r, wrist_speed, body_speed])
    return np.hstack([raw_filled, derived]).astype(np.float32)


def channels_from_video_dict(d) -> np.ndarray:
    """dict from extract_raw_video -> (N, 26) channel matrix."""
    dt = d["step"] / d["fps"]
    return add_derived(fill_missing(d["raw"]), dt)


def cached_channels(video_path: Path, key: str, extractor=None):
    """Load the pose cache of a video or compute it (and save it)."""
    POSE_CACHE.mkdir(parents=True, exist_ok=True)
    f = POSE_CACHE / f"{key}.npz"
    if f.exists():
        z = np.load(f)
        d = {"frame_idx": z["frame_idx"], "raw": z["raw"],
             "fps": float(z["fps"]), "step": int(z["step"])}
    else:
        d = extract_raw_video(video_path, extractor=extractor)
        np.savez_compressed(f, **d)
    detect_rate = float(np.mean(d["raw"][:, 18]))
    return channels_from_video_dict(d), d["frame_idx"], detect_rate


# ----------------------------------------------------------- per window
def resample_window(channels, frame_idx, start_frame, end_frame, seq_len=None):
    """
    Pick seq_len evenly spaced time steps between start_frame and end_frame
    (nearest sampled frame). Returns (seq_len, N_CHANNELS).
    """
    seq_len = seq_len or config.SEQ_LEN
    targets = np.linspace(start_frame, end_frame, seq_len)
    pos = np.searchsorted(frame_idx, targets)
    pos = np.clip(pos, 0, len(frame_idx) - 1)
    left = np.clip(pos - 1, 0, len(frame_idx) - 1)
    choose_left = np.abs(frame_idx[left] - targets) < np.abs(frame_idx[pos] - targets)
    pos = np.where(choose_left, left, pos)
    return channels[pos]


def window_statistics(seq: np.ndarray) -> np.ndarray:
    """(T, C) -> flat vector of C*5 statistics for the Random Forest."""
    return np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0),
                           seq[-1] - seq[0]]).astype(np.float32)


STAT_COLUMNS = [f"{c}_{s}" for s in STATS for c in CHANNEL_NAMES]   # same order


# ------------------------------------------------------------------ main
def build(limit=0):
    seg_path = config.PROCESSED_DIR / "segments.csv"
    if not seg_path.exists():
        sys.exit("[ERROR] segments.csv missing. Run  python behavior/extract_clips.py  first.")
    seg = pd.read_csv(seg_path)
    keys = sorted(seg["video_key"].unique())
    if limit:
        keys = keys[:limit]
        seg = seg[seg["video_key"].isin(keys)]
        print(f"[INFO] --limit: only {len(keys)} videos")
    from detection.pose import PoseExtractor
    extractor = PoseExtractor(static_image_mode=False)

    seqs, labels, seg_ids, splits, rates = [], [], [], [], []
    t0 = time.time()
    for n, key in enumerate(keys, 1):
        rows = seg[seg["video_key"] == key]
        vpath = config.BASE_DIR / rows.iloc[0]["video_path"]
        try:
            channels, frame_idx, rate = cached_channels(vpath, key, extractor)
        except Exception as e:  # noqa
            print(f"[WARN] {key}: {e} -> video skipped")
            continue
        rates.append(rate)
        for r in rows.itertuples():
            seqs.append(resample_window(channels, frame_idx, r.start_frame, r.end_frame))
            labels.append(r.class_id)
            seg_ids.append(r.segment_id)
            splits.append(r.split)
        print(f"[{n}/{len(keys)}] {key}: {len(rows)} windows, pose found in "
              f"{rate * 100:.0f}% of frames  ({time.time() - t0:.0f}s elapsed)")
    extractor.close()
    if not seqs:
        sys.exit("[ERROR] No windows were produced.")

    X = np.stack(seqs).astype(np.float32)
    y = np.array(labels)
    splits = np.array(splits)
    seg_ids = np.array(seg_ids)

    # ---- validation of the numbers
    print("\n=== Validation ===")
    print(f"windows: {len(X)}   shape per window: {X.shape[1:]}")
    print(f"NaN values: {int(np.isnan(X).sum())}   inf values: {int(np.isinf(X).sum())}")
    mean_rate = float(np.mean(rates))
    print(f"Average pose-detection rate: {mean_rate * 100:.1f}% of sampled frames")
    if mean_rate < 0.5:
        print("[WARN] MediaPipe found a person in fewer than half of the frames. "
              "MERL is recorded from a top-down/overhead camera, where pose "
              "estimation is harder. The models still use motion features, but "
              "expect lower accuracy - report this honestly as a limitation.")
    assert not np.isnan(X).any(), "NaN in features"

    # ---- save sequences + scaler (fit on TRAIN frames only)
    from sklearn.preprocessing import StandardScaler
    import joblib
    train_mask = splits == "train"
    if not train_mask.any():
        sys.exit("[ERROR] No training windows - cannot fit the scaler.")
    scaler = StandardScaler().fit(X[train_mask].reshape(-1, N_CHANNELS))
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)

    out = {}
    for name in ("train", "validation", "test"):
        m = splits == name
        out[f"X_{name}"], out[f"y_{name}"], out[f"id_{name}"] = X[m], y[m], seg_ids[m]
    np.savez_compressed(config.PROCESSED_DIR / "sequences.npz", **out)

    stats = np.stack([window_statistics(s) for s in X])
    df = pd.DataFrame(stats, columns=STAT_COLUMNS)
    df.insert(0, "class_id", y)
    df.insert(0, "split", splits)
    df.insert(0, "segment_id", seg_ids)
    df.to_csv(config.PROCESSED_DIR / "features.csv", index=False)
    print(f"\nSaved sequences.npz, features.csv ({df.shape[0]} rows x "
          f"{len(STAT_COLUMNS)} features) and {config.SCALER_PATH.name}")
    print(df["split"].value_counts().to_string())
    print("\nNext: python behavior/train_random_forest.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (test)")
    build(ap.parse_args().limit)
