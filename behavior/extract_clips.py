"""
behavior/extract_clips.py  -  PHASE 2 + 3.
1. Finds the MERL videos and label files inside dataset/raw/.
2. Reads the real temporal annotations (start/end frame of every action).
3. Builds  dataset/processed/annotations.csv  (one row per annotated action).
4. Turns actions into fixed windows -> segments.csv and the three split files
      dataset/train/train.csv
      dataset/validation/validation.csv
      dataset/test/test.csv
   The split is at VIDEO level: one video is never in two splits (no leakage).
   If the dataset folders are already called Train/Validation/Test, that
   official split is kept. Otherwise videos are split by subject/video id.
5. Optional  --preview  saves a picture with sample frames of every class.

Run:   python behavior/extract_clips.py
       python behavior/extract_clips.py --preview
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov"}


# ----------------------------------------------------------------- helpers
def video_key(path: Path) -> str:
    """'1_1_crop.mp4' and '1_1_label.mat' both become '1_1'."""
    m = re.match(r"^(\d+_\d+)", path.stem)
    return m.group(1) if m else re.sub(r"(_crop|_label|_labels)$", "", path.stem)


def split_from_path(path: Path):
    """Guess train/validation/test from the folder names, else None."""
    for part in reversed([p.lower() for p in path.parts]):
        if part.startswith("train"):
            return "train"
        if part.startswith("val"):
            return "validation"
        if part.startswith("test"):
            return "test"
    return None


def read_mat_labels(path: Path):
    """
    Read one MERL label file. Returns a list of (class_id, start, end) with
    class_id 0..4 following config.CLASS_NAMES and frames counted from 0.
    Expected content: a cell array with one Nx2 [start, end] matrix per class.
    """
    from scipy.io import loadmat
    data = loadmat(str(path))
    n_classes = len(config.CLASS_NAMES)
    cells = None
    for key, val in data.items():
        if key.startswith("__"):
            continue
        if getattr(val, "dtype", None) == object and val.size == n_classes:
            cells = val.ravel()
            break
    if cells is None:
        raise ValueError(
            f"{path.name}: expected a cell array with {n_classes} entries (one "
            "per action class). Run  python behavior/inspect_dataset.py  and "
            "send me the printed 'Content of first label file' block."
        )
    rows = []
    for class_id, cell in enumerate(cells):
        arr = np.asarray(cell)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(1, -1)
        if arr.shape[1] < 2:
            raise ValueError(f"{path.name}: class {class_id} has shape {arr.shape}")
        for start, end in arr[:, :2]:
            rows.append((class_id, int(start) - 1, int(end) - 1))  # MATLAB is 1-based
    return rows


def read_csv_labels(path: Path):
    """Alternative label format: CSV with class_id,start_frame,end_frame."""
    df = pd.read_csv(path)
    need = {"class_id", "start_frame", "end_frame"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path.name} needs columns {sorted(need)}")
    return [(int(r.class_id), int(r.start_frame), int(r.end_frame))
            for r in df.itertuples()]


def video_info(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"OpenCV cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


# ------------------------------------------------------------ build tables
def build_annotations():
    raw = config.RAW_DIR
    files = [p for p in raw.rglob("*") if p.is_file()]
    videos = {video_key(p): p for p in files if p.suffix.lower() in VIDEO_EXT}
    label_files = {video_key(p): p for p in files
                   if p.suffix.lower() in {".mat", ".csv"}}
    if not videos:
        sys.exit(f"[ERROR] No videos in {raw}. See README 'Getting the dataset'.")
    if not label_files:
        sys.exit(f"[ERROR] No label files (.mat/.csv) in {raw}.")

    missing = sorted(set(videos) - set(label_files))
    if missing:
        print(f"[WARN] {len(missing)} videos have no label file and are skipped: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    rows = []
    for key in sorted(set(videos) & set(label_files)):
        vpath, lpath = videos[key], label_files[key]
        try:
            fps, n_frames = video_info(vpath)
            actions = (read_mat_labels(lpath) if lpath.suffix.lower() == ".mat"
                       else read_csv_labels(lpath))
        except Exception as e:  # noqa - report and continue with other videos
            print(f"[WARN] skipping {key}: {e}")
            continue
        # use paths RELATIVE to dataset/raw, so the user's own folder names
        # (e.g. C:\\Users\\test_user\\...) can never be mistaken for a split
        split = (split_from_path(vpath.relative_to(raw))
                 or split_from_path(lpath.relative_to(raw)))
        for class_id, start, end in actions:
            if not (0 <= class_id < len(config.CLASS_NAMES)) or end <= start:
                continue
            rows.append({
                "video_key": key,
                "video_path": str(vpath.relative_to(config.BASE_DIR)),
                "split": split,
                "class_id": class_id,
                "class_name": config.CLASS_NAMES[class_id],
                "start_frame": max(start, 0),
                "end_frame": min(end, n_frames - 1) if n_frames > 0 else end,
                "fps": fps,
                "n_frames": n_frames,
            })
    if not rows:
        sys.exit("[ERROR] No valid annotations could be read. See messages above.")
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the official split if the folders provide it, else split by video."""
    keys = df.groupby("video_key")["split"].first()
    if keys.notna().all() and set(keys.unique()) >= {"train", "test"}:
        print("[INFO] Using the official split found in the dataset folders.")
        if "validation" not in set(keys.unique()):
            print("[WARN] No validation folder found: taking 20% of the train "
                  "videos as validation (video level).")
            rng = np.random.RandomState(config.RANDOM_STATE)
            train_keys = sorted(keys[keys == "train"].index)
            val_keys = set(rng.choice(train_keys, max(1, int(0.2 * len(train_keys))),
                                      replace=False))
            df["split"] = [("validation" if k in val_keys else s)
                           for k, s in zip(df["video_key"], df["split"])]
        return df

    print("[INFO] No official split folders found -> splitting by VIDEO "
          "(70% train / 15% validation / 15% test, fixed random seed).")
    rng = np.random.RandomState(config.RANDOM_STATE)
    all_keys = sorted(df["video_key"].unique())
    rng.shuffle(all_keys)
    n = len(all_keys)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    mapping = {}
    for i, k in enumerate(all_keys):
        mapping[k] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
    df["split"] = df["video_key"].map(mapping)
    return df


def make_segments(ann: pd.DataFrame) -> pd.DataFrame:
    """
    Turn every annotated action into one or more windows:
    - shorter than MIN_SEG_SEC  -> dropped
    - longer than MAX_SEG_SEC   -> cut into pieces of MAX_SEG_SEC
    """
    out = []
    for r in ann.itertuples():
        length = r.end_frame - r.start_frame + 1
        min_len = int(config.MIN_SEG_SEC * r.fps)
        max_len = int(config.MAX_SEG_SEC * r.fps)
        if length < min_len:
            continue
        start = r.start_frame
        while start <= r.end_frame:
            end = min(start + max_len - 1, r.end_frame)
            if end - start + 1 >= min_len:
                out.append({**r._asdict(), "start_frame": start, "end_frame": end})
            start = end + 1
    seg = pd.DataFrame(out).drop(columns=["Index"], errors="ignore")
    seg.insert(0, "segment_id", range(len(seg)))
    return seg


def save_tables(ann, seg):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ann.to_csv(config.PROCESSED_DIR / "annotations.csv", index=False)
    seg.to_csv(config.PROCESSED_DIR / "segments.csv", index=False)
    folders = {"train": config.DATASET_DIR / "train",
               "validation": config.DATASET_DIR / "validation",
               "test": config.DATASET_DIR / "test"}
    for name, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        seg[seg["split"] == name].to_csv(folder / f"{name}.csv", index=False)


def verify(seg):
    print("\n=== Verification ===")
    # 1. no video in two splits
    vids = seg.groupby("video_key")["split"].nunique()
    assert (vids == 1).all(), "DATA LEAKAGE: a video appears in two splits!"
    print("[OK] every video belongs to exactly one split (no leakage)")
    # 2. counts
    print("\nVideos per split:")
    print(seg.groupby("split")["video_key"].nunique().to_string())
    print("\nSegments per split and class:")
    table = seg.pivot_table(index="class_name", columns="split", values="segment_id",
                            aggfunc="count", fill_value=0)
    print(table.to_string())
    for name in ("train", "validation", "test"):
        if name not in table.columns or table[name].sum() == 0:
            print(f"[WARN] split '{name}' is empty - training will fail.")
    missing = set(config.CLASS_NAMES) - set(seg["class_name"])
    if missing:
        print(f"[WARN] classes with NO segments: {sorted(missing)}")


def preview(seg):
    """Save a picture: 5 frames from one random segment of every class."""
    import cv2
    rng = np.random.RandomState(config.RANDOM_STATE)
    rows_img = []
    for class_name in config.CLASS_NAMES:
        sub = seg[seg["class_name"] == class_name]
        if sub.empty:
            continue
        r = sub.iloc[rng.randint(len(sub))]
        cap = cv2.VideoCapture(str(config.BASE_DIR / r["video_path"]))
        frames = []
        for f in np.linspace(r["start_frame"], r["end_frame"], 5).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if ok:
                frame = cv2.resize(frame, (240, int(240 * frame.shape[0] / frame.shape[1])))
                cv2.putText(frame, class_name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1)
                frames.append(frame)
        cap.release()
        if frames:
            rows_img.append(np.hstack(frames))
    if rows_img:
        w = min(r.shape[1] for r in rows_img)
        sheet = np.vstack([r[:, :w] for r in rows_img])
        config.REPORTS_DIR.mkdir(exist_ok=True)
        out = config.REPORTS_DIR / "sample_segments.jpg"
        cv2.imwrite(str(out), sheet)
        print(f"[OK] sample frames saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="save sample frames")
    args = ap.parse_args()

    ann = assign_splits(build_annotations())
    seg = make_segments(ann)
    if seg.empty:
        sys.exit("[ERROR] No segments left after filtering.")
    save_tables(ann, seg)
    verify(seg)
    print(f"\nSaved: {config.PROCESSED_DIR / 'annotations.csv'}")
    print(f"Saved: {config.PROCESSED_DIR / 'segments.csv'} and train/validation/test csv files")
    if args.preview:
        preview(seg)
    print("\nNext: python behavior/feature_extraction.py")


if __name__ == "__main__":
    main()
