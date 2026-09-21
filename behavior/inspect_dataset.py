"""
behavior/inspect_dataset.py  -  PHASE 2, step 1.
Looks at what is REALLY inside dataset/raw/ (folders, videos, label files) so
that we never guess the dataset format.

Run:  python behavior/inspect_dataset.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def print_tree(root: Path, max_depth=3, max_items=8):
    """Print a small folder tree."""
    def walk(folder: Path, depth: int):
        if depth > max_depth:
            return
        items = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name))
        for shown, item in enumerate(items):
            if shown >= max_items:
                print("  " * depth + f"... ({len(items) - max_items} more)")
                break
            print("  " * depth + ("[DIR] " if item.is_dir() else "") + item.name)
            if item.is_dir():
                walk(item, depth + 1)
    print(root)
    walk(root, 1)


def describe_mat(path: Path):
    """Print the content of a .mat label file."""
    from scipy.io import loadmat
    data = loadmat(str(path))
    for key, val in data.items():
        if key.startswith("__"):
            continue
        print(f"   key '{key}': type={type(val).__name__}, "
              f"shape={getattr(val, 'shape', None)}, dtype={getattr(val, 'dtype', None)}")
        if getattr(val, "dtype", None) == object:
            for i, cell in enumerate(val.ravel()[:6]):
                arr = getattr(cell, "shape", None)
                print(f"      cell[{i}] shape={arr}  first rows="
                      f"{cell[:3].tolist() if hasattr(cell, 'tolist') else cell}")


def main():
    root = config.RAW_DIR
    if not root.exists() or not any(p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep"):
        print(f"[ERROR] {root} is empty.\n"
              "Download the MERL Shopping Dataset (see README.md, section "
              "'Getting the dataset') and unzip it INSIDE dataset/raw/.")
        sys.exit(1)

    print("=== Folder tree (first levels) ===")
    print_tree(root)

    files = [p for p in root.rglob("*") if p.is_file()]
    ext_count = Counter(p.suffix.lower() for p in files)
    print("\n=== File types found ===")
    for ext, n in ext_count.most_common():
        print(f"  {ext or '(no extension)'}: {n}")

    videos = sorted(p for p in files if p.suffix.lower() in {".mp4", ".avi", ".mov"})
    labels = sorted(p for p in files if p.suffix.lower() in {".mat", ".csv", ".txt", ".json"})
    print(f"\nVideos found: {len(videos)}   Possible label files: {len(labels)}")

    if videos:
        import cv2
        print("\n=== First video properties ===")
        cap = cv2.VideoCapture(str(videos[0]))
        if not cap.isOpened():
            print(f"[ERROR] OpenCV cannot open {videos[0]}")
        else:
            print(f"  file      : {videos[0].relative_to(root)}")
            print(f"  size      : {int(cap.get(3))} x {int(cap.get(4))}")
            print(f"  fps       : {cap.get(5):.2f}")
            print(f"  frames    : {int(cap.get(7))}")
            print(f"  duration  : {cap.get(7) / max(cap.get(5), 1):.1f} s")
        cap.release()

    mats = [p for p in labels if p.suffix.lower() == ".mat"]
    if mats:
        print(f"\n=== Content of first label file: {mats[0].relative_to(root)} ===")
        try:
            describe_mat(mats[0])
        except Exception as e:  # noqa
            print(f"[ERROR] could not read label file: {e}")
    print("\nNext: python behavior/extract_clips.py")


if __name__ == "__main__":
    main()
