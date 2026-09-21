"""
config.py - ONE place for every setting in the project.
Change values here instead of editing many files.
All paths are relative to the project folder, so the project works on any PC.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"              # put the MERL dataset here
PROCESSED_DIR = DATASET_DIR / "processed"  # pose cache, annotations, features
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "processed"
DB_PATH = BASE_DIR / "retailvision.db"

RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
LSTM_MODEL_PATH = MODELS_DIR / "behavior_lstm.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABELS_PATH = MODELS_DIR / "class_names.json"

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_MB = 500

# ------------------------------------------------------- behaviour classes
# MERL Shopping Dataset action classes (same order as the label file).
CLASS_NAMES = [
    "Reach to Shelf",
    "Retract from Shelf",
    "Hand in Shelf",
    "Inspect Product",
    "Inspect Shelf",
]

# ------------------------------------------------------ sampling / windows
# We analyse about 10 frames per second (for a 30 fps video: every 3rd frame).
# Reason: shopping actions last about 1-3 seconds, so 10 fps still gives 10-30
# frames per action but costs 3x less time in MediaPipe than full frame-rate.
TARGET_FPS = 10
FRAME_WIDTH = 640          # frames are resized to this width (keeps aspect ratio)

# LSTM input: every action segment is resampled to SEQ_LEN time steps.
# 16 steps ~ 1.6 s at 10 fps, i.e. about the length of a typical action, and it
# is small enough to train on a normal laptop CPU.
SEQ_LEN = 16
MIN_SEG_SEC = 0.5          # ignore annotated segments shorter than this
MAX_SEG_SEC = 3.0          # longer segments are cut into windows of this length

# Sliding window used at prediction time on uploaded videos
PRED_WINDOW_SEC = 1.5
PRED_STRIDE_SEC = 0.5
MIN_CONFIDENCE = 0.45      # windows below this are shown as "No clear action"

# Which model the Flask app uses: "lstm" (final model) or "rf" (baseline)
APP_MODEL = "lstm"

RANDOM_STATE = 42

# ------------------------------------------------------------- detection
YOLO_WEIGHTS = "yolov8n.pt"     # downloaded automatically on first use
YOLO_CONF = 0.35
TRACKER_CONFIG = "bytetrack.yaml"   # ByteTrack, built into Ultralytics
PROCESS_MAX_SECONDS = 0         # 0 = whole video; e.g. 60 to test quickly
MIN_TRACK_SECONDS = 1.0         # shorter tracks are treated as noise

# ------------------------------------------------------------ retail zones
# Zones are rectangles in NORMALISED coordinates (0-1) of the video frame:
# (x1, y1, x2, y2). "anchor" is the point of the shelf that belongs to the
# zone. Default = 2x2 grid with the shelf anchor at each zone centre.
# >>> For a real store video, EDIT these numbers to match the real shelves. <<<
ZONES = {
    "Zone A": {"rect": (0.0, 0.0, 0.5, 0.5), "anchor": (0.25, 0.25)},
    "Zone B": {"rect": (0.5, 0.0, 1.0, 0.5), "anchor": (0.75, 0.25)},
    "Zone C": {"rect": (0.0, 0.5, 0.5, 1.0), "anchor": (0.25, 0.75)},
    "Zone D": {"rect": (0.5, 0.5, 1.0, 1.0), "anchor": (0.75, 0.75)},
}
MIN_ZONE_VISIT_SEC = 0.5   # shorter zone stays are ignored (flicker)

# ------------------------------------------------- engagement (project-defined)
ENGAGEMENT_WEIGHTS = {"duration": 0.40, "proximity": 0.30,
                      "interaction": 0.20, "movement": 0.10}
DURATION_CAP_SEC = 10.0    # interaction time that gives duration_score = 1
SPEED_CAP = 0.30           # normalised frame-widths per second = "very mobile"
BEHAVIOR_WEIGHTS = {       # how strongly each behaviour counts as interaction
    "Reach to Shelf": 0.6,
    "Retract from Shelf": 0.4,
    "Hand in Shelf": 0.8,
    "Inspect Product": 1.0,
    "Inspect Shelf": 0.7,
}
# Project-defined bands (NOT scientifically validated)
ENGAGEMENT_BANDS = [(30, "Low"), (60, "Moderate"), (80, "High"), (100, "Very High")]
LOST_INTEREST_MAX_SEC = 5.0

HEATMAP_GRID = 64
