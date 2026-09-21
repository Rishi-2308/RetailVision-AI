"""
detection/person_detector.py
YOLO person detection through Ultralytics.
YOLO is ONLY used to find people (class 0). It does NOT recognise behaviour -
that is done by our own trained Random Forest / LSTM models.
"""
import config


def load_yolo(weights=None):
    """Load the YOLO model (downloads yolov8n.pt the first time)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from e
    try:
        return YOLO(weights or config.YOLO_WEIGHTS)
    except Exception as e:
        raise RuntimeError(
            f"Could not load YOLO weights '{weights or config.YOLO_WEIGHTS}'. "
            "The first run needs internet to download them. Original error: "
            f"{e}"
        ) from e


def detect_persons(model, frame_bgr, conf=None):
    """
    Detect persons in ONE frame (no tracking).
    Returns a list of dicts: {"bbox": (x1,y1,x2,y2), "confidence": float}
    """
    results = model.predict(frame_bgr, classes=[0], conf=conf or config.YOLO_CONF,
                            verbose=False)
    persons = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        persons.append({"bbox": (x1, y1, x2, y2), "confidence": float(box.conf[0])})
    return persons
