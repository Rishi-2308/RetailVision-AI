"""
detection/tracker.py
ByteTrack person tracking through Ultralytics (model.track).
Gives every person a stable ID across frames so we can measure movement,
dwell time and behaviour PER SHOPPER.
"""
import config


def track_frame(model, frame_bgr, conf=None):
    """
    Detect + track persons in one frame. Call it on consecutive frames of the
    same video (persist=True keeps the IDs).
    Returns a list of dicts: {"track_id": int, "bbox": (x1,y1,x2,y2),
                              "confidence": float}
    """
    results = model.track(frame_bgr, persist=True, classes=[0],
                          conf=conf or config.YOLO_CONF,
                          tracker=config.TRACKER_CONFIG, verbose=False)
    out = []
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:      # nobody tracked in this frame
        return out
    ids = boxes.id.int().tolist()
    for tid, box in zip(ids, boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append({"track_id": int(tid), "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0])})
    return out


def reset_tracker(model):
    """Forget old IDs before starting a new video (drops the internal predictor)."""
    model.predictor = None
