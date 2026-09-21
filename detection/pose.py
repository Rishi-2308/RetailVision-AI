"""
detection/pose.py
MediaPipe Pose wrapper. Turns one video frame (or one person crop) into a
small vector of body-landmark coordinates.

All coordinates are returned NORMALISED to the FULL frame (0-1), even when the
pose is computed on a cropped person box. That makes features from the
training videos (full frame) and from uploaded videos (person crops) directly
comparable.
"""
import numpy as np

# MediaPipe landmark indices we use (head, shoulders, elbows, wrists, hips)
LANDMARKS = {
    "head": 0,
    "l_sh": 11, "r_sh": 12,
    "l_el": 13, "r_el": 14,
    "l_wr": 15, "r_wr": 16,
    "l_hip": 23, "r_hip": 24,
}
LANDMARK_ORDER = list(LANDMARKS.keys())
# 9 landmarks x (x, y) = 18 columns
POSE_COLUMNS = [f"{n}_{axis}" for n in LANDMARK_ORDER for axis in ("x", "y")]
MIN_VISIBILITY = 0.3


class PoseExtractor:
    """Creates a MediaPipe Pose model once and reuses it."""

    def __init__(self, static_image_mode=False, model_complexity=1):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt"
            ) from e
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Your MediaPipe version has no 'solutions' API. "
                "Install the pinned version: pip install mediapipe==0.10.14"
            )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=not static_image_mode,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr, bbox=None):
        """
        frame_bgr : full frame (OpenCV BGR image)
        bbox      : optional (x1, y1, x2, y2) in pixels - pose is computed on
                    this crop only (used for one tracked shopper)
        Returns a numpy array of 18 values or None if no reliable pose found.
        """
        import cv2
        h, w = frame_bgr.shape[:2]
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad_x, pad_y = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            if x2 - x1 < 20 or y2 - y1 < 20:
                return None
            crop = frame_bgr[y1:y2, x1:x2]
        else:
            x1, y1, x2, y2 = 0, 0, w, h
            crop = frame_bgr
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return None
        lms = result.pose_landmarks.landmark
        cw, ch = x2 - x1, y2 - y1
        vals, vis = [], []
        for name in LANDMARK_ORDER:
            lm = lms[LANDMARKS[name]]
            vis.append(lm.visibility)
            vals.append((x1 + lm.x * cw) / w)   # back to full-frame 0-1
            vals.append((y1 + lm.y * ch) / h)
        if float(np.mean(vis)) < MIN_VISIBILITY:
            return None
        return np.clip(np.array(vals, dtype=np.float32), 0.0, 1.0)

    def close(self):
        self._pose.close()


def motion_energy(prev_gray, gray, bbox=None):
    """
    Mean absolute pixel change between two consecutive (small, grey) frames.
    A simple, real measurement of how much movement happens (inside bbox if
    given). Both images must have the same size.
    """
    if prev_gray is None:
        return 0.0
    diff = np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
    if bbox is not None:
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = bbox
        # bbox is given in normalised coordinates here
        diff = diff[int(y1 * h):max(int(y2 * h), int(y1 * h) + 1),
                    int(x1 * w):max(int(x2 * w), int(x1 * w) + 1)]
    return float(diff.mean() / 255.0) if diff.size else 0.0
