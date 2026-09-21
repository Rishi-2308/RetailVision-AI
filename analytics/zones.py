"""
analytics/zones.py
Retail zones are rectangles (normalised 0-1 coordinates) defined in config.ZONES.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def zone_of_point(x: float, y: float):
    """Name of the zone that contains the point (x, y), or None."""
    for name, z in config.ZONES.items():
        x1, y1, x2, y2 = z["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None


def shelf_proximity(x: float, y: float, zone_name: str) -> float:
    """
    1.0 = standing on the shelf anchor of the zone, 0.0 = far away (half of the
    zone diagonal or more). Depends on the anchors set in config.ZONES.
    """
    z = config.ZONES[zone_name]
    ax, ay = z["anchor"]
    x1, y1, x2, y2 = z["rect"]
    max_d = math.hypot(x2 - x1, y2 - y1) / 2.0
    d = math.hypot(x - ax, y - ay)
    return float(max(0.0, 1.0 - d / max_d)) if max_d > 0 else 0.0


def draw_zones(frame_bgr):
    """Draw the zone rectangles and names on a frame (returns a copy)."""
    import cv2
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for name, z in config.ZONES.items():
        x1, y1, x2, y2 = z["rect"]
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w) - 1, int(y2 * h) - 1)
        cv2.rectangle(out, p1, p2, (255, 200, 0), 1)
        cv2.putText(out, name, (p1[0] + 6, p1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 200, 0), 2)
    return out
