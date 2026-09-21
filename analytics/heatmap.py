"""
analytics/heatmap.py
Retail Zone Interaction Heatmap.
NOT an eye-tracking heatmap: it shows where shoppers spent time (weight 1 per
sample) and where behaviours were recognised (extra weight).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def build_heatmap(xs, ys, weights, background_bgr, out_path):
    """
    xs, ys     : normalised positions (0-1)
    weights    : one weight per position (>= 0)
    background : frame (BGR image) to draw the heat on
    Returns True if the file was written, False if there was no data.
    """
    import cv2
    xs, ys, weights = np.asarray(xs), np.asarray(ys), np.asarray(weights, dtype=float)
    if len(xs) == 0:
        return False
    g = config.HEATMAP_GRID
    grid = np.zeros((g, g), dtype=np.float32)
    ix = np.clip((xs * g).astype(int), 0, g - 1)
    iy = np.clip((ys * g).astype(int), 0, g - 1)
    np.add.at(grid, (iy, ix), weights)
    grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=2.0)
    if grid.max() <= 0:
        return False
    grid = grid / grid.max()
    h, w = background_bgr.shape[:2]
    heat = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(heat, 0, 1)[..., None] * 0.7
    blended = (background_bgr * (1 - alpha) + heat_color * alpha).astype(np.uint8)
    cv2.imwrite(str(out_path), blended)
    return True
