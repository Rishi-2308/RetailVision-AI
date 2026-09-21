"""
analytics/dwell_time.py
Dwell time = how long one shopper stays inside one retail zone.
Input: the times (seconds) and positions (x, y normalised) of ONE tracked shopper.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from analytics.zones import zone_of_point  # noqa: E402


def format_time(seconds: float) -> str:
    """75.4 -> '01:15'  (video time, mm:ss)."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _majority_filter(labels, size=5):
    """Remove one-frame flicker between zones."""
    out, half = [], size // 2
    for i in range(len(labels)):
        chunk = labels[max(0, i - half): i + half + 1]
        out.append(Counter(chunk).most_common(1)[0][0])
    return out


def zone_visits(times, xs, ys):
    """
    Returns a list of visits:
      {"zone", "entry", "exit", "dwell"}  (seconds, video time)
    Visits shorter than config.MIN_ZONE_VISIT_SEC are dropped.
    """
    times = np.asarray(times, dtype=float)
    if len(times) == 0:
        return []
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    labels = _majority_filter([zone_of_point(x, y) for x, y in zip(xs, ys)])

    runs = []          # [zone, first_index, last_index]
    for i, z in enumerate(labels):
        if runs and runs[-1][0] == z:
            runs[-1][2] = i
        else:
            runs.append([z, i, i])

    visits = []
    for z, a, b in runs:
        if z is None:
            continue
        entry, exit_ = float(times[a]), float(times[b]) + dt
        dwell = exit_ - entry
        if dwell < config.MIN_ZONE_VISIT_SEC:
            continue
        if visits and visits[-1]["zone"] == z and entry - visits[-1]["exit"] < 0.5:
            visits[-1]["exit"], visits[-1]["dwell"] = exit_, exit_ - visits[-1]["entry"]
        else:
            visits.append({"zone": z, "entry": entry, "exit": exit_, "dwell": dwell})
    return visits
