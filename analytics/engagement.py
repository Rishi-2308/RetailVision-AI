"""
analytics/engagement.py
VISUAL SHOPPER ENGAGEMENT SCORE  (project-defined behavioural metric)

It is NOT a measurement of thoughts, emotions, purchase intention or true
attention. It only combines four observable signals into a 0-100 number:

  score = 100 * ( 0.40 * duration_score      time spent interacting
                + 0.30 * proximity_score     closeness to the shelf anchor
                + 0.20 * interaction_score   kind of behaviour detected
                + 0.10 * movement_score )    steadiness of the body

The bands (Low / Moderate / High / Very High) are project-defined thresholds.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from analytics.zones import shelf_proximity  # noqa: E402


def level_for(score: float) -> str:
    for upper, name in config.ENGAGEMENT_BANDS:
        if score <= upper:
            return name
    return config.ENGAGEMENT_BANDS[-1][1]


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def engagement_for_visit(visit, events, times, xs, ys):
    """
    visit  : dict(zone, entry, exit, dwell)
    events : behaviour events of the same shopper (behavior,start,end,confidence)
    times/xs/ys : positions of the shopper (video seconds, normalised x, y)
    Returns dict(score, level, components, reasons, interaction_seconds,
                 behaviors)
    """
    times, xs, ys = np.asarray(times), np.asarray(xs), np.asarray(ys)
    inside = (times >= visit["entry"]) & (times <= visit["exit"])

    # -- interaction time and kinds of behaviour inside this visit
    inter_sec, weighted, kinds = 0.0, 0.0, {}
    for ev in events:
        o = _overlap(ev["start"], ev["end"], visit["entry"], visit["exit"])
        if o <= 0:
            continue
        inter_sec += o
        weighted += o * config.BEHAVIOR_WEIGHTS.get(ev["behavior"], 0.5)
        kinds[ev["behavior"]] = kinds.get(ev["behavior"], 0.0) + o

    duration_score = min(inter_sec / config.DURATION_CAP_SEC, 1.0)
    interaction_score = (weighted / inter_sec) if inter_sec > 0 else 0.0
    if inside.sum() > 0:
        proximity_score = float(np.mean([shelf_proximity(x, y, visit["zone"])
                                         for x, y in zip(xs[inside], ys[inside])]))
    else:
        proximity_score = 0.0
    if inside.sum() > 1:
        dt = np.diff(times[inside])
        step = np.hypot(np.diff(xs[inside]), np.diff(ys[inside]))
        speed = float(np.mean(step / np.maximum(dt, 1e-6)))
    else:
        speed = 0.0
    movement_score = 1.0 - min(speed / config.SPEED_CAP, 1.0)

    w = config.ENGAGEMENT_WEIGHTS
    score = 100.0 * (w["duration"] * duration_score + w["proximity"] * proximity_score
                     + w["interaction"] * interaction_score + w["movement"] * movement_score)
    score = float(np.clip(score, 0, 100))

    # -- reasons: only statements backed by the numbers above
    reasons = []
    if duration_score >= 0.5:
        reasons.append(f"Long shelf interaction ({inter_sec:.1f} s)")
    elif inter_sec > 0:
        reasons.append(f"Short shelf interaction ({inter_sec:.1f} s)")
    else:
        reasons.append("No shelf interaction detected")
    if proximity_score >= 0.6:
        reasons.append("Close to shelf")
    elif proximity_score < 0.3:
        reasons.append("Far from shelf")
    if "Inspect Product" in kinds:
        reasons.append("Product inspection detected")
    if "Hand in Shelf" in kinds:
        reasons.append("Hand-in-shelf interaction detected")
    if movement_score >= 0.7 and inter_sec > 0:
        reasons.append("Stable interaction period")
    elif movement_score < 0.4:
        reasons.append("High body movement (moving through)")

    return {
        "score": round(score, 1), "level": level_for(score),
        "components": {"duration": round(duration_score, 3),
                       "proximity": round(proximity_score, 3),
                       "interaction": round(interaction_score, 3),
                       "movement": round(movement_score, 3)},
        "reasons": reasons, "interaction_seconds": round(inter_sec, 2),
        "behaviors": kinds,
    }


def detect_lost_interest(visit, eng, video_end):
    """
    'Potential Lost Interest' (behavioural analytics, NOT a psychological
    conclusion): the shopper interacted with the shelf but only very briefly
    and then left the zone before the video ended.
    Returns a dict or None.
    """
    inter = eng["interaction_seconds"]
    left_zone = visit["exit"] < video_end - 1.0
    if 0 < inter < config.LOST_INTEREST_MAX_SEC and left_zone:
        dominant = max(eng["behaviors"], key=eng["behaviors"].get)
        return {"zone": visit["zone"], "interaction_seconds": inter,
                "behavior": dominant, "outcome": "Left zone"}
    return None
