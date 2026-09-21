"""
analytics/shopper_journey.py
Builds the ordered journey of ONE shopper:
Entry -> Zone A -> Inspect Shelf -> Zone B -> Reach to Shelf -> ... -> Exit
"""


def build_journey(visits, events):
    """
    visits : list of dict(zone, entry, exit)
    events : list of dict(behavior, start, end)
    Returns a list of steps: {"type": "entry|zone|behavior|exit", "label", "time"}
    """
    steps = [{"type": "entry", "label": "Entry", "time": visits[0]["entry"] if visits else 0.0}]
    for v in sorted(visits, key=lambda v: v["entry"]):
        steps.append({"type": "zone", "label": v["zone"], "time": v["entry"]})
        inside = [e for e in events
                  if e["start"] < v["exit"] and e["end"] > v["entry"]]
        last = None
        for e in sorted(inside, key=lambda e: e["start"]):
            if e["behavior"] != last:            # skip immediate repeats
                steps.append({"type": "behavior", "label": e["behavior"],
                              "time": max(e["start"], v["entry"])})
                last = e["behavior"]
    steps.append({"type": "exit", "label": "Exit", "time": visits[-1]["exit"] if visits else 0.0})
    return steps
