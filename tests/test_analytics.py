"""
tests/test_analytics.py  -  quick logic tests (no video, no AI model needed).
Run:  python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from analytics.dwell_time import zone_visits  # noqa: E402
from analytics.engagement import detect_lost_interest, engagement_for_visit, level_for  # noqa: E402
from analytics.shopper_journey import build_journey  # noqa: E402
from analytics.zones import shelf_proximity, zone_of_point  # noqa: E402
from behavior.feature_extraction import (N_CHANNELS, add_derived, fill_missing,  # noqa: E402
                                         resample_window, window_statistics)
from behavior.predict import windows_to_events  # noqa: E402


class TestZones(unittest.TestCase):
    def test_zone_of_point(self):
        self.assertEqual(zone_of_point(0.1, 0.1), "Zone A")
        self.assertEqual(zone_of_point(0.9, 0.1), "Zone B")
        self.assertEqual(zone_of_point(0.1, 0.9), "Zone C")
        self.assertEqual(zone_of_point(0.9, 0.9), "Zone D")

    def test_proximity_range(self):
        self.assertAlmostEqual(shelf_proximity(0.25, 0.25, "Zone A"), 1.0)
        self.assertEqual(shelf_proximity(1.0, 1.0, "Zone A"), 0.0)


class TestDwell(unittest.TestCase):
    def test_single_zone_dwell(self):
        t = np.arange(0, 10, 0.1)
        v = zone_visits(t, np.full_like(t, 0.2), np.full_like(t, 0.2))
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0]["zone"], "Zone A")
        self.assertAlmostEqual(v[0]["dwell"], 10.0, places=1)

    def test_two_zones(self):
        t = np.arange(0, 10, 0.1)
        x = np.where(t < 5, 0.2, 0.8)
        v = zone_visits(t, x, np.full_like(t, 0.2))
        self.assertEqual([z["zone"] for z in v], ["Zone A", "Zone B"])


class TestEngagement(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0, 12, 0.1)
        self.x = np.full_like(self.t, 0.25)
        self.y = np.full_like(self.t, 0.25)
        self.visit = {"zone": "Zone A", "entry": 0.0, "exit": 12.0, "dwell": 12.0}

    def test_high_engagement(self):
        ev = [{"behavior": "Inspect Product", "start": 1.0, "end": 11.0, "confidence": 0.9}]
        e = engagement_for_visit(self.visit, ev, self.t, self.x, self.y)
        self.assertGreater(e["score"], 80)
        self.assertIn("Product inspection detected", e["reasons"])
        self.assertLessEqual(e["score"], 100)

    def test_no_interaction_is_low(self):
        e = engagement_for_visit(self.visit, [], self.t, self.x, self.y)
        self.assertIn("No shelf interaction detected", e["reasons"])
        self.assertLess(e["score"], 60)

    def test_levels(self):
        self.assertEqual(level_for(10), "Low")
        self.assertEqual(level_for(45), "Moderate")
        self.assertEqual(level_for(70), "High")
        self.assertEqual(level_for(95), "Very High")

    def test_lost_interest(self):
        visit = {"zone": "Zone C", "entry": 2.0, "exit": 8.0, "dwell": 6.0}
        eng = {"interaction_seconds": 4.2, "behaviors": {"Inspect Shelf": 4.2}}
        lost = detect_lost_interest(visit, eng, video_end=60.0)
        self.assertEqual(lost["zone"], "Zone C")
        self.assertIsNone(detect_lost_interest(visit, {"interaction_seconds": 0, "behaviors": {}}, 60.0))


class TestJourneyAndEvents(unittest.TestCase):
    def test_journey(self):
        visits = [{"zone": "Zone A", "entry": 0, "exit": 5}, {"zone": "Zone B", "entry": 5, "exit": 9}]
        events = [{"behavior": "Inspect Shelf", "start": 1, "end": 3}]
        labels = [s["label"] for s in build_journey(visits, events)]
        self.assertEqual(labels, ["Entry", "Zone A", "Inspect Shelf", "Zone B", "Exit"])

    def test_events_merge_and_threshold(self):
        w = [{"start": 0.0, "end": 1.5, "label": "Reach to Shelf", "confidence": 0.9},
             {"start": 0.5, "end": 2.0, "label": "Reach to Shelf", "confidence": 0.8},
             {"start": 1.0, "end": 2.5, "label": "Hand in Shelf", "confidence": 0.1}]
        ev = windows_to_events(w)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["behavior"], "Reach to Shelf")


class TestFeatures(unittest.TestCase):
    def test_shapes(self):
        raw = np.random.rand(50, 20).astype(np.float32)
        raw[5:8, :18] = np.nan
        ch = add_derived(fill_missing(raw), 0.1)
        self.assertEqual(ch.shape, (50, N_CHANNELS))
        self.assertFalse(np.isnan(ch).any())
        seq = resample_window(ch, np.arange(50) * 3, 10, 120)
        self.assertEqual(seq.shape, (config.SEQ_LEN, N_CHANNELS))
        self.assertEqual(window_statistics(seq).shape, (N_CHANNELS * 5,))


class TestDatabase(unittest.TestCase):
    def test_roundtrip(self):
        tmp = tempfile.mkdtemp()
        config.DB_PATH = Path(tmp) / "test.db"
        from database import database as db
        db.init_db()
        vid = db.create_video("a.mp4", "a.mp4")
        db.save_analysis(vid, [{
            "track_id": 1, "label": "Customer #01", "start": 0, "end": 10,
            "behaviors": [{"behavior": "Inspect Product", "confidence": 0.9, "start": 1, "end": 3, "zone": "Zone A"}],
            "visits": [{"zone": "Zone A", "entry": 0, "exit": 10, "dwell": 10, "engagement_score": 70,
                        "interaction_seconds": 2}],
            "analytics": {"engagement_score": 70, "level": "High", "total_dwell": 10,
                          "dominant_behavior": "Inspect Product", "reasons": ["Close to shelf"],
                          "components": {}, "journey": [], "lost_interest": []}}])
        s = db.video_summary(vid)
        self.assertEqual(s["total_shoppers"], 1)
        self.assertEqual(s["most_visited_zone"], "Zone A")
        self.assertEqual(s["high_engagement"], 1)
        self.assertEqual(db.get_video_results(vid)[0]["label"], "Customer #01")


if __name__ == "__main__":
    unittest.main()
