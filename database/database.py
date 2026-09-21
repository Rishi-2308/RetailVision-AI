"""
database/database.py
SQLite storage (Python's built-in sqlite3 - no extra library needed).

Tables
  videos     one row per uploaded video (+ processing status)
  shoppers   one row per tracked shopper ("Customer #01") in a video
  behaviors  recognised behaviour events of a shopper (with confidence, zone)
  zones      zone visits: entry, exit, dwell time, engagement of the visit
  analytics  one row per shopper: engagement score, total dwell, dominant
             behaviour, reasons, journey, potential-lost-interest records
"""
import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    original_name TEXT,
    uploaded_at TEXT,
    status TEXT DEFAULT 'uploaded',
    message TEXT,
    duration REAL, fps REAL, width INTEGER, height INTEGER,
    model_used TEXT,
    annotated_video TEXT, heatmap_image TEXT, preview_image TEXT
);
CREATE TABLE IF NOT EXISTS shoppers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    track_id INTEGER, label TEXT,
    start_time REAL, end_time REAL
);
CREATE TABLE IF NOT EXISTS behaviors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shopper_id INTEGER NOT NULL REFERENCES shoppers(id) ON DELETE CASCADE,
    behavior TEXT, confidence REAL, start_time REAL, end_time REAL, zone TEXT
);
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shopper_id INTEGER NOT NULL REFERENCES shoppers(id) ON DELETE CASCADE,
    zone TEXT, entry_time REAL, exit_time REAL, dwell_time REAL,
    engagement_score REAL, interaction_seconds REAL
);
CREATE TABLE IF NOT EXISTS analytics (
    shopper_id INTEGER PRIMARY KEY REFERENCES shoppers(id) ON DELETE CASCADE,
    engagement_score REAL, engagement_level TEXT,
    total_dwell_time REAL, dominant_behavior TEXT,
    reasons TEXT, journey TEXT, lost_interest TEXT, components TEXT
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with connect() as c:
        c.executescript(SCHEMA)


# ------------------------------------------------------------------ videos
def create_video(filename, original_name):
    with connect() as c:
        cur = c.execute(
            "INSERT INTO videos (filename, original_name, uploaded_at, status) VALUES (?,?,?,?)",
            (filename, original_name, datetime.now().isoformat(timespec="seconds"), "uploaded"))
        return cur.lastrowid


def update_video(video_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE videos SET {cols} WHERE id=?", (*fields.values(), video_id))


def get_video(video_id):
    with connect() as c:
        row = c.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        return dict(row) if row else None


def list_videos():
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM videos ORDER BY id DESC")]


def clear_video_results(video_id):
    with connect() as c:
        c.execute("DELETE FROM shoppers WHERE video_id=?", (video_id,))


# ---------------------------------------------------------------- results
def save_analysis(video_id, shoppers):
    """
    shoppers: list of dict
      track_id, label, start, end,
      behaviors: [{behavior, confidence, start, end, zone}]
      visits:    [{zone, entry, exit, dwell, engagement_score, interaction_seconds}]
      analytics: {engagement_score, level, total_dwell, dominant_behavior,
                  reasons, journey, lost_interest, components}
    """
    clear_video_results(video_id)
    with connect() as c:
        for s in shoppers:
            cur = c.execute(
                "INSERT INTO shoppers (video_id, track_id, label, start_time, end_time) VALUES (?,?,?,?,?)",
                (video_id, s["track_id"], s["label"], s["start"], s["end"]))
            sid = cur.lastrowid
            c.executemany(
                "INSERT INTO behaviors (shopper_id, behavior, confidence, start_time, end_time, zone) VALUES (?,?,?,?,?,?)",
                [(sid, b["behavior"], b["confidence"], b["start"], b["end"], b.get("zone"))
                 for b in s["behaviors"]])
            c.executemany(
                "INSERT INTO zones (shopper_id, zone, entry_time, exit_time, dwell_time, engagement_score, interaction_seconds) VALUES (?,?,?,?,?,?,?)",
                [(sid, v["zone"], v["entry"], v["exit"], v["dwell"],
                  v["engagement_score"], v["interaction_seconds"]) for v in s["visits"]])
            a = s["analytics"]
            c.execute(
                "INSERT INTO analytics (shopper_id, engagement_score, engagement_level, total_dwell_time, dominant_behavior, reasons, journey, lost_interest, components) VALUES (?,?,?,?,?,?,?,?,?)",
                (sid, a["engagement_score"], a["level"], a["total_dwell"],
                 a["dominant_behavior"], json.dumps(a["reasons"]), json.dumps(a["journey"]),
                 json.dumps(a["lost_interest"]), json.dumps(a["components"])))


def get_video_results(video_id):
    """Everything needed for the results page of one video."""
    with connect() as c:
        shoppers = []
        for s in c.execute("SELECT * FROM shoppers WHERE video_id=? ORDER BY id", (video_id,)):
            s = dict(s)
            a = c.execute("SELECT * FROM analytics WHERE shopper_id=?", (s["id"],)).fetchone()
            a = dict(a) if a else {}
            for k in ("reasons", "journey", "lost_interest", "components"):
                a[k] = json.loads(a[k]) if a.get(k) else ([] if k != "components" else {})
            s["analytics"] = a
            s["behaviors"] = [dict(r) for r in c.execute(
                "SELECT * FROM behaviors WHERE shopper_id=? ORDER BY start_time", (s["id"],))]
            s["visits"] = [dict(r) for r in c.execute(
                "SELECT * FROM zones WHERE shopper_id=? ORDER BY entry_time", (s["id"],))]
            shoppers.append(s)
        return shoppers


def video_summary(video_id=None):
    """Numbers for cards + charts. video_id=None -> all videos together."""
    where, params = ("WHERE s.video_id=?", (video_id,)) if video_id else ("", ())
    with connect() as c:
        r = c.execute(f"""SELECT COUNT(*) n, AVG(a.total_dwell_time) dwell,
                                 AVG(a.engagement_score) eng,
                                 SUM(CASE WHEN a.engagement_score > 60 THEN 1 ELSE 0 END) high
                          FROM shoppers s JOIN analytics a ON a.shopper_id = s.id {where}""",
                      params).fetchone()
        total = r["n"] or 0

        zone_rows = c.execute(f"""SELECT z.zone zone, COUNT(DISTINCT z.shopper_id) shoppers,
                                         AVG(z.dwell_time) avg_dwell,
                                         AVG(z.engagement_score) avg_eng
                                  FROM zones z JOIN shoppers s ON s.id = z.shopper_id {where}
                                  GROUP BY z.zone ORDER BY z.zone""", params).fetchall()
        beh_rows = c.execute(f"""SELECT b.behavior behavior, COUNT(*) n
                                 FROM behaviors b JOIN shoppers s ON s.id = b.shopper_id {where}
                                 GROUP BY b.behavior ORDER BY n DESC""", params).fetchall()
        lvl_rows = c.execute(f"""SELECT a.engagement_level lvl, COUNT(*) n
                                 FROM analytics a JOIN shoppers s ON s.id = a.shopper_id {where}
                                 GROUP BY a.engagement_level""", params).fetchall()
        bz_rows = c.execute(f"""SELECT b.zone zone, b.behavior behavior, COUNT(*) n
                                FROM behaviors b JOIN shoppers s ON s.id = b.shopper_id {where}
                                {'AND' if where else 'WHERE'} b.zone IS NOT NULL
                                GROUP BY b.zone, b.behavior""", params).fetchall()

    zones = [dict(z) for z in zone_rows]
    most_zone = max(zones, key=lambda z: z["shoppers"])["zone"] if zones else "-"
    most_beh = beh_rows[0]["behavior"] if beh_rows else "-"
    order = [n for _, n in config.ENGAGEMENT_BANDS]
    levels = {r["lvl"]: r["n"] for r in lvl_rows}
    return {
        "total_shoppers": total,
        "avg_dwell": round(r["dwell"] or 0, 1),
        "avg_engagement": round(r["eng"] or 0, 1),
        "high_engagement": int(r["high"] or 0),
        "most_visited_zone": most_zone,
        "most_common_behavior": most_beh,
        "zones": zones,
        "behavior_distribution": {r["behavior"]: r["n"] for r in beh_rows},
        "engagement_distribution": {lvl: levels.get(lvl, 0) for lvl in order},
        "behavior_by_zone": [dict(r) for r in bz_rows],
    }
if __name__ == "__main__":
    init_db()
    print("SQLite database initialized successfully.")