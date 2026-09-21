"""
pipeline.py  -  PHASE 7, 8, 9 combined.
Full analysis of ONE uploaded video:

  VIDEO -> OpenCV frames -> YOLO person detection -> ByteTrack IDs
        -> MediaPipe pose per shopper -> LSTM behaviour recognition
        -> dwell time / zones / engagement / journey / heatmap -> SQLite

Called by app.py, but it can also be run alone:
    python pipeline.py path\\to\\video.mp4
"""
import sys
from pathlib import Path

import numpy as np

import config
from analytics.dwell_time import zone_visits
from analytics.engagement import detect_lost_interest, engagement_for_visit
from analytics.heatmap import build_heatmap
from analytics.shopper_journey import build_journey
from analytics.zones import draw_zones
from behavior.feature_extraction import add_derived, fill_missing, sample_step_for
from behavior.predict import BehaviorPredictor, windows_to_events
from database import database as db

STAGES = ["Video uploaded", "Frame extraction", "Person detection", "Tracking",
          "Behavior recognition", "Engagement analysis", "Analytics generation"]


def _noop(stage, text=""):
    pass


def _open_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise IOError("This video cannot be opened by OpenCV (corrupt file or "
                      "unsupported codec). Try converting it to MP4 (H.264).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise IOError("Video has invalid size.")
    return cap, fps, n, w, h


# ---------------------------------------------------- stage: detect + track
def detect_and_track(video_path, progress):
    """Returns (tracks, meta, first_frame_small). Tracks hold per-frame data."""
    import cv2
    from detection.person_detector import load_yolo
    from detection.pose import PoseExtractor, motion_energy
    from detection.tracker import reset_tracker, track_frame

    cap, fps, n_frames, w, h = _open_video(video_path)
    step = sample_step_for(fps)
    max_frame = int(config.PROCESS_MAX_SECONDS * fps) if config.PROCESS_MAX_SECONDS else n_frames
    scale = config.FRAME_WIDTH / w
    small_size = (config.FRAME_WIDTH, int(h * scale))

    progress(1, "Reading frames with OpenCV")
    yolo = load_yolo()
    reset_tracker(yolo)
    pose = PoseExtractor(static_image_mode=True)

    tracks, prev_gray, first_small, i, sampled = {}, None, None, 0, 0
    try:
        while i < max_frame:
            if i % step != 0:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, small_size)
            if first_small is None:
                first_small = small.copy()
            gray = cv2.cvtColor(cv2.resize(small, (80, 60)), cv2.COLOR_BGR2GRAY)
            if sampled == 0:
                progress(2, "Detecting persons with YOLO")
            found = track_frame(yolo, small)
            if sampled == 0:
                progress(3, "Tracking shoppers with ByteTrack")
            for t in found:
                x1, y1, x2, y2 = t["bbox"]
                nb = (x1 / small_size[0], y1 / small_size[1], x2 / small_size[0], y2 / small_size[1])
                row = np.full(20, np.nan, dtype=np.float32)
                p = pose.extract(small, t["bbox"])
                if p is not None:
                    row[:18], row[18] = p, 1.0
                else:
                    row[18] = 0.0
                row[19] = motion_energy(prev_gray, gray, nb)
                d = tracks.setdefault(t["track_id"], {"t": [], "frame": [], "bbox": [],
                                                      "conf": [], "raw": []})
                d["t"].append(i / fps)
                d["frame"].append(i)
                d["bbox"].append(nb)
                d["conf"].append(t["confidence"])
                d["raw"].append(row)
            prev_gray = gray
            sampled += 1
            i += 1
            if sampled % 25 == 0:
                progress(3, f"Tracking shoppers ({100 * i // max(max_frame, 1)}%)")
    finally:
        cap.release()
        pose.close()
    if first_small is None:
        raise IOError("No frames could be read from the video.")
    meta = {"fps": fps, "step": step, "width": w, "height": h,
            "duration": min(i, n_frames) / fps, "n_sampled": sampled,
            "small_size": small_size}
    return tracks, meta, first_small


# ------------------------------------------------------- stage: analytics
def analyse_tracks(tracks, meta, predictor, progress):
    """Behaviour + zones + engagement for every valid track."""
    dt = meta["step"] / meta["fps"]
    valid = []
    for tid, d in tracks.items():
        dur = d["t"][-1] - d["t"][0] + dt
        if dur >= config.MIN_TRACK_SECONDS and len(d["t"]) >= 8:
            valid.append((tid, d))
    valid.sort(key=lambda kv: kv[1]["t"][0])

    progress(4, "Recognising behaviours with the trained model")
    shoppers = []
    for n, (tid, d) in enumerate(valid, 1):
        times = np.array(d["t"])
        bbox = np.array(d["bbox"])
        raw = np.vstack(d["raw"])
        channels = add_derived(fill_missing(raw), dt)
        pose_rate = float(np.mean(raw[:, 18]))

        windows = predictor.predict_track(channels, times)
        events = windows_to_events(windows)
        xs, ys = (bbox[:, 0] + bbox[:, 2]) / 2, (bbox[:, 1] + bbox[:, 3]) / 2

        progress(5, f"Engagement analysis (shopper {n}/{len(valid)})")
        visits = zone_visits(times, xs, ys)
        if not visits:
            continue
        for v in visits:
            e = engagement_for_visit(v, events, times, xs, ys)
            v["engagement_score"], v["eng"] = e["score"], e
            v["interaction_seconds"] = e["interaction_seconds"]

        # each event gets the zone where it overlaps most
        for ev in events:
            best, best_o = None, 0.0
            for v in visits:
                o = min(ev["end"], v["exit"]) - max(ev["start"], v["entry"])
                if o > best_o:
                    best, best_o = v["zone"], o
            ev["zone"] = best

        best_visit = max(visits, key=lambda v: v["engagement_score"])
        dur_by_beh = {}
        for ev in events:
            dur_by_beh[ev["behavior"]] = dur_by_beh.get(ev["behavior"], 0) + (ev["end"] - ev["start"])
        dominant = max(dur_by_beh, key=dur_by_beh.get) if dur_by_beh else "No clear action"
        lost = [x for x in (detect_lost_interest(v, v["eng"], meta["duration"]) for v in visits) if x]
        shoppers.append({
            "track_id": tid, "label": f"Customer #{len(shoppers) + 1:02d}",
            "start": float(times[0]), "end": float(times[-1] + dt),
            "behaviors": events, "visits": visits, "pose_rate": pose_rate,
            "xs": xs, "ys": ys, "times": times,
            "analytics": {
                "engagement_score": best_visit["engagement_score"],
                "level": best_visit["eng"]["level"],
                "total_dwell": float(sum(v["dwell"] for v in visits)),
                "dominant_behavior": dominant,
                "reasons": best_visit["eng"]["reasons"],
                "components": best_visit["eng"]["components"],
                "journey": build_journey(visits, events),
                "lost_interest": lost,
            },
        })
    return shoppers


# ------------------------------------------------------ annotated output
def write_annotated_video(video_path, out_path, shoppers, tracks, meta):
    """Second pass: draw boxes, shopper labels and current behaviour."""
    import cv2
    by_track = {s["track_id"]: s for s in shoppers}
    per_frame = {}
    for tid, s in by_track.items():
        d = tracks[tid]
        for fr, bb in zip(d["frame"], d["bbox"]):
            per_frame.setdefault(fr, []).append((s, bb))

    cap, fps, _, _, _ = _open_video(video_path)
    size = meta["small_size"]
    out_fps = max(1.0, fps / meta["step"])
    writer = None
    for codec in ("avc1", "mp4v"):        # avc1 = browser friendly, if available
        wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), out_fps, size)
        if wr.isOpened():
            writer = wr
            break
    if writer is None:
        cap.release()
        return False
    i = 0
    try:
        while True:
            if i % meta["step"] != 0:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            frame = draw_zones(cv2.resize(frame, size))
            t = i / fps
            for s, bb in per_frame.get(i, []):
                x1, y1, x2, y2 = int(bb[0] * size[0]), int(bb[1] * size[1]), int(bb[2] * size[0]), int(bb[3] * size[1])
                beh = next((e["behavior"] for e in s["behaviors"] if e["start"] <= t <= e["end"]), "")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{s['label']}" + (f" | {beh}" if beh else "")
                cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + 8 * len(text), y1), (0, 128, 0), -1)
                cv2.putText(frame, text, (x1 + 2, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1)
            writer.write(frame)
            i += 1
            if config.PROCESS_MAX_SECONDS and t > config.PROCESS_MAX_SECONDS:
                break
    finally:
        cap.release()
        writer.release()
    return True


# ------------------------------------------------------------------- main
def process_video(video_id, progress=_noop):
    """Run everything for one video record. Raises on fatal errors."""
    import cv2
    video = db.get_video(video_id)
    if video is None:
        raise ValueError(f"Video {video_id} not found in database")
    path = config.UPLOAD_DIR / video["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Uploaded file missing: {path}")

    progress(0, "Loading trained behaviour model")
    predictor = BehaviorPredictor()          # fails early with a clear message

    tracks, meta, first_frame = detect_and_track(path, progress)
    shoppers = analyse_tracks(tracks, meta, predictor, progress)

    progress(6, "Generating analytics, heatmap and annotated video")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heat_name, video_name, prev_name = (f"heatmap_{video_id}.png", f"annotated_{video_id}.mp4",
                                        f"preview_{video_id}.jpg")
    xs, ys, ws = [], [], []
    for s in shoppers:
        for x, y, t in zip(s["xs"], s["ys"], s["times"]):
            in_event = any(e["start"] <= t <= e["end"] for e in s["behaviors"])
            xs.append(x); ys.append(y); ws.append(1.0 + (2.0 if in_event else 0.0))
    has_heat = build_heatmap(xs, ys, ws, draw_zones(first_frame), config.OUTPUT_DIR / heat_name)
    cv2.imwrite(str(config.OUTPUT_DIR / prev_name), draw_zones(first_frame))
    has_video = False
    if shoppers:
        has_video = write_annotated_video(path, config.OUTPUT_DIR / video_name, shoppers, tracks, meta)

    db.save_analysis(video_id, shoppers)
    low_pose = [s["label"] for s in shoppers if s["pose_rate"] < 0.3]
    msg = f"{len(shoppers)} shopper(s) analysed with the {predictor.model_name.upper()} model."
    if not shoppers:
        msg = ("No shopper was tracked long enough. Check that people are clearly visible "
               "(YOLO is trained on normal camera views, not always on overhead views).")
    elif low_pose:
        msg += (" Pose was found in <30% of frames for " + ", ".join(low_pose) +
                " - behaviour results for them are less reliable.")
    db.update_video(video_id, status="done", message=msg, duration=meta["duration"],
                    fps=meta["fps"], width=meta["width"], height=meta["height"],
                    model_used=predictor.model_name,
                    annotated_video=video_name if has_video else None,
                    heatmap_image=heat_name if has_heat else None, preview_image=prev_name)
    progress(7, "Done")
    return msg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python pipeline.py path\\to\\video.mp4")
    import shutil
    src = Path(sys.argv[1])
    if not src.exists():
        sys.exit(f"File not found: {src}")
    db.init_db()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, config.UPLOAD_DIR / src.name)
    vid = db.create_video(src.name, src.name)
    print(process_video(vid, lambda i, t="": print(f"[{i}/6] {STAGES[min(i, 6)]}: {t}")))
    for s in db.get_video_results(vid):
        a = s["analytics"]
        print(f"{s['label']}: engagement {a['engagement_score']} ({a['engagement_level']}), "
              f"dwell {a['total_dwell_time']:.1f}s, dominant: {a['dominant_behavior']}")
