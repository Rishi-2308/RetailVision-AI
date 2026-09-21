"""
app.py  -  PHASE 11: Flask web application.

Run:   python app.py
Open:  http://127.0.0.1:5000
"""
import threading
import traceback
import uuid
from pathlib import Path

import cv2
from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, url_for)
from werkzeug.utils import secure_filename

import config
from database import database as db

app = Flask(__name__)
app.config["SECRET_KEY"] = "retailvision-dev-key-change-me"
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()

# in-memory progress of running jobs:  {video_id: {...}}
JOBS = {}
JOBS_LOCK = threading.Lock()
STAGES = ["Video uploaded", "Frame extraction", "Person detection", "Tracking",
          "Behavior recognition", "Engagement analysis", "Analytics generation"]


# ------------------------------------------------------------- helpers
def model_files_ready():
    """Which trained-model files exist (cheap check, does not load them)."""
    lstm = config.LSTM_MODEL_PATH.exists() and config.SCALER_PATH.exists()
    rf = config.RF_MODEL_PATH.exists()
    return (lstm if config.APP_MODEL == "lstm" else rf), {"lstm": lstm, "rf": rf}


@app.template_filter("mmss")
def mmss(seconds):
    try:
        s = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "-"
    return f"{s // 60:02d}:{s % 60:02d}"


@app.template_filter("num1")
def num1(v):
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "-"


def run_job(video_id):
    """Background thread that runs the whole pipeline for one video."""
    def progress(stage, text=""):
        with JOBS_LOCK:
            JOBS[video_id].update(stage=stage, text=text)

    try:
        import pipeline          # imported here: heavy libraries load only when needed
        db.update_video(video_id, status="processing", message="")
        msg = pipeline.process_video(video_id, progress)
        with JOBS_LOCK:
            JOBS[video_id].update(state="done", stage=7, text=msg)
    except Exception as e:      # noqa - shown to the user, full trace in terminal
        traceback.print_exc()
        db.update_video(video_id, status="error", message=str(e))
        with JOBS_LOCK:
            JOBS[video_id].update(state="error", error=str(e))


# -------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html", has_results=bool(db.list_videos()))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    ready, files = model_files_ready()
    if request.method == "POST":
        f = request.files.get("video")
        if f is None or f.filename == "":
            flash("Please choose a video file first.", "warning")
            return redirect(url_for("upload"))
        ext = Path(f.filename).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            flash(f"Unsupported file type '{ext}'. Use .mp4, .avi or .mov.", "danger")
            return redirect(url_for("upload"))
        stored = f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename) or 'video' + ext}"
        path = config.UPLOAD_DIR / stored
        f.save(path)
        cap = cv2.VideoCapture(str(path))
        valid = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        cap.release()
        if not valid:
            path.unlink(missing_ok=True)
            flash("That file could not be read as a video. Try another file.", "danger")
            return redirect(url_for("upload"))
        vid = db.create_video(stored, f.filename)
        return redirect(url_for("process", video_id=vid))
    return render_template("upload.html", ready=ready, files=files,
                           model=config.APP_MODEL, max_mb=config.MAX_UPLOAD_MB)


@app.route("/process/<int:video_id>")
def process(video_id):
    video = db.get_video(video_id)
    if video is None:
        abort(404)
    with JOBS_LOCK:
        job = JOBS.get(video_id)
        running = job is not None and job.get("state") == "running"
        if not running and (video["status"] in ("uploaded", "error") or request.args.get("rerun")):
            JOBS[video_id] = {"state": "running", "stage": 0, "text": "Starting", "error": ""}
            threading.Thread(target=run_job, args=(video_id,), daemon=True).start()
        elif video["status"] == "done" and job is None:
            return redirect(url_for("results", video_id=video_id))
    return render_template("processing.html", video=video, stages=STAGES)


@app.route("/status/<int:video_id>")
def status(video_id):
    with JOBS_LOCK:
        job = dict(JOBS.get(video_id, {}))
    if not job:
        video = db.get_video(video_id)
        if video is None:
            abort(404)
        job = {"state": "done" if video["status"] == "done" else video["status"],
               "stage": 7 if video["status"] == "done" else 0, "text": video.get("message") or "",
               "error": video.get("message") if video["status"] == "error" else ""}
    return jsonify(job)


@app.route("/results/<int:video_id>")
def results(video_id):
    video = db.get_video(video_id)
    if video is None:
        abort(404)
    if video["status"] != "done":
        return redirect(url_for("process", video_id=video_id))
    return render_template("results.html", video=video,
                           shoppers=db.get_video_results(video_id),
                           summary=db.video_summary(video_id))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", videos=[v for v in db.list_videos() if v["status"] == "done"])


@app.route("/api/dashboard")
def api_dashboard():
    vid = request.args.get("video_id", type=int)
    return jsonify(db.video_summary(vid))


@app.route("/demo")
def demo():
    """'View Demo' = open the most recent finished analysis."""
    done = [v for v in db.list_videos() if v["status"] == "done"]
    if not done:
        flash("No analysed video yet. Upload a video first to see the demo.", "info")
        return redirect(url_for("upload"))
    return redirect(url_for("results", video_id=done[0]["id"]))


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (limit {config.MAX_UPLOAD_MB} MB).", "danger")
    return redirect(url_for("upload"))


if __name__ == "__main__":
    # use_reloader=False: the background analysis thread must not be restarted
    app.run(debug=True, use_reloader=False, threaded=True)
