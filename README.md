# RetailVision AI - Intelligent Shopper Behavior & Attention Analytics

Computer-vision retail analytics with **Python 3.11 + Flask**:

```
VIDEO -> OpenCV -> YOLO person detection -> ByteTrack tracking
      -> MediaPipe pose / movement features -> LSTM behavior recognition
      -> engagement, dwell time, zones, journey, heatmap -> SQLite -> Flask dashboard
```

The behaviour model is **trained by you** on the MERL Shopping Dataset
(5 classes: Reach to Shelf, Retract from Shelf, Hand in Shelf, Inspect Product,
Inspect Shelf). YOLO is used only to find people. Random Forest is the baseline,
LSTM is the final model. **No accuracy number is written anywhere in this project -
every number comes from your own training run.**

---

## 0. Read this first (honest notes)

* The code was written and **logic-tested with synthetic data** (dataset parsing, splits,
  feature building, Random Forest, analytics, SQLite, all Flask routes). YOLO, MediaPipe and
  TensorFlow/LSTM could not be run where this project was built, so **your first run of
  Phase 1 (`check_environment.py`) is the real test**. If anything fails, copy the full
  error message - it will be fixed without touching the working parts.
* Nobody can promise an accuracy before training. The result depends on the real MERL data.
  MERL is filmed from an **overhead camera**; MediaPipe and YOLO are trained mostly on
  normal (side/front) views, so pose may be found in fewer frames. `feature_extraction.py`
  prints the real pose-detection rate. Motion features are used as well, but report this
  honestly as a limitation.
* A model trained on MERL will work best on videos that look like MERL. Videos from other
  camera angles are a "domain shift" - results can be worse.
* Engagement score = **project-defined behavioural metric** (not emotion, intention or true
  attention). Bands Low/Moderate/High/Very High are project-defined.
* Privacy: anonymous IDs only ("Customer #01"), no face recognition, no gender/emotion
  inference. Use only consented / staged videos.
* The optional *supplementary dataset* is **not implemented** in code (it was optional in the brief).

---

## 1. Setup on Windows (Phase 1)

1. Install **Python 3.11 (64-bit)** from python.org (tick "Add Python to PATH").
2. Unzip this project, open the folder in **VS Code** (File -> Open Folder).
3. Open the VS Code terminal (Ctrl+`) and run **one** of these:

**Automatic:**
```
setup_windows.bat
```
**Manual:**
```
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\check_environment.py
```
(If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use *Command Prompt*.)

4. VS Code: `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose `.venv`.

**Expected:** every line of `check_environment.py` shows `[OK]` and ends with
`ALL CHECKS PASSED - Phase 1 complete.` (The first YOLO run downloads `yolov8n.pt`, needs internet.)

Quick logic self-test (no AI libraries needed): `python -m unittest discover -s tests -v` -> `OK`.

---

## 2. Getting the dataset (Phase 2)

1. Search "MERL Shopping Dataset" on the Mitsubishi Electric Research Laboratories (merl.com)
   website and follow their license/download instructions (the download needs accepting a license).
   Reference paper: Singh et al., *A Multi-Stream Bi-Directional Recurrent Neural Network for
   Fine-Grained Action Detection*, CVPR 2016.
2. Unzip it **inside** `dataset\raw\` (any sub-folder layout is fine).
3. Inspect what you really got:
```
python behavior\inspect_dataset.py
```
It prints the folder tree, video properties and the content of a label file.
**If it reports an error about the label file format, paste the whole output** - the label
reader (`read_mat_labels` in `behavior/extract_clips.py`) expects a `.mat` file holding a cell
array with one `[start, end]` matrix per class; a CSV with `class_id,start_frame,end_frame`
also works.

4. Build annotations and the train/validation/test split:
```
python behavior\extract_clips.py --preview
```
Creates `dataset\processed\annotations.csv`, `segments.csv`, `dataset\train\train.csv`,
`dataset\validation\validation.csv`, `dataset\test\test.csv` and `reports\sample_segments.jpg`
(sample frames of every class). **Split rules:** if the dataset folders are named
Train/Validation/Test the official split is kept; otherwise videos are split as whole videos
(70/15/15, fixed seed). A video is never in two splits (checked automatically).

## 3. Preprocessing + feature extraction (Phase 3-4)

```
python behavior\feature_extraction.py          (add --limit 3 for a quick test)
```
* Frames are sampled at about **10 fps** (`TARGET_FPS`): actions last 1-3 s, so 10 fps keeps
  10-30 frames per action while MediaPipe runs ~3x faster than at 30 fps.
* Each action is resampled to **16 time steps** (`SEQ_LEN`, about 1.6 s) for the LSTM.
* 26 measured features per frame: 9 pose landmarks (x, y), pose-found flag, motion energy,
  body centre, wrist-to-shoulder distances, wrist speed, body speed.
* Results are cached in `dataset\processed\pose\` (re-runs are fast).
* Writes `sequences.npz`, `features.csv`, `models\scaler.pkl` (scaler fitted on TRAIN only) and
  prints validation (NaN count, pose-detection rate). Takes a while on the full dataset.

## 4. Train the models (Phase 5-6)

```
python behavior\train_random_forest.py
python behavior\train_lstm.py
python behavior\compare_models.py
```
Outputs in `reports\`: `random_forest_metrics.json`, `lstm_metrics.json`,
`confusion_matrix_rf.png`, `confusion_matrix_lstm.png`, `lstm_loss_curve.png`,
`lstm_accuracy_curve.png`, `model_comparison.md` (Random Forest vs LSTM table with
Accuracy / Precision / Recall / F1 measured on the untouched test set).
Models are saved to `models\random_forest.pkl`, `models\behavior_lstm.h5`.

Or run phases 2-6 together: `run_all_training.bat`.

## 5. Run the web application (Phase 7-11)

```
python app.py
```
Open http://127.0.0.1:5000 -> **Upload Video** -> wait on the processing page -> results.
Dashboard: http://127.0.0.1:5000/dashboard. (Bootstrap and Chart.js load from a CDN, so the
browser needs internet.)

Command-line alternative: `python pipeline.py path\to\video.mp4`.

Settings (`config.py`): `APP_MODEL` ("lstm" or "rf"), `PROCESS_MAX_SECONDS` (e.g. 30 for a
quick test), `MIN_CONFIDENCE`, `ZONES`, engagement weights.

**Zones must be calibrated:** by default the frame is a 2x2 grid (Zone A-D) with the shelf
anchor at each zone centre. Edit `ZONES` in `config.py` (normalised 0-1 rectangles and
anchor points) so they match the real shelves of your video - this directly affects
proximity, zones, dwell time and heatmap.

## 6. Engagement score (project-defined)

```
score = 100 x (0.40 duration + 0.30 proximity + 0.20 interaction + 0.10 movement)
```
duration = interaction seconds / 10; proximity = closeness to the zone's shelf anchor;
interaction = weight of the recognised behaviour (Inspect Product 1.0, Hand in Shelf 0.8,
Inspect Shelf 0.7, Reach 0.6, Retract 0.4); movement = body steadiness. Bands:
0-30 Low, 31-60 Moderate, 61-80 High, 81-100 Very High. Only reasons supported by these
numbers are displayed. A shopper's score is the highest score among their zone visits.
"Potential Lost Interest" = a brief interaction (<5 s) followed by leaving the zone.

## 7. Database (SQLite, file `retailvision.db`)

`videos`, `shoppers` (id, video_id, start/end), `behaviors` (behavior, confidence, times, zone),
`zones` (zone, entry, exit, dwell, engagement), `analytics` (engagement score, total dwell,
dominant behavior, reasons, journey, lost interest).

## 8. Project layout

```
app.py  pipeline.py  config.py  requirements.txt
detection/   person_detector.py  tracker.py  pose.py
behavior/    inspect_dataset.py extract_clips.py feature_extraction.py
             train_random_forest.py train_lstm.py compare_models.py predict.py evaluation.py
analytics/   engagement.py dwell_time.py zones.py heatmap.py shopper_journey.py
database/    database.py
notebooks/   01..05 (read the real outputs of the scripts)
templates/ static/ scripts/check_environment.py tests/ reports/ models/ dataset/
```

## 9. Known limitations (write these in your report)

* Overhead MERL view vs. YOLO/MediaPipe training views (see section 0).
* Training uses annotated segments; in the app windows are cut with a sliding window, and the
  model has no "background / no action" class, so a confidence threshold (`MIN_CONFIDENCE`)
  is used to show "no clear action".
* Pose for uploaded videos is computed on the YOLO person crop; for MERL training on the full
  frame (MERL has essentially one shopper per video).
* ByteTrack can swap/lose IDs in crowds; one shopper may become two IDs.
* Times are video times (mm:ss), not clock times.
* Annotated video uses the codec OpenCV offers (often not playable in a browser): download it
  and open in VLC/Windows Media Player. Messages such as "h264 ... Failed to initialize VideoWriter"
  in the terminal are harmless (it falls back to another codec).

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| `pip install` fails | Use Python 3.11 64-bit; run `python -m pip install --upgrade pip`; paste the full error |
| MediaPipe: `no attribute 'solutions'` | `pip install mediapipe==0.10.14` |
| NumPy 2 errors | `pip install "numpy==1.26.4"` |
| `.venv\Scripts\activate` blocked | Use Command Prompt or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Upload page warns "trained model not found" | Finish sections 2-4 first |
| "No shopper was tracked" | People must be clearly visible; try a side/front-view video; lower `YOLO_CONF` |
| Slow processing | Set `PROCESS_MAX_SECONDS = 30` in `config.py` |
