"""
scripts/check_environment.py  -  PHASE 1 test.
Run:  python scripts/check_environment.py
Every line must say [OK]. If a line says [FAIL], copy the message and fix it
before going on.
"""
import sys
import importlib

failures = []


def ok(msg):
    print(f"[OK]   {msg}")


def fail(name, err):
    print(f"[FAIL] {name}: {err}")
    failures.append(name)


# 1. Python version -------------------------------------------------------
v = sys.version_info
if (v.major, v.minor) == (3, 11):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
elif v.major == 3 and v.minor in (10, 12):
    print(f"[WARN] Python {v.major}.{v.minor} works for most libraries, but "
          "Python 3.11 is the tested version.")
else:
    fail("Python", f"found {v.major}.{v.minor}, need 3.11")

# 2. Simple imports -------------------------------------------------------
for mod in ["numpy", "pandas", "scipy", "sklearn", "joblib", "matplotlib", "flask"]:
    try:
        m = importlib.import_module(mod)
        ok(f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa
        fail(mod, e)

# 3. OpenCV ---------------------------------------------------------------
try:
    import cv2
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (60, 60), (0, 255, 0), 2)
    assert img.sum() > 0
    ok(f"OpenCV {cv2.__version__} can create and draw on an image")
except Exception as e:  # noqa
    fail("OpenCV", e)

# 4. MediaPipe Pose -------------------------------------------------------
try:
    import mediapipe as mp
    import numpy as np
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    res = pose.process(blank)          # blank image -> no person, that is fine
    pose.close()
    ok(f"MediaPipe {mp.__version__} Pose runs (landmarks on blank image: "
       f"{'yes' if res.pose_landmarks else 'none, as expected'})")
except Exception as e:  # noqa
    fail("MediaPipe", e)

# 5. YOLO (Ultralytics) ---------------------------------------------------
try:
    from ultralytics import YOLO
    import numpy as np
    print("       (first run downloads yolov8n.pt ~6 MB, needs internet)")
    model = YOLO("yolov8n.pt")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = model.predict(blank, verbose=False)
    ok(f"YOLO loaded and ran on a test image ({len(out[0].boxes)} boxes, "
       "0 expected)")
    # ByteTrack needs the 'lap' package; Ultralytics installs it on first use.
    model.track(blank, persist=True, tracker="bytetrack.yaml", verbose=False)
    ok("ByteTrack tracker runs (if it just installed 'lap', that is normal)")
except Exception as e:  # noqa
    fail("YOLO", e)

# 6. TensorFlow / Keras ---------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    m = keras.Sequential([keras.layers.Input((16, 4)), keras.layers.LSTM(8),
                          keras.layers.Dense(5, activation="softmax")])
    ok(f"TensorFlow {tf.__version__} builds an LSTM model")
except Exception as e:  # noqa
    fail("TensorFlow", e)

print()
if failures:
    print("FAILED:", ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED - Phase 1 complete.")
