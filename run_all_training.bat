@echo off
REM Runs PHASE 2-6 in order. Stops at the first error.
call .venv\Scripts\activate.bat
python behavior\inspect_dataset.py || exit /b 1
python behavior\extract_clips.py --preview || exit /b 1
python behavior\feature_extraction.py || exit /b 1
python behavior\train_random_forest.py || exit /b 1
python behavior\train_lstm.py || exit /b 1
python behavior\compare_models.py || exit /b 1
echo.
echo Training finished. Start the app with:  python app.py
