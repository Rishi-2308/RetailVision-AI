@echo off
REM One-click setup for Windows (run from the RetailVisionAI folder).
REM Requires Python 3.11 installed with the "py" launcher.
echo === Checking Python 3.11 ===
py -3.11 --version || (echo Python 3.11 not found. Install it from python.org and tick "Add to PATH". & exit /b 1)
echo === Creating virtual environment .venv ===
py -3.11 -m venv .venv || exit /b 1
echo === Installing libraries (this takes several minutes) ===
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Installation failed - copy the error above and send it to me. & exit /b 1)
echo === Testing the environment ===
python scripts\check_environment.py
echo.
echo Done. In VS Code: Ctrl+Shift+P -^> "Python: Select Interpreter" -^> choose .venv
