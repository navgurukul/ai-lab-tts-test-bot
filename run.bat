@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python tts_test_bot.py
if errorlevel 1 pause
