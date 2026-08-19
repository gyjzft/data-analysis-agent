@echo off
chcp 65001 >nul
cd /d "C:\Users\gyj\Desktop\finding_job\data_analysis_agent"
call "C:\Users\gyj\Desktop\finding_job\ai_venv\Scripts\activate.bat"
set PYTHONPATH=C:\Users\gyj\Desktop\finding_job\data_analysis_agent
python -m src.ui.app
pause
