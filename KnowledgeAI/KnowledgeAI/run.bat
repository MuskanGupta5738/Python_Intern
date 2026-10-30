@echo off
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Python nahi mila. Python install karo aur "Add to PATH" tick karo.
        pause
        exit /b 1
    )
)

if not exist venv\.installed (
    echo Installing packages, thoda time lagega...
    venv\Scripts\python -m pip install --upgrade pip
    venv\Scripts\python -m pip install -r backend\requirements.txt -r frontend\requirements.txt
    if errorlevel 1 (
        echo Install fail hua. Upar ka error dekho.
        pause
        exit /b 1
    )
    echo done > venv\.installed
)

echo Starting backend...
start "KnowledgeAI Backend" cmd /k "cd /d %~dp0backend && ..\venv\Scripts\python -m uvicorn main:app --port 8000"

timeout /t 5 /nobreak >nul

echo Starting frontend...
start "KnowledgeAI Frontend" cmd /k "cd /d %~dp0frontend && ..\venv\Scripts\python -m streamlit run app.py"

echo.
echo Browser mein http://localhost:8501 khulega. Band karne ke liye dono naye windows band kar do.
