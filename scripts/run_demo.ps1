$ErrorActionPreference = "Stop"

Write-Host "Starting FastAPI on 8000..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn deployments.fastapi_app.main:app --reload --port 8000"

Start-Sleep -Seconds 2

Write-Host "Starting Streamlit on 8501..."
streamlit run deployments\streamlit_app\Home.py