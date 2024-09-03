nssm install ctspai "C:\ProgramData\miniconda3\envs\ai\Scripts\uvicorn.exe" "app.main:app --port 8000 --host 0.0.0.0 --reload"
@REM https://nssm.cc/commands
@REM nssm set ctspAIprediction
@REM nssm restart ctspAIprediction