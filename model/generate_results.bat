@echo off
echo 🚀 AIRPORT BOTTLENECK ANALYSIS - RESULTS.TXT GENERATOR
echo ============================================================
echo.
echo Generating results.txt from your data.json file...
echo.

cd /d "%~dp0"
python generate_results.py

echo.
echo ✅ Done! Check results.txt for detailed analysis.
echo.
pause
