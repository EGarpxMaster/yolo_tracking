@echo off
echo Installing Python Traffic Counter with YOLOv11x and BoTSORT...
echo.

echo Checking Python version...
python --version
echo.

echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo Basic usage:
echo python main.py --input input/highway.mp4 --output output/highway.mp4
echo.
echo With class selection:
echo python main.py --input input/video.mp4 --output output/result.mp4 --classes people_and_vehicles
echo.
echo Note: Class names are now always shown (e.g., "5 car", "12 person")
echo Use --show-labels for detailed format (e.g., "ID:5 car", "ID:12 person")
echo.
echo Available classes: vehicles, people, people_and_vehicles, transportation, traffic, all
echo.
pause