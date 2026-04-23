import runpy
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent / "stock_ann_predictor"
sys.path.insert(0, str(BASE_DIR))

runpy.run_path(str(BASE_DIR / "main.py"), run_name="__main__")