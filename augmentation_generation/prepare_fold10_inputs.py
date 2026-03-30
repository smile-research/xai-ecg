"""
Reads fold10_labels.csv and copies corresponding .dat/.hea waveform files
from physionet records500 into a flat directory: fold10_images_for_xai/
Safe to re-run — skips already copied files.
"""
import shutil
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
RECORDS_DIR = BASE / "physionet.org/files/ptb-xl/1.0.3"
OUTPUT_DIR = BASE / "fold10_images_for_xai"
CSV_PATH = BASE.parent / "explanation_generation/configs/fold10_labels.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

copied, missing, skipped = 0, 0, 0

for csv_path in df["path"].unique():
    # records500/00000/00009_hr-0.png -> records500/00000/00009_hr
    parts = Path(csv_path)
    stem = parts.stem.rsplit("-", 1)[0]  # 00009_hr-0 -> 00009_hr
    record_dir = RECORDS_DIR / parts.parent  # records500/00000

    for ext in (".dat", ".hea"):
        src = record_dir / (stem + ext)
        dst = OUTPUT_DIR / (stem + ext)
        if dst.exists():
            skipped += 1
            continue
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

print(f"Copied: {copied}, Skipped (already exist): {skipped}, Missing (not downloaded yet): {missing}")
print(f"Output: {OUTPUT_DIR}")
