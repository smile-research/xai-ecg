#!/usr/bin/env python3
"""Scan explanations2 directory and generate paths CSV for radiomics extraction."""

import csv
import os
from pathlib import Path

ROOT = Path(__file__).parent
EXPLANATIONS_DIR = ROOT / "statystyki" / "explanations2"
OUTPUT = ROOT / "statystyki" / "paths4_remapped.csv"


def main():
    rows = []
    for model in sorted(EXPLANATIONS_DIR.iterdir()):
        if not model.is_dir():
            continue
        model_norm = model.name.replace(":v", "_v")
        for mode in sorted(model.iterdir()):
            if not mode.is_dir():
                continue
            for npz in sorted(mode.iterdir()):
                if npz.suffix == ".npz":
                    rows.append({
                        "full_path": str(npz.resolve()),
                        "model_name": model_norm,
                        "mode": mode.name,
                        "npz_name": npz.name,
                    })

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["full_path", "model_name", "mode", "npz_name"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Written {len(rows)} rows to {OUTPUT}")
    modes = sorted(set(r["mode"] for r in rows))
    models = sorted(set(r["model_name"] for r in rows))
    print(f"Modes: {modes}")
    print(f"Models: {models}")
    for mode in modes:
        count = sum(1 for r in rows if r["mode"] == mode)
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()
