"""Render contact frames for t4's wrong-letter clips: ball (raw coords)
plus both hygiene-passed player boxes (stab -> raw via shifts CSV).

Usage: uv run python experiments/t4_letter_frames.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import cv2
import numpy as np

from courtvision.config import load as load_cfg
from courtvision import boxes

cfg = load_cfg("t4")
Hm = np.load(cfg.homography)
offsets = cfg.load_offsets()
OUT = Path(__file__).resolve().parent.parent / "outputs" / "diag"
OUT.mkdir(exist_ok=True)

WANT = {
    "t4_point_30": [56, 95, 109, 129, 154, 201, 244],
    "t4_point_42": [0, 25, 76, 116, 164, 215, 242, 278],
    "t4_point_46": [104, 116, 129, 141, 195, 225, 269, 327],
    "t4_point_47": [84],
}

for clip, frames_want in WANT.items():
    players = boxes.load(cfg.players_dir / f"players_{clip}.csv", Hm,
                         offsets.get(clip, (0.0, 0.0)))
    ball = {int(r["frame"]): r
            for r in csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv"))}
    shifts = {int(r["frame"]): (float(r["dx"]), float(r["dy"]))
              for r in csv.DictReader(
                  open(cfg.out_dir / "plates" / f"shifts_{clip}.csv"))}
    cap = cv2.VideoCapture(str(cfg.clip_path(clip)))
    for fw in frames_want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fw)
        ok, img = cap.read()
        if not ok:
            print(f"{clip} f{fw}: read fail")
            continue
        dx, dy = shifts.get(fw, (0.0, 0.0))
        for side, col in (("near", (0, 255, 0)), ("far", (0, 0, 255))):
            row = players.get(fw, {}).get(side)
            if row is None:
                continue
            cx = float(row["cx"]) * 1280 + dx
            cy = float(row["cy"]) * 720 + dy
            w = float(row["w"]) * 1280
            h = float(row["h"]) * 720
            cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)),
                          (int(cx + w / 2), int(cy + h / 2)), col, 2)
            cv2.line(img, (int(cx), int(cy - h / 2 - 8)),
                     (int(cx), int(cy + h / 2 + 8)), col, 1)
        b = ball.get(fw)
        if b is not None:
            bx = float(b["cx_raw"]) * 1280
            by = float(b["cy_raw"]) * 720
            cv2.circle(img, (int(bx), int(by)), 7, (0, 255, 255), 2)
        cv2.putText(img, f"{clip} f{fw}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{clip}_f{fw:04d}.jpg"), img)
        print(f"wrote {clip}_f{fw:04d}.jpg ball={'y' if b else 'n'}")
    cap.release()
