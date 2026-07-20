"""Diagnosis 2c — render launch-window frames with ball track overlay
for wrong-end t5 clips, to pixel-verify that the detected 'serve
launch' is actually the RETURN of the true server's serve.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import cv2

from courtvision.config import load

cfg = load("t5")
out = Path(__file__).resolve().parent.parent / "outputs" / "diag"
out.mkdir(exist_ok=True)

CLIPS = {
    # clip: (launch_f, frames to render)
    "t5_point_06": (125, [95, 105, 115, 125, 132]),
    "t5_point_18": (125, [95, 105, 115, 125, 132]),
    "t5_point_44": (102, [70, 80, 90, 102, 110]),
    "t5_point_54": (9,   [0, 4, 9, 14, 20]),
}

for clip, (f_launch, want) in CLIPS.items():
    ball = list(csv.DictReader(open(cfg.ball_dir / f"ball_{clip}.csv")))
    track = {int(r["frame"]): (float(r["x_stab"]), float(r["y_stab"])) for r in ball}
    cap = cv2.VideoCapture(str(cfg.clip_path(clip)))
    for fw in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fw)
        ok, img = cap.read()
        if not ok:
            continue
        # overlay trailing tracked positions up to this frame (last 12)
        past = [f for f in sorted(track) if f <= fw][-12:]
        for i, f in enumerate(past):
            x, y = track[f]
            col = (0, 255, 255) if f == fw else (0, 165, 255)
            cv2.circle(img, (int(x), int(y)), 4 if f == fw else 2, col, -1)
        if fw in track:
            x, y = track[fw]
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 255), 2)
        tag = "LAUNCH" if fw == f_launch else ""
        cv2.putText(img, f"{clip} f{fw} {tag}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"{clip}_f{fw:03d}.png"), img)
    cap.release()
    print(f"{clip}: rendered {want} (launch f{f_launch})")
print(f"-> {out}")
