"""M2: final artifacts — the bounce map and the event-annotated demo video.

Bounce map: v4 events on the top-down court. Near-side bounces drawn
solid (positions trustworthy); far-side bounces drawn hollow with a note
(positions ±meters, see LOG). Hits drawn as X at the striker's end.

Demo video: the M1 side-by-side, plus event flashes — HIT/BOUNCE label
pops on the broadcast panel for ~12 frames after each event, and bounce
marks accumulate on the court panel.

Usage:
    uv run experiments/m2_render_artifacts.py clips/rally.mp4 \
        outputs/m0/trajectory_ballfix.csv outputs/m1/H_img_to_court.npy \
        outputs/m2/events_v4.csv
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m2"

W_COURT = 10.97
L_COURT = 23.77
SINGLES_INSET = 1.372
NET_Y = L_COURT / 2
SVC_FAR_Y = NET_Y - 6.40
SVC_NEAR_Y = NET_Y + 6.40
CENTER_X = W_COURT / 2

PPM = 22
MARGIN_M = 2.5
PANEL_H = 720
LAST_FRAME = 298
FLASH = 12


def draw_court_mpl(ax):
    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="white", lw=1.5, zorder=1)
    ax.add_patch(plt.Rectangle((-3, -4), W_COURT + 6, L_COURT + 8,
                               color="#3b5b92", zorder=0))
    line(0, 0, W_COURT, 0); line(0, L_COURT, W_COURT, L_COURT)
    line(0, 0, 0, L_COURT); line(W_COURT, 0, W_COURT, L_COURT)
    line(SINGLES_INSET, 0, SINGLES_INSET, L_COURT)
    line(W_COURT - SINGLES_INSET, 0, W_COURT - SINGLES_INSET, L_COURT)
    line(SINGLES_INSET, SVC_FAR_Y, W_COURT - SINGLES_INSET, SVC_FAR_Y)
    line(SINGLES_INSET, SVC_NEAR_Y, W_COURT - SINGLES_INSET, SVC_NEAR_Y)
    line(CENTER_X, SVC_FAR_Y, CENTER_X, SVC_NEAR_Y)
    ax.plot([0, W_COURT], [NET_Y, NET_Y], color="#dddddd", lw=2.5,
            linestyle=(0, (4, 2)), zorder=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("traj_csv")
    parser.add_argument("h_npy")
    parser.add_argument("events_csv")
    args = parser.parse_args()

    H = np.load(args.h_npy)
    boxes = {}
    with open(args.traj_csv) as f:
        for row in csv.DictReader(f):
            boxes[int(row["frame"])] = (float(row["cx"]), float(row["cy"]),
                                        float(row["w"]), float(row["h"]))
    events = []
    with open(args.events_csv) as f:
        for row in csv.DictReader(f):
            events.append({"frame": int(row["frame"]), "kind": row["kind"],
                           "signal": row["signal"],
                           "court_y": float(row["court_y_m"])})

    # court x for each event from the mapped track (median around the frame)
    frames_sorted = sorted(boxes)
    def court_pos(fi, win=4):
        pts = []
        for f in range(fi - win, fi + win + 1):
            if f in boxes:
                cx, cy, _, _ = boxes[f]
                p = cv2.perspectiveTransform(
                    np.float32([[cx * 1280, cy * 720]]).reshape(-1, 1, 2), H).reshape(2)
                pts.append(p)
        return np.median(np.array(pts), axis=0)

    for e in events:
        e["pos"] = court_pos(e["frame"])

    # ---- bounce map ----
    fig, ax = plt.subplots(figsize=(6, 10))
    draw_court_mpl(ax)
    for e in events:
        x, y = e["pos"]
        near = y > NET_Y
        if e["kind"] == "bounce":
            if near:
                ax.scatter(x, y, s=180, color="#ffd23f", edgecolors="black",
                           linewidths=1.5, zorder=3)
            else:
                ax.scatter(x, np.clip(y, -2, L_COURT + 2), s=180,
                           facecolors="none", edgecolors="#ffd23f",
                           linewidths=2.5, zorder=3)
        else:
            ax.scatter(x, np.clip(y, -3.5, L_COURT + 3.5), s=140, marker="X",
                       color="#ff5555", edgecolors="black", linewidths=0.8,
                       zorder=3)
        ax.annotate(f"f{e['frame']}", (x, np.clip(y, -3.5, L_COURT + 3.5)),
                    textcoords="offset points", xytext=(12, 0), fontsize=8,
                    color="white", zorder=4)
    ax.scatter([], [], s=180, color="#ffd23f", edgecolors="black", label="bounce (position solid)")
    ax.scatter([], [], s=180, facecolors="none", edgecolors="#ffd23f", label="bounce (far side: position ±m)")
    ax.scatter([], [], s=140, marker="X", color="#ff5555", label="hit")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-3, W_COURT + 3)
    ax.set_ylim(L_COURT + 4, -4)
    ax.set_aspect("equal")
    ax.set_title("M2 — every hit and bounce of the rally, placed on the court\n"
                 "(7 hits, 6 bounces, all frame-verified)")
    ax.set_xlabel("court x (m)")
    ax.set_ylabel("court y (m)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "event_map.png", dpi=150)
    print("-> outputs/m2/event_map.png")

    # ---- demo video: side-by-side with event flashes ----
    def court_to_panel(x, y):
        return int((x + MARGIN_M) * PPM), int((y + MARGIN_M) * PPM)

    w_panel_full = int((W_COURT + 2 * MARGIN_M) * PPM)
    h_panel_full = int((L_COURT + 2 * MARGIN_M) * PPM)
    court_img = np.full((h_panel_full, w_panel_full, 3), (146, 91, 59), np.uint8)

    def cline(x1, y1, x2, y2, color=(255, 255, 255), t=2):
        cv2.line(court_img, court_to_panel(x1, y1), court_to_panel(x2, y2), color, t)

    cline(0, 0, W_COURT, 0); cline(0, L_COURT, W_COURT, L_COURT)
    cline(0, 0, 0, L_COURT); cline(W_COURT, 0, W_COURT, L_COURT)
    cline(SINGLES_INSET, 0, SINGLES_INSET, L_COURT)
    cline(W_COURT - SINGLES_INSET, 0, W_COURT - SINGLES_INSET, L_COURT)
    cline(SINGLES_INSET, SVC_FAR_Y, W_COURT - SINGLES_INSET, SVC_FAR_Y)
    cline(SINGLES_INSET, SVC_NEAR_Y, W_COURT - SINGLES_INSET, SVC_NEAR_Y)
    cline(CENTER_X, SVC_FAR_Y, CENTER_X, SVC_NEAR_Y)
    cline(0, NET_Y, W_COURT, NET_Y, color=(200, 200, 200), t=3)
    scale = PANEL_H / h_panel_full
    base_panel = cv2.resize(court_img, (int(w_panel_full * scale), PANEL_H))
    panel_w = base_panel.shape[1]

    def to_panel_px(pt):
        return (int((pt[0] + MARGIN_M) * PPM * scale),
                int((np.clip(pt[1], -2, L_COURT + 2) + MARGIN_M) * PPM * scale))

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(OUT_DIR / "events_demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W + panel_w, PANEL_H))

    trail, marks = [], []
    i = 0
    while i <= LAST_FRAME:
        ok, frame = cap.read()
        if not ok:
            break
        if i in boxes:
            cx, cy, w, h = boxes[i]
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * Hh)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * Hh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            pt = cv2.perspectiveTransform(
                np.float32([[cx * W, cy * Hh]]).reshape(-1, 1, 2), H).reshape(2)
            trail.append(to_panel_px(pt))

        for e in events:
            d = i - e["frame"]
            if d == 0:
                marks.append(e)
            if 0 <= d < FLASH:
                label = e["kind"].upper()
                color = (85, 85, 255) if e["kind"] == "hit" else (255, 255, 255)
                cv2.putText(frame, label, (W // 2 - 120, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0, 0, 0), 10)
                cv2.putText(frame, label, (W // 2 - 120, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.6, color, 5)

        panel = base_panel.copy()
        for p in trail[:-1]:
            cv2.circle(panel, p, 2, (80, 200, 255), -1)
        if trail:
            cv2.circle(panel, trail[-1], 6, (0, 255, 255), 2)
        for e in marks:
            p = to_panel_px(e["pos"])
            if e["kind"] == "bounce":
                cv2.circle(panel, p, 9, (63, 210, 255), -1)
                cv2.circle(panel, p, 9, (0, 0, 0), 2)
            else:
                cv2.drawMarker(panel, p, (85, 85, 255), cv2.MARKER_TILTED_CROSS, 18, 4)

        canvas = np.zeros((PANEL_H, W + panel_w, 3), np.uint8)
        canvas[:Hh, :W] = frame
        canvas[:, W:] = panel
        writer.write(canvas)
        if i in (57, 85, 232):
            cv2.imwrite(str(OUT_DIR / f"demo_{i:04d}.png"), canvas)
        i += 1

    writer.release()
    print(f"-> outputs/m2/events_demo.mp4 ({i} frames)")


if __name__ == "__main__":
    main()
