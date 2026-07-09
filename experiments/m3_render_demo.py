"""M3: demo artifacts — the rally that types its own chart.

Video (shots_demo.mp4): M2's side-by-side, plus
  - both player boxes tracked on the broadcast panel (near orange,
    far pink), striker's box flashes thick with FOREHAND/BACKHAND at
    each hit
  - player feet as live dots on the court panel (homography-mapped —
    feet are on the ground plane, so these are exact, unlike the ball)
  - a bottom bar where the pseudo-MCP string TYPES ITSELF: letter at
    each hit, direction digit at each landing, '?'s where the clip
    can't answer (serve, final landing, ending)

Still (shot_map.png): every shot as an arrow from where the striker
stood to where the ball landed, labeled with its letter.

Usage:
    uv run experiments/m3_render_demo.py clips/rally.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "m3"
ROOT = Path(__file__).resolve().parent.parent

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
BAR_H = 92
LAST_FRAME = 298
FLASH = 14

C_BALL = (0, 255, 255)
C_NEAR = (0, 165, 255)
C_FAR = (255, 100, 255)


def load():
    H = np.load(ROOT / "outputs/m1/H_img_to_court.npy")
    ball = {}
    with open(ROOT / "outputs/m0/trajectory_ballfix.csv") as f:
        for row in csv.DictReader(f):
            ball[int(row["frame"])] = (float(row["cx"]), float(row["cy"]),
                                       float(row["w"]), float(row["h"]))
    players = {}
    with open(OUT_DIR / "players_traj.csv") as f:
        for row in csv.DictReader(f):
            players.setdefault(int(row["frame"]), {})[row["player"]] = {
                "cx": float(row["cx"]), "cy": float(row["cy"]),
                "w": float(row["w"]), "h": float(row["h"]),
                "foot_x": float(row["foot_x"]), "foot_y": float(row["foot_y"]),
            }
    events = []
    with open(ROOT / "outputs/m2/events_v4.csv") as f:
        for row in csv.DictReader(f):
            events.append({"frame": int(row["frame"]), "kind": row["kind"]})
    chart = list(csv.DictReader(open(OUT_DIR / "chart_v2.csv")))
    return H, ball, players, events, chart


def build_string_events(events, chart):
    """(frame, char) list: letter at each hit, zone digit at its landing."""
    out = []
    hits = [e for e in events if e["kind"] == "hit"]
    bounces = [e for e in events if e["kind"] == "bounce"]
    for k, (h, s) in enumerate(zip(hits, chart)):
        out.append((h["frame"], s["shot_type"]))
        nxt = hits[k + 1]["frame"] if k + 1 < len(hits) else 10 ** 9
        landing = next((b for b in bounces if h["frame"] < b["frame"] < nxt), None)
        if landing:
            out.append((landing["frame"], str(s["direction_zone"])))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    args = parser.parse_args()

    H, ball, players, events, chart = load()
    hit_info = {int(s["frame"]): s for s in chart}
    string_events = build_string_events(events, chart)

    def img_to_court(px, py):
        return cv2.perspectiveTransform(
            np.float32([[px, py]]).reshape(-1, 1, 2), H).reshape(2)

    # event court positions for the accumulating marks (ball shadow, as M2)
    def ball_court_pos(fi, win=4):
        pts = []
        for f in range(fi - win, fi + win + 1):
            if f in ball:
                cx, cy, _, _ = ball[f]
                pts.append(img_to_court(cx * 1280, cy * 720))
        return np.median(np.array(pts), axis=0)

    for e in events:
        e["pos"] = ball_court_pos(e["frame"])

    # ---- court panel base ----
    w_panel_full = int((W_COURT + 2 * MARGIN_M) * PPM)
    h_panel_full = int((L_COURT + 2 * MARGIN_M) * PPM)
    court_img = np.full((h_panel_full, w_panel_full, 3), (146, 91, 59), np.uint8)

    def cpt(x, y):
        return int((x + MARGIN_M) * PPM), int((y + MARGIN_M) * PPM)

    def cline(x1, y1, x2, y2, color=(255, 255, 255), t=2):
        cv2.line(court_img, cpt(x1, y1), cpt(x2, y2), color, t)

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

    def to_panel_px(pt, clip_lo=-2, clip_hi=L_COURT + 2):
        return (int((pt[0] + MARGIN_M) * PPM * scale),
                int((np.clip(pt[1], clip_lo, clip_hi) + MARGIN_M) * PPM * scale))

    # ---- video ----
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(OUT_DIR / "shots_demo.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W + panel_w, PANEL_H + BAR_H))

    ball_trail, marks = [], []
    i = 0
    while i <= LAST_FRAME:
        ok, frame = cap.read()
        if not ok:
            break

        # player boxes, every frame
        pl = players.get(i, {})
        for side, color in (("near", C_NEAR), ("far", C_FAR)):
            if side not in pl:
                continue
            p = pl[side]
            x1 = int((p["cx"] - p["w"] / 2) * W); y1 = int((p["cy"] - p["h"] / 2) * Hh)
            x2 = int((p["cx"] + p["w"] / 2) * W); y2 = int((p["cy"] + p["h"] / 2) * Hh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ball box
        if i in ball:
            cx, cy, w, h = ball[i]
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * Hh)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * Hh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 2)
            ball_trail.append(to_panel_px(img_to_court(cx * W, cy * Hh)))

        # event flashes
        for e in events:
            d = i - e["frame"]
            if d == 0:
                marks.append(e)
            if not (0 <= d < FLASH):
                continue
            if e["kind"] == "bounce":
                cv2.putText(frame, "BOUNCE", (W // 2 - 120, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 9)
                cv2.putText(frame, "BOUNCE", (W // 2 - 120, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4)
            else:
                s = hit_info[e["frame"]]
                side = s["striker"]
                color = C_NEAR if side == "near" else C_FAR
                word = "FOREHAND" if s["shot_type"] == "f" else "BACKHAND"
                cv2.putText(frame, word, (W // 2 - 170, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 9)
                cv2.putText(frame, word, (W // 2 - 170, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
                if side in pl:
                    p = pl[side]
                    x1 = int((p["cx"] - p["w"] / 2) * W); y1 = int((p["cy"] - p["h"] / 2) * Hh)
                    x2 = int((p["cx"] + p["w"] / 2) * W); y2 = int((p["cy"] + p["h"] / 2) * Hh)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)

        # court panel
        panel = base_panel.copy()
        for p in ball_trail[:-1]:
            cv2.circle(panel, p, 2, (80, 200, 255), -1)
        if ball_trail:
            cv2.circle(panel, ball_trail[-1], 6, C_BALL, 2)
        for e in marks:
            p = to_panel_px(e["pos"])
            if e["kind"] == "bounce":
                cv2.circle(panel, p, 9, (63, 210, 255), -1)
                cv2.circle(panel, p, 9, (0, 0, 0), 2)
            else:
                cv2.drawMarker(panel, p, (85, 85, 255), cv2.MARKER_TILTED_CROSS, 18, 4)
        for side, color in (("near", C_NEAR), ("far", C_FAR)):
            if side not in pl:
                continue
            p = pl[side]
            fpt = to_panel_px(img_to_court(p["foot_x"], p["foot_y"]),
                              clip_lo=-4, clip_hi=L_COURT + 4)
            cv2.circle(panel, fpt, 8, color, -1)
            cv2.circle(panel, fpt, 8, (0, 0, 0), 2)

        # MCP string bar
        s = "?" + "".join(ch for f, ch in string_events if f <= i)
        if i >= 293:
            s += "??"
        bar = np.full((BAR_H, W + panel_w, 3), (24, 24, 24), np.uint8)
        cv2.putText(bar, "pseudo-MCP", (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (160, 160, 160), 2)
        cv2.putText(bar, s, (18, 78), cv2.FONT_HERSHEY_DUPLEX, 1.3,
                    (255, 255, 255), 2)
        cv2.putText(bar, "serve + ending: not in clip", (W + panel_w - 420, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

        canvas = np.zeros((PANEL_H + BAR_H, W + panel_w, 3), np.uint8)
        canvas[:Hh, :W] = frame
        canvas[:PANEL_H, W:] = panel
        canvas[PANEL_H:, :] = bar
        writer.write(canvas)
        if i in (57, 153, 285):
            cv2.imwrite(str(OUT_DIR / f"demo_{i:04d}.png"), canvas)
        i += 1

    writer.release()
    print(f"-> outputs/m3/shots_demo.mp4 ({i} frames)")

    # ---- shot map ----
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.add_patch(plt.Rectangle((-3, -4), W_COURT + 6, L_COURT + 8,
                               color="#3b5b92", zorder=0))

    def mline(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="white", lw=1.5, zorder=1)

    mline(0, 0, W_COURT, 0); mline(0, L_COURT, W_COURT, L_COURT)
    mline(0, 0, 0, L_COURT); mline(W_COURT, 0, W_COURT, L_COURT)
    mline(SINGLES_INSET, 0, SINGLES_INSET, L_COURT)
    mline(W_COURT - SINGLES_INSET, 0, W_COURT - SINGLES_INSET, L_COURT)
    mline(SINGLES_INSET, SVC_FAR_Y, W_COURT - SINGLES_INSET, SVC_FAR_Y)
    mline(SINGLES_INSET, SVC_NEAR_Y, W_COURT - SINGLES_INSET, SVC_NEAR_Y)
    mline(CENTER_X, SVC_FAR_Y, CENTER_X, SVC_NEAR_Y)
    ax.plot([0, W_COURT], [NET_Y, NET_Y], color="#dddddd", lw=2.5,
            linestyle=(0, (4, 2)), zorder=1)

    hits = [e for e in events if e["kind"] == "hit"]
    bounces = [e for e in events if e["kind"] == "bounce"]
    for k, s in enumerate(chart):
        color = "#ff9d2e" if s["striker"] == "near" else "#e06cf0"
        sx, sy = float(s["striker_court_x"]), float(s["striker_court_y"])
        ax.scatter(sx, sy, s=150, color=color, edgecolors="black",
                   linewidths=1.2, zorder=3)
        ax.annotate(f"{s['shot']}{s['shot_type']}", (sx, sy),
                    textcoords="offset points", xytext=(10, 4),
                    fontsize=11, fontweight="bold", color="white", zorder=4)
        h = hits[k]
        nxt = hits[k + 1]["frame"] if k + 1 < len(hits) else 10 ** 9
        landing = next((b for b in bounces if h["frame"] < b["frame"] < nxt), None)
        if landing is not None:
            lx, ly = landing["pos"]
            ly = np.clip(ly, -2, L_COURT + 2)
            trusted = "low" not in s["landing_trust"]
            ax.annotate("", xy=(lx, ly), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2,
                                        linestyle="-" if trusted else (0, (4, 3)),
                                        alpha=0.9 if trusted else 0.55), zorder=2)
    ax.scatter([], [], s=150, color="#ff9d2e", edgecolors="black", label="Zverev (near) strikes")
    ax.scatter([], [], s=150, color="#e06cf0", edgecolors="black", label="Gasquet (far) strikes")
    ax.plot([], [], color="gray", lw=2, label="shot -> landing")
    ax.plot([], [], color="gray", lw=2, linestyle=(0, (4, 3)), label="landing low-trust (far side)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-3, W_COURT + 3)
    ax.set_ylim(L_COURT + 4.5, -4.5)
    ax.set_aspect("equal")
    ax.set_title("M3 — the rally as shots: who hit, with which wing,\n"
                 "from where, to where  (letters 7/7 frame-verified)")
    ax.set_xlabel("court x (m)")
    ax.set_ylabel("court y (m)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shot_map.png", dpi=150)
    print("-> outputs/m3/shot_map.png")


if __name__ == "__main__":
    main()
