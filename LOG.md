# Build log

Running log of what happened, in order, including the dead ends. Raw material
for devlog posts at trmccormick.com. Newest entries at the bottom.

---

## 2026-07-08 — Project kickoff

- Rebooted trmccormick.com as a build-in-public devlog for this project
  (killed the 35-post instrumentation blog that never shipped — if I haven't
  written about it yet, I don't actually want to write about it).
- Chose the moonshot goal: broadcast video in → Match Charting Project
  notation out. Milestones M0–M4 defined in README.
- Compute decision: hosted SAM 3 via fal.ai first, self-host later only if
  volume demands it. fal has `fal-ai/sam-3/video` (masked overlay video) and
  `fal-ai/sam-3/video-rle` (RLE masks + per-frame boxes) at $0.005 per 16
  frames. I already had a FAL_KEY from generating blog art.

## 2026-07-09 — M0 experiment 1: first contact with SAM 3

**Setup.** Scaffolded this repo (uv, Python 3.12). Grabbed a 16 s broadcast
rally (Zverev–Gasquet, Montreal) at 720p/30fps = 480 frames from a highlights
compilation via yt-dlp. First clip attempt was unusable — highlights reels
cut away from the broadcast angle after ~5 s (rally → player close-up →
crowd), so finding a *continuous* single-angle rally took two tries. Lesson:
**clip sourcing is its own pipeline step.** Compilations are full of camera
cuts; full-match VODs will be the real source.

**Run 1 — text prompt "tennis ball, tennis player", both endpoints.**
- `sam-3/video` overlay: players segmented nicely early on... along with the
  line judges and ball kids (fair — they are dressed like players). Then at
  ~frame 240 the mask flipped to segmenting THE ENTIRE COURT. fal processes
  video in chunks (`Completed chunk 2/2` in the logs) and the second chunk
  apparently re-ran concept detection and latched onto something else.
  Lesson: **chunk boundaries are concept-drift boundaries.**
- `sam-3/video-rle` with the same prompt: returned exactly ONE box per frame
  — a giant merged region, not per-instance data. `scores` all null.

**Run 2 — text prompt "tennis ball" alone, threshold 0.3.**
- 480 boxes returned, but the trajectory plot showed long dead-flat plateaus
  — a tracker glued to something stationary. Rendered the boxes back onto
  the video: the box is the **union of every tennis ball in frame**. Ball
  kids hold spare balls. The actual game ball (a ~8 px yellow smudge) sits
  outside or dwarfed inside a half-court-sized union box.
  Lesson: **open-vocabulary text prompts match ALL instances of a concept —
  "tennis ball" is not "THE tennis ball".** Broadcast tennis has ~6 balls
  in frame at any time.

**Current hypothesis for getting the game ball:** visual prompt instead of
text — a box drawn around the specific ball in a specific frame (SAM 2-style
single-object tracking). Found the game ball by eye in frame 240 at
(622, 462) px. Next: box-prompt run.

**Run 3 — box prompt on the game ball (pixels, frame_index=240).**
- Sent `box_prompts: [{x_min, y_min, x_max, y_max, frame_index: 240}]` in
  pixel coordinates. 470/480 frames came back with boxes.
- First parse produced NEGATIVE box widths. That was the tell:
  **box-prompted responses use [cx, cy, w, h] (normalized center+size), not
  [x1, y1, x2, y2].** Undocumented; discovered by decoding frame 240's box
  and finding it exactly ball-sized at exactly the prompt location.

**Run 4 — reparse with correct format. M0 ACHIEVED.**
- Frames 0–290: ball-sized boxes (~8–24 px) tracing clean flight arcs. SAM 3
  propagated the frame-240 prompt BACKWARDS to frame 0 as well as forwards —
  bidirectional tracking works on the hosted endpoint.
- The y-vs-frame plot is the money chart: ~3 full baseline-to-baseline round
  trips, smooth parabolic arcs with sharp reversals at each hit. The
  hit/bounce discontinuity structure M2 needs is plainly visible.
- Track dies at ~frame 300 (rally ends, ball leaves play) and the box goes
  giant-and-stuck. Lesson: **track death shows up as a frozen oversized box**
  — easy to filter (dropped anything wider than 0.05 normalized).
- Verified visually: rendered boxes back onto the clip; the box rides the
  ball through flight, including in front of busy crowd backgrounds.

**M0 verdict:** SAM 3 can hold onto a tennis ball at broadcast speed — IF you
prompt it visually with the specific ball. Text prompts alone can't isolate
the game ball from the spares. Cost of the whole milestone: ~$0.60 in API
calls. Open problem for M3: the one-manual-click-per-rally bootstrap needs
automating (candidate: text-prompt all balls, pick the one that MOVES).

**Artifacts:** `outputs/m0/` — overlay video, box-render videos
(`boxes_ballfix.mp4` is the keeper), trajectory plots (flat-plateau failure
+ `trajectory_ballfix.png` money chart), raw JSON responses.

## 2026-07-09 — Site gated behind Coming Soon

Launched the rebooted trmccormick.com public for a few hours, then put the
password gate back with a minimal "Court Vision — Coming Soon" card. Building
in public, but launching when the M0 writeup is actually on the site — a
coming-soon devlog with one post felt premature. Gate is `src/theme/Root.js`
in the website repo; same password as before the reboot.
