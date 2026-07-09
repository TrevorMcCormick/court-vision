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

## 2026-07-09 — M0 writeup on the (still gated) site

- Wrote devlog #2 from this log: "M0: Can SAM 3 Hold Onto a Tennis Ball?" —
  chronological runs 1–4, ball-only scope owned explicitly (players carried
  forward as unfinished business), ~$0.60 cost line, all four artifacts
  embedded.
- Site got its first "sets" plumbing: a post can declare `video:` frontmatter
  (or drop `{slug}-hero.mp4` in img/blog) and the blog feed tile + post hero
  play the clip muted/looping — first step toward the disneyplus-style
  browse-rows vision. The M0 tracked-rally clip (trimmed to frames 0–290,
  570KB) is the first tile. Gate still up; launch is a separate decision.

## 2026-07-09 — M1 experiment 1: the clean plate

**Run 1 — static-camera check via temporal median.**
- Per-pixel median over 96 sampled frames (`m1_clean_plate.py`): the players
  and ball VANISH, court lines stay razor sharp. Edge IoU frame0-vs-median
  0.831, Laplacian variance ratio 0.90 → camera is static for the rally.
  **One homography can serve all 480 frames.**
- Lesson: **the temporal median frame is free player removal.** No
  segmentation, no inpainting — just `np.median` — and line detection gets
  an empty court to work with. (Faint ghosts of line judges remain; they
  barely move. Broadcast text overlays survive too, of course.)

**Run 2 — Hough on the clean plate, masked to white-inside-blue.**
- Blue-court mask (largest blue contour, 26.5% of frame) → white mask
  restricted to its dilated hull → HoughLinesP: 48 segments covering every
  court line. The white mask is basically a line drawing of the court.
- Two details for the fit step: the **net tape reads as a horizontal line**
  (and it sags — curved, and wider than the court), and faint "ATP WORLD
  TOUR" text ghosts survive on the net band. Plan: fit from the four outer
  doubles corners (extremes only — the net can't be topmost or bottommost),
  validate against the held-out service/singles lines.

**Run 3 — homography from four corners. Sub-pixel on held-out lines.**
- Clustered segments (5 horizontal families incl. the net at y≈244 — safely
  mid-pack, never an extreme; 5 vertical), merged extremes into the two
  baselines + two doubles sidelines, intersected → four corners →
  `getPerspectiveTransform` against the court model (10.97 × 23.77 m).
- Validation held out everything else: reprojected singles sidelines,
  service lines, and center line land **0.0–0.8 px mean** from the detected
  white pixels. Four corners was enough; no refinement needed.
- Detail worth keeping: the model's "net line" is the net's *ground plane*
  projection — dead straight — while the real net tape sags visibly above
  it in the overlay. The court is geometry; the net is physics.

**Run 4 — M0 track mapped to court coordinates. M1 ACHIEVED.**
- Applied img→court H to the M0 trajectory (290 points) and drew it on a
  to-scale court (`track_on_court.png`). The shadow track reads correctly:
  crossings at the net, clusters near the baselines where the ball is low.
- The caveat is the finding: **a homography maps the ground plane, and the
  ball is airborne most of the rally.** Points project up to 18 m beyond
  the far baseline when the ball is high — not a fit error (held-out lines
  are sub-pixel), just physics. The shadow is exact at bounces, which is
  the hand-off to M2: find the bounces, and the mapped positions become
  chartable.

**M1 verdict:** achieved, and — unlike M0 — with **zero manual input**. The
clean plate, the line detection, the corner fit: all automatic. Total cost:
$0.00 (no API calls; the whole milestone is numpy + OpenCV on one frame).
Contrast with M0's one-manual-click bootstrap noted for the writeup.

**Artifacts:** `outputs/m1/` — clean_plate.png, white_mask.png,
lines_overlay.png, model_reprojection.png (the alignment proof),
track_on_court.png (money chart), track_court.csv, H matrices (.npy).
- Rendered the M1 demo video for the devlog: broadcast + box on the left,
  top-down shadow track drawing itself on the right (`sidebyside.mp4`,
  `m1_render_sidebyside.py`). Pixels in, meters out, one clip.

## 2026-07-09 — M2 experiment 1: hits and bounces

**Run 1 — velocity structure of the M0 track.**
- 290 usable frames, 10 small gaps (SAM blips, 1–2 frames each) — gaps need
  care in finite differences.
- Trap identified: **image-y is not height.** It mixes physical height with
  court depth — a ball at the far baseline sits high in the frame even when
  it's physically low. The M0 "parabolas" are projection mixtures.
- The court-shadow velocity explodes to ±200 m/s when the ball is high on
  the far side — projection amplification of airborne motion. Ground speeds
  are only meaningful near bounces. Same lesson as M1, seen from the
  velocity side.
- The per-half-rally signature is visible by eye: small cusp (bounce),
  then a deep extreme with a violent reversal (hit). Detector plan: cusps
  in image-y velocity = candidate events; classify by whether the
  along-court direction of travel flips (hit) or persists (bounce), using
  windowed medians to survive the projection spikes and gaps.

**Runs 2–4 — detect, verify by eye, fix, repeat.**
- v1 (image-y cusps + direction-flip classify): 9 events, 8 "hits". Smelled
  wrong — rallies pair bounce→hit. Pulled video frames at every event
  (`m2_verify_frames.py`) and eyeballed them. Ground truth: 6 hits
  (f2, 57, 99, 153, 185, 243), 5 bounces (f44, ~85, f139, ~179, f232).
- v1's two failures, both instructive:
  1. **Near bounces classified as hits** — post-bounce shadow noise
     (-2..-4 m/s) fooled the sign-flip test. Fix: classify on OUTGOING
     shadow speed (returns leave >5 m/s; bounces collapse to ~0). v2: 9/9
     correct on matched events.
  2. **Far bounces are invisible in image-y, structurally.** After a far
     bounce the ball rises AND recedes — both push image-y the same
     direction, so there is no cusp at all. Not a threshold problem; the
     signal doesn't exist in that coordinate.
- The far-bounce signal that does exist: **shadow-speed collapse.** While
  airborne the shadow races (projection-amplified, -30..-50 m/s); the
  instant the ball is on the ground the shadow moves at true ball speed
  (-5..-8 m/s). A >3x |vcy| collapse without a sign flip, in the far half,
  marks the bounce. → v3.

**Runs 5–6 — false positive teaches the physics; v4 goes 13/13.**
- Frame-checked v3's new detections. f71 "bounce" = the ball crossing the
  net, mid-air. The lesson: **the shadow-speed collapse isn't an instant at
  contact — it unwinds gradually through the descent** as the ball loses
  height and the projection amplification deflates. Early collapses are
  mid-descent artifacts; **the bounce is the LAST collapse of a descent.**
- The rally's tail also wasn't over when I thought: frames 250–298 show one
  more far bounce (~257) and Gasquet's final swing (~280). Ground truth
  grew to 7 hits + 6 bounces.
- v4 (last-collapse-per-segment + post-window median positions):
  **13/13 events matched (±8 frames), 13/13 classified correctly, 0 false
  positives.** All frame-verified.
- Honest caveats, on the record: thresholds are tuned to this one rally
  (generalization untested); far-bounce positions are ±meters — at the far
  baseline, one frame of timing is meters of shadow, and the ball is
  already moving away again post-bounce. Near-side bounce depths look
  right (19.4 / 19.6 / 20.4 m — classic groundstroke depth, ~3.5 m inside
  the baseline).

**M2 verdict:** achieved on the M0 rally — every hit and bounce detected
and frame-verified, hit-vs-bounce separated by outgoing shadow speed.
Still $0 total for the milestone. The event sequence IS proto-charting:
hit(f2) → bounce → hit ... is one shot-by-shot rally description. M3 turns
that into notation.
- Rendered the M2 devlog artifacts: `event_map.png` (all 13 events placed
  on the court, far-bounce uncertainty drawn honestly as hollow markers)
  and `events_demo.mp4` (side-by-side with HIT/BOUNCE flashes and event
  marks accumulating on the court panel).

## 2026-07-09 — M3 scoping: the proto-chart and its question marks

Hand-rolled the fullest MCP-style chart the M2 events can support
(`m3_proto_chart.py`). Result for the rally:

    # striker dir depth landing
    1 far      2    8    18.7m   <- directions + depths: derivable NOW
    2 near     1    9   -10.8m   <- far-bounce position error, visible in chart form
    ...
    7 far      ?    ?       ?    <- rally end unresolved

    pseudo-MCP: ??2?1?1?2?2?2???  rallyCount 7 (+ unseen serve)

- Free validation: the striker column alternates far/near/far/near…
  perfectly — never enforced, pure emergent consistency of the detections.
- Every '?' is an M3 requirement, now concrete:
  1. **Player tracking** — shot type f/b needs the striker's position and
     contact side. M0's unfinished business is now on the critical path.
  2. **Full-point clips** — serves and point endings aren't in a
     mid-rally highlight clip. Clip sourcing round 2: full-match VOD,
     multiple complete points, camera-cut handling.
  3. **In/out + endings** — */@/# codes need trustworthy far-bounce
     positions (shot 2 "landed" at -10.8m and the rally continued —
     that's the error bar talking, and it's meters).
  4. **Player identity/handedness** — MCP direction zones are defined
     relative to the receiver; needs to know who's who.
  5. **Rally segmentation** — point/game context requires slicing a match
     into points across broadcast cuts.
- M3 is a different shape than M1/M2: multiple sessions, back to clip
  sourcing, and player tracking first. Scoped; not started.
