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

## 2026-07-09 — M3 experiment 1: the players, at last

**Run 1 — two box prompts, one call.**
- Prompted both players on frame 240 (boxes drawn by eye, render-verified:
  `prompt_boxes_240.png`; the red-shirted line judge behind Gasquet kept
  deliberately OUTSIDE Gasquet's box). One `video-rle` call, two
  `box_prompts`.
- The response merged them: `boxes` is ONE box per frame — the UNION of
  both players... plus that line judge, whom nobody prompted. M0's union
  lesson again, now for visual prompts: **multiple box prompts in one call
  come back as one concept-blob, not per-object tracks.** (Whether the
  endpoint merges per-object masks at serialization or treats two boxes as
  exemplars of one concept, the per-object structure is gone either way.)
- **The mask still knows what the box forgot.** Decoded the RLE (format:
  space-separated start/run pairs over the row-major 1280×720 frame) and
  took connected components: three clean, spatially disjoint blobs —
  Zverev, Gasquet, judge — at every frame spot-checked. Baseline rally;
  nobody crosses anybody in image space.

**Run 2 — component split; the line judge erases himself.**
- Static-pixel filter before splitting: any pixel on in >85% of frames is
  scenery. That's exactly the judge (x 770–807, y 61–106, never moves).
  Players never stand that still for 16 seconds. **The mask's only
  impostor filtered himself out by never moving.**
- Near/far assignment by which half the component's bottom edge sits in:
  **480/480 frames for BOTH players, zero gaps** — the ball only managed
  470. Player-sized objects are easy mode for SAM 3.
- Two brief far-track glitches (~f390, ~f425), both AFTER the rally ends
  (~f300): the component jumps to another far-side figure between points.
  Outside charting scope; on the record.
- **Feet are on the ground plane.** Box bottom-center through the M1
  homography is exact — the airborne-ball caveat doesn't apply to feet.
  The pipeline's most trustworthy positions are now the players', not the
  ball's.

**Run 3 — striker + contact side → the letters. 7/7.**
- Striker by proximity (ball-to-box distance at the hit frame) agrees 7/7
  with M2's court-y heuristic and preserves the emergent far/near
  alternation. Three independent answers, same rally.
- Contact side: ball x minus striker center x, MIRRORED for the far player
  (he faces the camera — his right is image-left). Right-handed assumption
  for both (true for Zverev and Gasquet) — **handedness is assumed, not
  detected**; on the record, and it's M3 requirement #4's problem anyway.
- Letters: f f f b f f b. Frame-verified all seven with zoomed contact
  strips (`m3_verify_shots.py`). Shot 3's dx was −1.5 px — dead-center
  contact, coin-flip territory — and the video still says forehand (low
  pickup off the right shoe), so the letter stands but the signal was
  luck. Contact-side dx needs a confidence threshold before it meets a
  rally it wasn't tuned on.
- Shot 7 is Gasquet's rally-ending one-handed backhand, takeback coiled
  high in the strip. If you have to eyeball-verify a backhand, make it
  Gasquet's.

**Chart v2:** `??2?1?1?2?2?2???` → `?f2f1f1b2f2f2b??`. rallyCount 7
(+ unseen serve). Striker positions now in the chart: Zverev struck from
0–1.3 m behind his baseline, Gasquet from 3.1–3.6 m behind his — very
Gasquet, but far-side foot pixels are meters-per-pixel territory, so far
depths stay soft.
- Session cost: $0.15 (one video-rle call). Project total: ~$0.75.
- Rendered: `shots_demo.mp4` (player boxes + FOREHAND/BACKHAND flashes +
  live court dots + the pseudo-MCP string typing itself along the bottom)
  and `shot_map.png` (every shot as a striker→landing arrow, dashed where
  the landing is far-side ±meters).
- Remaining '?'s unchanged and unbluffed: serve, shot-7 landing, ending
  codes, direction-zone semantics, rally segmentation — all gated on
  **full-point clips. Clip sourcing round 2 is the next session.**

## 2026-07-09 — M3 experiment 3: clip sourcing round 2 — the cutting room

**Sourcing.** The M0 rally turns out to be a slice of a famous point — the
49-shot rally from the Zverev–Gasquet Montreal R2 match (that 5–6 30-40
score bug was match-point context all along). Better: a **24-minute
extended-highlights reel of the same match** exists. Same broadcast, same
court, same framing → M1's homography should transfer. Downloaded at
720p: 36,234 frames. One wrinkle for later: **the reel is 25 fps where
rally.mp4 was 30** — M2's velocity thresholds are secretly per-frame;
fps must become a parameter before charting these points.

**Run 1 — color classifier; noon breaks what evening tuned.**
- Per-frame features: blue-court fraction (M1's HSV range), largest-blue-
  contour centroid, green-apron fraction. Evening-tuned rule → ~43%
  court view.
- Eyeball check: all sampled court-view frames correct — and six obvious
  broadcast frames in the reject pile, ALL daytime. Diagnosis: in daylight
  the shaded stadium seats read blue and MERGE with the court into one
  50%-of-frame blob, while the sunlit apron washes out below the green
  saturation threshold (S=38 vs 40). **Evening-tuned color rules break at
  noon.** This match starts in hard sun and ends under lights.

**Run 2 — geometry probes; the homography is a camera fingerprint.**
- M1's court quad lands pixel-perfect on reel frames from both ends of
  the match (`framing_check.png`) — same framing as rally.mp4. So stop
  classifying colors and probe geometry: 50 interior court points must
  read court-blue, 17 apron points just outside the doubles lines must
  not. Sun and shade both keep the court blue enough (S 113–128); the
  washed apron isn't blue either way. **The homography doubles as a
  camera-pose fingerprint.**
- 50.5% of frames court-view → 60 segments ≥3 s (gaps ≤0.6 s merged),
  725 s of chartable play.
- Verified by eye: all 60 segment starts are true broadcast views, most
  opening on a serve toss — **highlight editors cut to the point just
  before the serve, so the serve comes free with the cut.** Sampled ends:
  dead ball before every cut. Editors deliver complete points.
- Audited all 21 inter-segment gaps >15 s: 20 are genuine cutaways —
  including sideline and net-cam REPLAYS, which the probes correctly
  reject (replays would double-chart points and carry the wrong
  homography). One real loss: **a 131-second early-set-1 block where the
  broadcast ran a slightly tighter zoom** — interior probes still hit
  court, but the bigger in-frame court swallows the apron probes
  (apron_hit 0.35–0.41). The "fixed broadcast framing" assumption is only
  mostly true: framing drifted early, then settled. Known hole, on the
  record; auto-refitting H per block (M1's fit is already zero-manual) is
  the fix if a future match needs it.

**Extraction.** 60 frame-accurate point clips → `clips/points/`
(gitignored as ever). `point_51` = 59.6 s of continuous court view =
**the 49-shot rally in full, serve included, found by the segmenter on
its own** — the pipeline has been charting a 16-second excerpt of a
49-shot novel; now it has the whole book. Artifacts:
`point_timeline.png` (24 minutes → 60 green bars + one honest orange
hole), `points_montage.mp4` / `points_flash.mp4` (every point start,
wall to wall).
- Session cost: **$0.00** — no API calls; the whole session is HSV
  thresholds and 67 pixel probes per frame.
- What this unblocks: serves are IN these clips, endings are IN them,
  and charting becomes a loop over 60 inputs. Next session: point
  anatomy — serve detection inside a segment — plus the fps parameter
  cleanup.

## 2026-07-09 — M3 experiment 4–5: $0 player tracking and point anatomy v1

**The pitch.** SAM player tracking works but costs ~$0.15/clip — $9 for
the 60 points, forever, every rerun. M1 gave the alternative away free:
the temporal median erases players, so frame-minus-plate IS a player
detector. Getting bg-sub to actually work surfaced three findings that
matter beyond the savings.

**Finding 1 — the "static" camera pans.** ECC alignment of every frame
to frame 0: up to **32 px of real mid-point camera movement** (the
operator follows play), slow ~9 px drift over the 60 s rally, plus a
transient nudge mid-rally. Invisible to the eye; fatal to a median plate;
meters of error at the far baseline. M1's "one homography per rally"
survived because the M0 clip was 16 seconds — at 60 s the assumption
quietly dies. Fix: translation-only ECC stabilization of every frame
before plate, subtraction, or homography. (`camera_pan.png` is the
receipt.)

**Finding 2 — short clips bake the far player into the plate.** He's
small and stands nearly still between shots; the per-clip median keeps a
ghost of him and his diff goes ~zero. Near player 100% coverage, far
~35%. Fix: subtract a **deep plate** — median of the clip's own plate
and its 4 neighbors' (ECC-aligned); players stand in different spots
point to point, so ghosts wash out. Far coverage: 35% → **96% median**
(near stays 100%).

**Finding 3 — the far corners hide ballkids.** First serve-detection run
produced "far players" at court x = 13.5 m and −3 m — OUTSIDE the court.
The crouching far-corner ballkids sit ~2.5 m outside the doubles lines,
inside the generous tracking region, and out-diff the motionless
pre-serve far player. Fix: per-half regions (tight far, generous near).
Impostor lesson #3, same shape as the line judges: **everyone near a
tennis court looks like a tennis player to a difference image.**

**Point anatomy v1 → v2.** Server = whoever hugs the center mark early;
toss = the blob-height peak (a toss stretches the silhouette); side of
the mark = deuce/ad for free. v1 (nearest-to-center wins, no gates) was
seduced by cold opens, zoom-tail garbage, and RECEIVERS who stand near
the center hash — frame strips put it near a coin flip. v2 gates every
claim (early coverage ≥60%, ≤2 m from the mark, ≤~3 m from the own
baseline, height peak ≥1.12× median) and refuses to guess otherwise:
**40/60 clips yield a confident serve; 20 are explicit, reasoned
rejections** (the reel's cold open, mid-point rejoins, two long clips
with sparse early far tracks — including the 49-shot point).
- Frame-verified sample of the confident calls: **6/8 correct server
  end, 5/8 usable serve moments.** Failure modes, on the record: a
  returner 1 m off the center hash whose return swing reads as a "toss"
  (point_40), and servers whose 4-second window catches ball-bouncing
  routine instead of the toss (point_10). Geometry-only anatomy tops out
  here; **the ball adjudicates** — serve = first ball launch — when ball
  tracking runs over these clips.
- fps is a parameter everywhere new (`CAP_PROP_FPS`, never assumed);
  M2's hardcoded `* 30.0` stays in the M2 scripts as history and gets
  parameterized when the detector graduates into the charting loop.
- Session cost: **$0.00.** Project total: still ~$0.75. The $9 SAM
  charting loop decision is now genuinely optional for players — SAM is
  needed for the BALL only.
- Artifacts: `serve_gallery.mp4` (all 40 confident serves, called),
  `camera_pan.png`, verified serve strips (`serve_checks/`),
  per-clip plates + tracks (`plates/`, `players/`).

## 2026-07-09 — M3 experiment 6–7: the charting loop, pilot run

**The toss is the bootstrap.** M0's open problem — one manual ball-click
per clip — closes itself: in the frames before contact, the toss ball is
a small, isolated diff blob directly above the server's head. Box it,
prompt SAM, done. 5/8 pilot clips bootstrapped off the toss. Fallback for
the rest: fastest small mover — which promptly locked onto **the score
bug's flip animation and a vibrating net sign** (broadcast overlays are
fast small movers too). Fixed with overlay masks + a 3-point ballistic
chain requirement (consistent direction and step). Two clips still
bootstrap onto line-junction glare during camera pans; parked, not
bluffed.

**Pilot: 6 clips through SAM, $0.55.** (The first send bounced for free —
`box_prompts` coordinates must be integers.) Ball coverage 73–93% per
clip; dropped frames are mostly post-point dead time.

**v4 detector, finally fps-parameterized** — swing thresholds and gap
windows scale by 30/fps, physical m/s thresholds don't. And a new
failure mode the M0 rally could never show: **the pre-serve dribble is
real hits and real bounces.** The server bouncing the ball reads as
play. Fix: window events from the confident serve frame.

**First end-to-end charts, asterisks included.**
    point_16: s2b?b1f1f2b3b?b3?   (serve ✓, 8 shots, rally-shaped)
    point_31: s?f??               (short point: serve, return, over)
Frame-checking point_16: the serve call is right, the structure is
right, but the striker column breaks alternation wherever the detected
event frame catches the ball mid-air BETWEEN players — proximity is the
wrong striker signal at apex frames. point_43's mover bootstrap tracked
something that wasn't the ball. Letters not yet trustworthy; structure
~70%.

**Decision: the full 54-clip run ($5) is deferred.** Every known fix —
alternation-constrained striker assignment, contact-frame refinement,
sparse-track handling — is free to iterate on the six tracks already
paid for. The rest of the reel's budget waits until the letters earn it
on the pilot.
- Session cost: $0.55. Project total: ~$1.30 of a $9 budget.
- Site fix, same session: homepage Devlog tiles were falling back to
  text — the series plugin supplies each post's `videoUrl` but the
  homepage card only ever rendered images. Tiles now play the hero
  video, same as the feed rows.

## 2026-07-09 — M3 experiment 8: charting loop v2, then the reel

**Proximity died twice.** v1 picked the striker by whoever was closest
at the event frame; frame checks showed the cusp catches the ball
mid-air between players. The first replacement — ball DIRECTION after
the hit, sign of court-y velocity — died on contact with the data:
post-hit vcy reads negative for BOTH ends. The homography assumes the
ball is ON the ground plane, so an airborne ball's "court velocity" is
dominated by its vertical motion in image space. **The homography lies
about airborne balls.** (It's been quietly lying since M1 — landings
are fine because bounces happen on the plane; anything mid-flight is
fiction.)

**What replaced it: alternation as the constraint, votes as the
evidence.** A rally has exactly two striker assignments — plus one
parity flip wherever a hit could hide inside a ball-track HOLE (holes
are observable; SAM losing the ball f54–78 of point_53 is exactly where
a far hit vanished — the flipped chain NFFN is frame-verified). Votes,
strongest first: touch (3) — the ball reached exactly one player's box
in the contact window; serve call (2) — a PRIOR, not an anchor; landing
half (1) — far-half bounces say the near player struck (the collapse
detector is far-half-only by construction, one-sided but honest).
Making the serve outvotable paid off immediately: **point_59's
SERVER-OVERRIDE** — the gated serve call said near, the ball said far,
the frames say the ball was right.

**Letters gated on evidence.** Contact refines to the frame the ball is
nearest the ASSIGNED striker (proximity is safe once the striker is
known), and the f/b letter is committed only if the ball actually
reached the box — gate scaled by apparent player height (a fixed pixel
gate rejects every far-end letter; the far player is 55 px tall).
Player boxes go rogue at the worst moments — a "far" box on a
spectator by the Perrier cooler, a "near" box on a court shadow — so
those letters stay '?'. Committed letters frame-verified: 3/3 on
point_16, plus point_24 spot-check. Box quality is the next free win.

**The reel run was won at the dry-run.** Montage-grid review of all 54
prompt boxes killed 25 of 29 mover bootstraps — glare, shadow edges,
line junctions, the net band, a FedEx cooler, a ballkid. The ballistic
3-point chain isn't enough during camera pans; static features form
fake chains. Toss bootstrap fixes (free, one iteration): the above-head
window now scales with APPARENT height — 130 px above a 55 px far
server was searching the crowd — and a ball-shape filter drops
elongated racquet-arc blobs. Recovered 7 far-end tosses the fixed
window had thrown into the stands.

**Sent 23, got 18.** SAM lost 5 outright (giant stuck boxes — including
every "weak accept" I talked myself into; the eyeball grade was the
ground truth). Two API lessons: an all-dropped result crashed the batch
(now a graceful skip), and fal chunks long videos — clips past ~490
frames throw "No prompts available for this video chunk". point_55
worked trimmed to 450; **point_58 (893 frames, the 49-shot point) is
parked** until multi-prompt or split-and-stitch.

**21 points charted, up from 5.** Six far serves finally in the strings
(s5/s6 — serve_zone() is actually CALLED now; v1 defined it and coded
serves by full-court thirds). point_24 reads a complete `s5f3b3?`.
Every string still ends '?' — endings aren't coded. The conflicts
column is the honesty metric: votes that lost the argument, kept on the
record.
- Bootstrap scorecard, on the record: toss 18/25 tracked (72%), mover
  3/29 credible at dry-run (10%) — the toss IS the bootstrap; the mover
  fallback needs rethinking, not tuning.
- Session cost: ~$1.50. Project total: ~$2.80 of a $9 budget.

## 2026-07-09 — M3: the test set exists, and it has answers in the back

**The reel graduates to dev set.** Everything tuned so far — diff
thresholds, swing gates, serve anatomy, vote weights — was tuned on the
Zverev–Gasquet reel, with my own frame-checks as the only ground truth.
That's a training set with a human in the loop, and it stops being
evidence the moment the same eyes that tuned it also grade it. Best
practice from here: **freeze the pipeline before it touches a test
match, and test on matches somebody ELSE charted.**

**The Match Charting Project has the answer key.** Sackmann's MCP has
7,500+ men's matches hand-charted point by point — the exact notation
this project has been converging on. Our match isn't in it (checked:
only the Halle and Montpellier 2017 Gasquet–Zverevs are charted). But
**Canada Masters 2017 is** — same tournament, same court, same
broadcast package:

- **Test 1: Nadal–Shapovalov R16** (20170810, 225 charted points).
  Night session under lights, and BOTH players left-handed — the two
  assumptions the dev reel can't test (every letter call assumes a
  right-hander; every diff threshold was tuned in daylight). Same court
  and overlays, so failures attribute cleanly to what changed.
  Footage: the 8:21 TennisTV highlights, 720p60 (30,053 frames — fps
  is already a parameter, but SAM cost scales with frames, so
  downsample before any send).
- **Test 2 (reserve): Federer–Haase SF** (20170812, 128 points). Day,
  both right-handed — the nearest-transfer control. If the pipeline
  fails HERE, the problem isn't lefties or lights.

Ground truth committed under `data/mcp/` (CC BY-NC-SA, attributed).
The score bug is the alignment key: every clip's bug shows the score,
every MCP row carries the score at point start — read the bug, look up
the row, compare strings.

**Eval protocol, declared before running it:** constants frozen at
[006e6d5]-plus-this-commit; segmentation may be re-tuned for the new
reel's cuts (finding points isn't the hypothesis — charting them is).
Metrics per aligned point: server end, serve direction (4/5/6), rally
length, letter accuracy over COMMITTED letters only (coverage reported
beside accuracy, no silent cherry-picking), endings once coded. Dev-set
work (box quality, endings, point_58) continues separately; the test
matches stay untouched until the next freeze.
- Sourcing note: no full-match VOD of either test match on YouTube;
  extended highlights only for Nadal–Shapovalov. Fewer complete points
  than the dev reel's 60 — the price of labels.
- Session cost: $0.00 (ground truth is a curl away).

## 2026-07-09 — T1 setup: the homography does not transfer, and night fights back

**Footage + working copy.** Test 1 footage is the 8:21 TennisTV
highlights of Nadal–Shapovalov (720p60 → 30 fps working copy,
`clips/t1_nadal_shapo_30fps.mp4`; SAM cost scales with frames). First
probe frames confirm the adversarial setup: same court, same bug, but
floodlights and two lefties. The camera is also LOWER than the dev
reel's — the near runback is cut off at the frame edge, which will
matter later for near-court landings.

**The dev homography misses by ~20 px** on t1 wide frames — different
production day, remounted camera. Refit needed. Four attempts, three
instructive corpses:

1. **M1 recipe verbatim.** Under floodlights the blue→green paint
   boundary reads (V~155, S~55) — the same signature as grazing-angle
   real lines. The extreme-cluster fit grabbed the paint edge and
   produced a court 15-20 px too big.
2. **Tighter white (V>200, S<50).** Kept only the camera-facing lines
   (center, service, near baseline at V~253) and LOST the far baseline
   and sidelines entirely — the fit degenerated.
3. **Guided bands around the dev-H prior.** The far-baseline band
   locked onto ground-paint/banner whites 70 px off at far-left; a
   sideline band silently mixed doubles pixels into a "singles" line.
   Residuals looked fine (2.9 px!) while the center line sat 32 px off
   its actual pixels. **Plausible-but-wrong, caught only by rendering
   the reprojection and looking.**
4. **Four manual corners** off gridded 6× zooms of the median plate —
   one human input per match, the M0 ball-click precedent. Even this
   failed twice before it worked, and the failures were the lesson:
   the "corners" I read at the near baseline were the SINGLES corners
   (the doubles corners sit at the frame edges, x≈74/1141, buried in
   paint-edge blur). Two arithmetic cross-checks settled it: the
   far-corner midpoint (603) matches the measured center line (606)
   while my near reads gave 643; and px/m scale is linear in image y —
   48 px/m at the far baseline through 80 at the near service line
   extrapolates to ~100 at the near baseline, predicting a doubles
   span of ~1095 px… exactly the white band's raw extent.

Final fit: far DOUBLES corners + near SINGLES corners, tight-mask
lines as untouched validators — **service line 0.8 px, near baseline
1.4 px, center line 6.1 px.** The t1 chain now has its ground plane.
- Meta-lesson, third time now: masks and residuals can agree on a
  wrong answer; a rendered overlay cannot. Every geometric fit gets an
  eyeball artifact from here on.
- Session cost: $0.00.

## 2026-07-09 — T1 staging: 25 clips, and the answer key is wired in

**Segmentation transferred cleanly** — the one stage that fought
hardest on the dev reel needed only paths and the new homography: the
geometry probe's blue check holds at night (court H≈112 under lights),
and 25 court-view segments fell out of the 8:21 reel (3–23 s each,
~230 s of chartable play). No threshold touched.

**The score bug is the join key, and it worked.** One contact sheet
per clip (mid-clip frame + enlarged bug), scores transcribed by eye
into `data/mcp/t1_clip_alignment.csv`, joined to MCP rows by
(sets, games, points). One trap: **MCP's Pts column is server-first,
the bug is Nadal-first** — 17/25 matched before that transform, 23/25
after. The two leftovers are honest ambiguity, not failure: point_10
is a 40-AD that happened twice in the same game (two deuce cycles),
and point_16 is deuce in THE long game of set 3 — ten deuces, ten
candidate rows. Rally content will disambiguate both at eval time.
Bonus finds: points 21/22 both map to MCP Pt 210 (the reel replayed
the break point), and the marker-read for point_06 was wrong — the
MCP join corrected my eyeball (Shapovalov served that game).

The reel spans the whole match — first point of the night to the 6-6
tiebreak, ending on Nadal serving at 4-6: match point, the point that
made Shapovalov. 25 clips, 23 with ground-truth strings attached
(`t1_mcp_map.csv`). What remains before the scorecard: plates/players/
serve/ball stages on the t1 clips (pipeline frozen), then chart vs MCP.
- Session cost: $0.00 so far; t1 ball tracking will be the next spend
  (~25 clips × ~200 frames at 30 fps ≈ $1.6).

## 2026-07-09 — T1 run: the first scorecard against a human chart

**Free stages transferred with one new impostor.** Stabilization,
plates, players ran with paths changed and nothing else — far-player
coverage is actually BETTER at night (100% median vs 96%; floodlights
kill the hard shadows that made dev plates lie). But t1's ballkids
crouch at the NET POSTS, inside the far half, and the largest-component
player pick grabs them — so serve-toss windows opened above ballkids'
heads and one umpire chair. Fix on the record: a server stands at his
baseline, so the bootstrap now gates each frame's player box by foot
position in court meters. Impostor lesson #4.

**Serve detection: 11/25 confident (dev: 40/60), and the answer key
grades it.** MCP knows the server for every point, and changeover
parity (swap after odd games, every 6 tiebreak points) maps identity to
END once you know Nadal's starting end — which the data itself settles:
"Nadal starts far" explains 11 of 14 calls, the alternative 3. Serve-end
accuracy on confident calls: **11/14 (79%)** vs the dev frame-check's
6/8. Two of the three misses were already flagged low-margin.

**Ball tracking: eyeball review remains the spend-guard.** 25 dry-run
boxes, 13 approved (9 first pass + 3 after the ballkid gate + 1 mover
that found a real streak), 12 sent for $0.85, 11 usable. The one weak
accept I talked myself into (point_13, "ball blur at the racquet,
probably") died at 12 frames — the eyeball grade is still ground truth,
now 6-for-6 across both reels.

**The scorecard (11 points, frozen pipeline, vs MCP):**
    server end     7/11
    rally len ±1   7/11
    serve zone     1/2
    letters exact  2/11
    letters MIRROR 9/11
The headline number is the last one, and it is the best bad result on
the project: the right-handed assumption, written down as a liability
in M3 and left frozen on purpose, **mirrors 9 of 11 letters against two
left-handers.** A broken contact-side detector would flip a coin; a
mirrored one is CORRECT hardware wearing the wrong sign. One config
line per match (player handedness) turns 2/11 into 9/11 — next freeze.
Other honest reads: the serve-override vote that fixed point_59 on dev
WRONGED two calls here (thin far tracks make touch votes lie harder
than serve priors); rally length suffers because clips contain fault
serves the detector charts as play (MCP codes the fault, then the
played point — we window from the first confident serve and catch
both).
- Test-set discipline held: zero constants changed between dev and the
  scorecard. Every number above is out-of-sample.
- Session cost: $0.85. Project total: ~$3.65 of $9.

## 2026-07-09 — T1 freeze #2: one config line does what the mirror promised; the other fix wasn't needed

**Handedness is a per-match config now, and the mirror flipped on cue.**
`LEFTY = {"near": True, "far": True}` — one constant in the chart loop;
a lefty striker's f/b call inverts. Both t1 players are left-handed,
which is the easy case: both ends flip all match, so changeover
end-swaps don't matter. (A mixed-handedness match needs player identity
per END via changeover parity — written down as future work, not
attempted.) The same mechanism went into the dev-reel script with both
False; re-ran three dev clips and diffed the charts byte-for-byte
identical — a verified no-op for right-handers.

**Scorecard, before → after (same 11 points, same charts, same eval):**
    server end     7/11  →  7/11
    rally len ±1   7/11  →  7/11
    serve zone     1/2   →  1/2
    letters exact  2/11  →  9/11
    letters MIRROR 9/11  →  2/11
The mirror inverted exactly — and the two letters that were "right"
under the wrong-handed logic (point_01's f, point_04's shot 2) are now
the two misses. They were contact-side misreads all along, wearing two
sign errors that cancelled. No new information, just the correct
attribution.

**The fault-serve re-windowing fix died at the evidence stage: the
faults are not in the clips.** The plan from last entry — detect a
serve-like restart (second event cluster at the server's end after a
≥2s dead gap) and re-window past the fault. Four tracked clips have a
fault first serve in MCP (02, 06, 08, 14). Their clips run 4.5–6.8 s of
CONTIGUOUS court view, and a pro's between-serve routine runs 10–25 s —
during which the broadcast cuts away, and a broadcast cut is exactly
what ENDS a segment. Fault and played point cannot share a clip by
construction. The receipt, point_14 (worst over-count, 5 shots vs
MCP's 3): events run continuously f29→f176 (0.97 s→5.87 s at 30 fps),
max inter-event gap 33 frames (1.1 s) — no dead gap, no restart,
nothing to re-window. And point_06 carries a fault in MCP with rally
length EXACTLY right (3/3). The real rally-length failures are the old
enemies: thin tracks under-counting (03: 2/5; 08: 1/3 behind two
30-frame holes) and phantom cusps over-counting (14: 5/3). Last entry's
"clips contain fault serves the detector charts as play" was a story I
told myself without opening the event streams. Zero lines of windowing
code written; this paragraph is the fix.
- Freeze discipline held: this is freeze #2, and the handedness config
  is the ONLY pipeline diff between the two scorecards above. The
  align script also gained Gm/Pts context columns on matched rows
  (future disambiguation fuel) — data plumbing, 23/25 matches
  unchanged, zero effect on the eval.
- Session cost: $0.00. Project total: ~$3.65 of $9.

## 2026-07-09 — T2 staging: the near-transfer control mostly keeps its promise

**Footage was the first dead end.** The longest Federer–Haase upload
(9:44) turned out to be a Bandicam screen-capture with the broadcast
shrunk into a corner sub-window — court view technically present,
score bug clipped at the frame edge, useless. The 6:29 "highlights"
is five minutes of press conference. Test 2 runs on the 5:07 Sky
Sports world-feed cut (720p25 → 30 fps working copy): full-frame
broadcast angle, same court, same black-and-yellow bug, day session
as ordered. Half t1's footage; the control costs clips.

**The automated homography fit works in daylight — first try.** The
same M1 recipe that died three ways under floodlights (t1) fit this
plate clean: extreme clusters found both baselines and both doubles
sidelines, and every t1 cross-check agreed — far-corner midpoint 642.4
vs measured center line 641.6, px/m linear in y (predicted 65.1 at
mid-frame, measured 65.2). Held-out lines: center 0.2 px, near service
1.2 px, singles 3.9/5.7 px, far service ~3.5 px by eyeball (the
dist-transform said 19.5 px, but that's distance to the WRONG line —
the far service line is so washed out it never entered the loose white
mask at all; V~228 peak, S~128). Zero manual corners. So the t1 misery
was the night, not the method.

**Segmentation did NOT transfer untouched, and the failure was
self-inflicted.** Court probe round 1: 5 segments, and a run I could
SEE was court view scored 0.68 blue. The probe grid's center column
runs along the center service line and the x=1.5 columns sit 13 cm
off the singles lines — under day glare those lines bloom to V~250,
S~25, and the single-pixel reads count them as not-court. Sharper
irony: the homography was fit ON a court-view run, so the projected
probes land pixel-perfect on the white exactly when it matters. Night
exposure never blew the lines out; t1 passed by luck, not design. Fix:
interior probes read a dilated blue mask (on-a-line still counts),
apron probes keep the raw one. 11 segments, 90 s of play.

**Alignment: 10/11 unique, and the bug swapped teams.** This bug is
HAASE-top where t1's was Nadal-top — player 1 is the BOTTOM row now,
so transcription flips rows before the server-first Pts transform.
All ten matched rows agree with my marker reads (t1 had one eyeball
miss). The ambiguous one is a 40-AD that happened twice in the same
game — same trap as t1's point_10. Set structure derived from MCP
itself: set 1 ran 9 games (6-3), set 2 to 6-6 plus a 12-point
tiebreak. The reel's last clip IS the last MCP row — match point,
5-6 in the breaker.

**Players/serves: coverage record, then dissolves ate the toss gate.**
Bg-sub players: near 100% / far 99% median — best of the three reels
(day + deep plates = no ghosts). But this editor CROSSFADES between
cuts, so each clip's first ~5 frames diff as one giant blob (h 0.82 of
frame vs 0.25 for a real player) and the toss gate's argmax grabbed it:
serve_frame=0, "toss ratio" 4.2x. A real toss stretches ~1.1-1.3x —
blobs taller than 2x the series median are transitions, dropped before
peak-finding. After the fix: 6/11 confident serves, ratios 1.23-1.40.
Changeover parity (9-game set 1 prior from MCP): "Federer starts near"
explains 5 of 6 calls, the alternative 1 — serve-end accuracy 5/6
(83%), and the one miss was already the thinnest margin (0.18 m).

**Ball dry-run: 11/11 bootstrapped, 6 approved.** GOOD: 02/03 (toss
balls, crisp), 08/09/11 (clear flight streaks), 01 (real blur off the
racquet — the mover chain earned it). REJECT: 04 (box on a line
judge's leg), 05/07 (sign-edge glare at the net posts — the Sky feed's
Fly Emirates/FedEx panels flicker like t1's net signs), 06 (a dissolve
ghost at f4; the transition curse again). WEAK: 10 (a faint smudge ON
the singles line — t1's weak-accept died at 12 frames, not paying that
tax twice). Nothing sent. The 6 GOODs are 1,566 frames ≈ $0.49 when
the orchestrator signs off.
- Two thresholds changed this session (probe dilation, toss blob cap),
  both at STAGING stages, both forced by transfer failures the dev
  reel could not have shown. The chart loop itself stays frozen.
- Session cost: $0.00. Pending t2 ball spend if approved: ~$0.49.

## 2026-07-09 — Endings v1, and the letter metric confesses

**Every string used to end in '?'. Now it ends in evidence.** Ending
v1, observable signals only: the last shot's far-half landing codes
out-deep/-wide (`d@`/`w@`/`x@`), in-court-and-nothing-came-back codes a
winner (`*`), and a ball track that dies within 1.5 m of the net inside
1.2 s of the last hit codes a net error (`n@`). Winner-vs-forced-vs-
unforced is charter judgment the pipeline does not attempt — `@` means
"error, attribution not judged" and eval compares TYPE only. First
numbers on t1: **6/11 committed, 3/6 correct** — and all three misses
trace to rally-length errors upstream (a phantom last shot with an
in-court landing reads as a winner; a thin track calls a serve out-deep
on a point that continued).

**The letter metric had been grading on a curve.** Frame-checking the
two "contact-side misses" (t1 points 01, 04) showed both are INDEX
misalignment — our chart is a shot short or long, so letter k compares
against the wrong MCP stroke. Both charts also looked plausibly right
on the frames. The eval now reports letters two ways: all-index
(9/11, unreliable) and **length-matched clips only (1/1)** — a smaller
and more honest number. Verify strips for t1 exist now
(t1_verify_chart.py).

**Both findings point the same direction: rally length is the
bottleneck.** Thin tracks under-count (points 03, 08, 15), phantom
cusps over-count (04, 07, 14), and every downstream metric — letters,
endings, strings — inherits the error. That is the next real fix, and
it is a detector problem, not a charting problem.
- Dev strings refreshed with endings for consistency (point_59 closes
  `w@`, matching its known out-wide final landing).
- Session cost: $0.00.

## 2026-07-09 — T2 run: the control catches the ball tracker

Charted the 6 tracked control points (frozen loop, LEFTY both-false —
day session, both right-handed) and scored them:

    server end       2/5      (t1: 7/11)
    rally len ±1     1/5      (t1: 7/11)
    serve zone       1/2
    letters (aligned) 2/2     (t1: 1/1)
    ending type      0/3      (t1: 3/6)

The control was supposed to be the EASY match — same court, daylight
like the dev reel, no handedness trap. It scored worse, and the reason
is upstream of everything the chart loop does: **SAM keeps losing the
ball in daylight.** Drop rates on the six tracks: 45%, 0%, 2%, 44%,
32%, 36% — t1's night tracks dropped far less. Night was never the
hard test for tracking: a floodlit ball on a dark background is the
easiest object in the frame. Day glare + white lines + long Federer
rallies is the hard test, and rally length collapses with the track
(6/12 and 6/9 on the two long points), taking letters, endings, and
even server end (the override votes get bolder as tracks thin) down
with it.

What survived: **letters on length-matched clips are 2/2 with
right-handed logic** — combined with t1's mirror, the contact-side
hardware now has evidence on both handednesses. And the one clip with
a clean track (point_08, 4/4 shots) matched MCP shot-for-shot on
strikers and both committed letters.

Scorecard-driven priority, now confirmed by two test matches from
opposite directions: **ball-track continuity is the bottleneck** —
holes and drops, not charting logic. Candidate next moves: multi-prompt
SAM (re-prompt after holes), or a cheap diff-based ball recovery in the
gaps using the tracked segments as anchors.
- Session cost: $0.49 (6 t2 tracks). Project total: ~$4.15 of $9.

## 2026-07-09 — WASB vs SAM: the $0 specialist takes ball duty

The landscape doc said no published SAM-vs-specialist comparison on
small fast balls exists; this entry is ours. WASB-SBDT (BMVC 2023, NTT,
MIT) with the pretrained tennis weights — 6 MB, per-frame heatmap, no
prompt, no bootstrap, no fal spend. Their Detector class hard-asserts
CUDA and everything routes through hydra, so the model definition and
affine code are imported from their src and the driver loop (3-frame
windows at 512x288, ImageNet norm, sigmoid + 0.5 threshold + weighted
blob centroid, their online tracker's 300 px gate) is ~200 lines of
ours (experiments/wasb_track_ball.py). Runs on MPS at roughly 15-20
fps. Sanity check first: three rendered frames on t1_point_01, circle
dead on the ball in all three — including a motion-blur streak coming
off the racquet, exactly the shape SAM's box prompt kept refusing.

**Tracking: WASB wins every case that matters and needs no permission
slip.** Mean coverage on the 11 A/B t1 clips 87% vs SAM's 82%, and on
the t2 day clips 88% vs 74%. The distribution is the story: SAM's
disasters evaporate (t1_point_25: 55%→93%; t1_point_08: 50%→76%;
t2_point_01: 55%→87%; t2_point_09: 68%→94%), while SAM keeps a 3-7 pt
edge only on the easy floodlit-night clips it was always good at. Max
holes shrink to match (t2_point_09: 40 frames → 4). And the 13 t1
clips SAM never touched because no toss/mover bootstrap was trusted?
WASB tracked all of them, 73-97% coverage, zero interaction.

**A/B through the frozen loop, same clips, zero constants touched:**
    t1 (11 clips)      SAM      WASB        t2 (5 scored)  SAM    WASB
    server end         7/11     7/11        server end     2/5    2/5
    rally len ±1       7/11     5/11        rally len ±1   1/5    4/5
    serve zone         1/2      2/2         serve zone     1/2    1/1
    letters (all)      9/11     13/16       letters (all)  6/13   16/19
    letters (aligned)  1/1      0/0         letters (algn) 2/2    11/14
    ending type        3/6      1/5         ending type    0/3    2/3
The t2 column is the verdict. The control match — the one that caught
SAM starving in day glare — flips from worst scorecard to best:
t2_point_01, the 12-shot Federer rally SAM charted as 6, comes out
12/12 with all 7 committed letters correct. Rally length 1/5 → 4/5 on
identical clips, identical chart logic. The only thing that changed is
the ball track.

**The t1 regression is real and reported: rally length 7/11 → 5/11,
endings 3/6 → 1/5.** Direction matters: SAM under-counted (thin
tracks hide hits), WASB over-counts (03: 8/5, 06: 6/3, 25: 12/10). A
denser track feeds the cusp detector MORE — pre-serve ball handling,
toss, bounces the thin track never saw — and the frozen thresholds
were battle-hardened against sparse SAM tracks, never against 90%
coverage. That's a chart-loop problem to fix in the open, not a
tracker problem; the same dense tracks scored 4/5 on t2.

**New coverage, 13 clips SAM never tracked (11 scored, 2 ambiguous-MCP
auto-skipped):** rally len ±1 6/11, letters 20/31, endings 2/3, server
end 1/11 — that last number is honest and explains itself: these are
exactly the clips where the serve detector had no confident call
(which is WHY SAM had no bootstrap), so the chart has no serve anchor
and no synth serve. Different failure, same root as always.

SAM retires from ball duty. Marginal tracking cost per match: $0.00,
forever, and the 25-clip fleet takes minutes on a laptop GPU. SAM
remains a candidate for player segmentation, where a box prompt is
cheap and the object is large.
- experiments/t1w_chart_point.py / t2w_chart_point.py / t1w_eval.py /
  t2w_eval.py are byte-identical twins of the frozen scripts except
  ball/ → ball_wasb/ and charts/ → charts_wasb/ — the A/B is
  clip-for-clip comparable and the frozen originals are untouched.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — Freeze #3: the track outlives the point

**The phantom events have a name: the dead-ball coda.** WASB's t1
regression (rally ±1 7/11 → 5/11) was all over-counts, no under-counts.
Dumping the event streams for the worst offenders (03: 8/5, 04: 5/3,
06: 6/3, 25: 12/10) and frame-checking the extra "hits" settled it in
pixels: **the point ends but the WASB track doesn't.** SAM's sparse
track died with the rally — an accidental end-of-point detector we
never knew we were relying on. WASB never loses the ball, so it follows
it into the aftermath: 03 f243 is the dead ball bouncing in the bottom
corner while both players stand around; 25 f412 is match point, ball at
the frame's bottom edge, crowd mid-standing-ovation; 06 f141 is the
ball drifting through the CROWD above the back fence. The frozen cusp
detector charts all of it as rally. The control that proves the story:
point_19 (5/6, correct) has a track that quits when the point does —
its event stream is clean to the last frame.

**One gate kills the coda: phantoms don't live on the court.** Every
frame-checked phantom sits outside the playable envelope in court-y —
≥ 26.4 m (behind the near baseline, dead ball in the foreground) or
≤ −18 m (a ball in the crowd is nowhere near the ground plane the
homography assumes). Real rally hits across BOTH trees span −8.3 to
25.5 m. The magnitude route died first, for the record: coda bounces
swing 109-210 px/f against real hits' 8-33 — a beautiful gap, until
t2's serves (swings 51, 86, 152) landed inside it. High tosses swing
past any cap, and the same ground-plane distortion maps 04's real far
serve to cy −20.3. So the gate is positional with a serve-window
exemption: cusps outside [−12, L_C + 2.2] are dropped unless within 14
frames of the gated serve call. Three new constants in the *w_ twins,
every frozen-era constant untouched, frozen scripts untouched.

**Before/after, same clips, same tracks:**

    t1 (11 A/B)       SAM    WASB   +gate      t2 (5 scored) SAM   WASB  +gate
    server end        7/11   7/11   8/11       server end    2/5   2/5   2/5
    rally len ±1      7/11   5/11   10/11      rally len ±1  1/5   4/5   4/5
    serve zone        1/2    2/2    2/2        serve zone    1/2   1/1   1/1
    letters (all)     9/11   13/16  13/15      letters (all) 6/13  16/19 16/19
    letters (aligned) 1/1    0/0    2/2        letters (algn)2/2   11/14 11/14
    ending type       3/6    1/5    3/7        ending type   0/3   2/3   2/3

t2 is byte-identical before and after — the gate never fires on a rally
that stays on the court, which is exactly the claim it makes. t1's one
remaining rally miss is point_08 (1/3): an UNDER-count from thin far-end
events, the old failure, not this one. Server end improved as a side
effect (03's coda had been outvoting the serve call), and endings went
1/5 → 3/7 — the two new misses are honest commits on charts that are
still a shot off. Receipts in outputs/t1/charts_wasb/: the
fixed_t1_point_{03_f243,03_f252,06_f141,25_f412}.png strips show each
former phantom with the caption it earned, verify3_t1_point_03_* shows
the repaired 5-shot chart. Residual on the record: 25 keeps one coda
event at f439 (cy 6.67, inside the envelope — a track jump, not a
position outlier) and rides on ±1.

**New-coverage t1 (11 scored clips SAM never tracked): rally 6/11,
letters 20/31, endings 3/4, server end 1/11** — endings up from 2/3,
everything else unchanged, as it should be: those clips fail by
UNDER-counting (no serve anchor, thin far-end cusps), and a gate that
only removes events can't help them. Different failure, honestly
untouched.

**Dev reel migrated to WASB: all 60 clips tracked, $0.**
wasb_track_ball.py grows --tree m3 (dev clips live in clips/points/,
no suffix) and ran the whole reel on the laptop GPU: the 23 SAM-tracked
points (21 of them charted) AND the 37 bootstrap rejects SAM never got
to touch. Coverage mean 72% (min 25, max 94; 34 clips ≥ 70%, three
under 50%: points 02, 10, 37) — well below t1/t2's 87-88%, so the dev
footage is harder for the specialist too, just not fatally. No
re-charting (dev has no ground truth); the tracks are on disk for
whenever the charting loop next wants them.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — T3/T4 staging: clay and grass vote against every Montreal assumption

Two more MCP-charted matches, chosen to break things t1/t2 never
stressed: **t3 = Roland Garros 2023 final, Djokovic–Ruud** (clay, RG
world feed, official 34-min extended highlights, 720p50) and **t4 =
Wimbledon 2024 final, Krejcikova–Paolini** (grass, WTA, Wimbledon feed,
official 12-min extended highlights, 720p25). Six candidates were
cross-referenced against the MCP match lists and yt-dlp before
committing; the 2023 Wimbledon QF with the best footage turned out not
to be charted, the 2025 final was a 44-minute blowout with a 7-minute
reel, and Nadal–Djokovic footage is gorgeous but mixed-handed — the
chart twin's single LEFTY dict can't represent a match where the flip
depends on which player holds each end. All four chosen players are
right-handed (looked up, not assumed — the MCP hand column claims
Vondrousova is right-handed, so it doesn't get to be the source).

**Clay homography: the M1 recipe died three new ways before it fit.**
(1) "Blue court hull" has no blue, and the obvious clay-hue hull
swallowed the whole frame — sunlit CROWD SKIN is H 6–8 too. Tighter
band, largest component, filled contour instead of convex hull. (2)
The unstabilized median plate SMEARS lines into the clay: this camera
pans ~16 px over the fit run, and dusted white-on-orange doesn't
survive what white-on-blue did. ECC-stabilize the plate frames first.
(3) The far baseline is BURIED — V~200 line on V~195 clay, no mask
finds it, and no fit can use it. Lines come from a tophat (thin-bright)
filter now, verticals are labeled by symmetry about the center-service
cluster (the Perrier crates and the Haier box put 20 Hough segments
inside the clay hull; junk is asymmetric, court lines aren't), and the
fit basis swaps the far baseline for the far service line. Fit checks:
singles sidelines 0.4/0.5 px held out, center 0.2 px, near service
6.3 px — a genuine lens bow the 15-intersection LS refit spreads but
cannot remove.

**Grass exposed the scorer, which is the useful kind of failure.** The
first far-line chooser scored candidates on the singles sidelines —
which are nearly INSENSITIVE to the horizontal labels (they
interpolate the doubles lines in x). On t4 it picked the net band. The
second attempt scored the LS fit's own rms — and a wrong-but-self-
consistent triple ([175, 368, 473] as baseline/service/baseline) fit
its own intersections to rms 1.7 px. Truth needed a different
question: score every candidate assignment by how well ALL NINE model
lines land on the observed line mask. The true assignment explains all
of them; impostors strand the lines they didn't use. Winner: mask
score 1.9 px, all line residuals <= 4.4 px mean — with the far
baseline IN the fit on grass, because Wimbledon's wear pattern spares
the lines and eats the grass around them (worn baseline dirt reads
H 15–19, nearly clay; the hull band includes it on purpose).

**The apron test is dead on both surfaces, long live the line probes.**
t1/t2 court-view detection was interior-blue AND apron-not-blue. On
clay the apron IS clay; on grass the run-off IS grass. Replacement:
interior probes read the court color, and LINE probes read the tophat
mask along the projected model lines — geometry a close-up can't fake.
Then the pan problem: a fixed projection walks the line probes off the
real lines (known court view read line_hit 0.29) and the whole 34-min
reel yielded 68 s of play. The line read is now a max over a small
shift grid (±24/±16 px), winning shift recorded per frame: 68 s became
542 s in 68 segments. t4: 48 segments, 602 s of a 733-s reel. Each
clip also gets a camera offset (median winning shift) that the serve
and chart twins subtract — the fit camera and the clip camera are not
the same camera on these broadcasters.

**Alignment: 59/68 and 36/48 unique.** Same deuce-recurrence traps as
t1/t2 (six t3 clips are 40-AD states that happened twice in a game),
plus false-positive court views with no bug (a net close-up, a
dissolve, two crowd pans). The RG bug is Djokovic-top, the Wimbledon
bug Krejcikova-top — both player-2-top, both transcribed player-1-first
before the server-first flip. The Wimbledon bug's green sets column
read 1–1 in the third set, which is how the staging discovered this
was the real 6-2 2-6 6-4 final and not the straight-setter memory
insisted it was. Derive the structure from the data.

**Players: the net tape ghosted the far player, twice.** RG jitter
makes the high-contrast tape diff against the plate every frame,
under the 0.90 static-erase, and the tape blob out-areas a motionless
far player: "far player" at court y 10.8 was a box ON the net,
frame-checked. First fix cut the far region at court y 11.0 and
missed: the tape hangs ~1.07 m above the ground plane, so its
back-projection lands at court y ~7–11.5, metres behind the net line.
Second cut at 6.5 killed it (far coverage median 100%→80%, the honest
price). And clay servers stand WIDE — Ruud serves from x = 1.4 m,
4.1 m off the center mark, so t2's 2.0 m center gate rejected real
serves all day; the t3 twin runs 4.3 m and lets the toss and baseline
gates carry the discrimination. Serves: t3 14/68 (most clips are
mid-rally fragments where no-serve is the CORRECT answer), t4 36/48.
Changeover parity: t4 decisive — Paolini-starts-far explains 21/25
(84%); t3 votes far 9/14 (64%), corroborated by pixels: Djokovic's red
shirt is the NEAR player while the bug shows Ruud serving game 1.

**WASB coverage: clay 82.7%, grass 72.4% — grass is the specialist's
hardest surface yet.** White ball, white lines, white kit; t1/t2 ran
87–88%. All 116 clips tracked, $0, no bootstrap, no permission slips.

**Scorecards (frozen loop + envelope gate, LEFTY all-false):**

    t3 (59 aligned)                    t4 (36 aligned)
    server end        10/59            server end        10/36
    rally len ±1      20/59            rally len ±1      15/36
    serve zone         1/6             serve zone         1/4
    letters (all)     46/75            letters (all)     70/127
    letters (aligned) 16/25            letters (aligned) 11/22
    ending type       11/34            ending type        1/12

**The headline breakage is upstream of every metric: segment ≠ point,
and each broadcaster breaks it in the opposite direction.** t3 charts
3.9 shots against MCP's 8.9 (39/59 clips undercount by >1) — the RG
editor and camera fragment long clay rallies across multiple segments,
and four consecutive clips carrying the SAME score (30-15, game 5) are
one rally in four pieces, some of them main-camera slow-mo replays
that pass every geometric probe AND keep the score bug. t4 charts 8.1
against 7.2 with 15/36 OVER-counting — Wimbledon never cuts between
points, so a 15-second segment holds two or three points and the chart
strings them into one 12-shot monster rally. Montreal's tidy
cut-per-point editing was doing segmentation work we never noticed and
never paid for. That's the next real fix: point boundaries need their
own detector (serve-anchored splitting, replay/slow-mo rejection),
because the reel's edit grammar can't be trusted to provide them.
- New per-match twins: t3/t4 fit_homography (clay/grass recipes),
  court_probe (line probes + shift search), extract_points (+ clip
  offsets), align_mcp, bgsub_players (net-tape exclusion),
  serve_detect (wide-stance gate), t3w/t4w chart + eval.
  wasb_track_ball grows --tree t3/t4. Frozen t1/t2/m3 scripts and all
  frozen-era constants untouched.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — Point boundaries: the score bug is the point ID

**The staging entry ended with "point boundaries need their own
detector." The detector was on screen the whole time.** The score bug
is a screen-space overlay — camera pan never moves it — static within
a point, and it changes exactly at point boundaries. So the bug region
IS the point ID: merge adjacent segments whose bug never changes (same
point, camera flaked), split segments where it changes mid-segment
(two points, editor never cut), and read the plateau timeline as the
point sequence. Validated on pixels before a line of the pass was
written: the four "30-15 dup" clips (old segs 12-15) share ONE
unbroken digit plateau across all four segments AND their gaps — a
21-second, 16-shot rally the probe fragmented because the camera
zooms mid-rally and the line probes drop. Same story for the 29-shot
rally (17+18) and 30+31+32. And the three "40-AD dup" clips are NOT
replays: a 40-40 plateau sits between them in the gaps. They're three
different deuce-cycle points, and the plateau ORDER carries exactly
the information the value alone cannot.

**The identity metric had to be a changed-pixel fraction, not a mean
diff.** Same-point crops read <= 0.005 changed pixels
(brightness-normalized, |diff| > 45), different-score crops >= 0.033 —
a 6x margin. The mean-diff step between segs 17/18 that looked like a
score change was crowd bleeding through the crop's right edge; the
crops, rendered and read, both say "2-4 / 40-30".

**Four traps between the idea and a working pass, all found by
looking:** (1) The RG bug GROWS a column per completed set — games and
points slide right ~27 px at each set boundary, and a fixed crop
spends sets 2-3 staring at the frozen "7 6 / 6 3" columns (segs 48/49
and 64/65 read as single plateaus until the era windows went in). (2)
The set-1 tiebreak banner pushes the whole bug down ~14 px and it
drifts — the digit crop now follows the presence template's per-frame
winning dy. (3) A plateau ref frozen on a fade-in transition frame
sits at threshold distance from its own settled plateau and lets
compression noise fake a change 100 frames later (seg 4 split into
two identical-scored halves). But the opposite fix — a rolling ref —
glides straight through the RG update animation, which spreads a
score change over ~12 frames of ~0.01 steps and merged four deuce
points into one. The working design: frozen ref per plateau, anchored
on the median of a settling window. (4) The first replay detector
flagged 15 live t4 segments — motion magnitude is framing, not speed;
a quiet wide shot reads exactly like slow motion until you look at
WHICH frames are duplicates.

**The replay mechanism is REFUTED, and that's the finding.** Every
dup-score group turned out to be fragments or distinct deuce points.
The one cadence suspect (t3 seg 62: isolated duplicate frames at the
1-in-6 rate a 2x slow-mo of a 50fps source would leave) was acquitted
by physics: image-space ball gravity from the WASB tracks is unimodal
across all 68 old t3 clips (30th-pct |d2y/dt2| 0.7-2.3 px/f^2, seg 62
at 1.78, upper half) and all 48 t4 clips. A 2x slow-mo clip would sit
~4x low. There is no probe-passing slow-mo replay in either reel —
the staging entry's "main-camera slow-mo replays" claim was wrong.
point_boundary.py keeps dup_frac bookkeeping and a dead-air drop
(median motion < 0.15; live pieces never dip under 0.26) for the day
one shows up; neither fired in the final runs.

**New segments: t3 68 -> 59 points (4 merges, 2 no-bug drops), t4
48 -> 49 (18 split pieces, 1 merge, 21 boundary stubs dropped).**
Clips re-extracted, bugs re-transcribed by eye from fresh contact
sheets — and the transcription pass caught four errors in the staging
alignment: old t3 segs 22 and 41 ("no bug: close-up/dissolve false
positive") and old t4 segs 31 and 47 ("no bug: crowd shot") are all
real court-view points with the bug up — the old mid-clip sheet frame
just caught a bad moment. Old t3 seg 60 was misread 0-0; it's 30-0.
The t4 "3rd DEUCE" banner on the recovered seg-47 point confirms the
deuce ordering the plateaus imply.

**Alignment: 59/59 and 49/49 unique, from 59/68 and 36/48.** True
boundaries plus a new order pass in the align twins: the reel is
chronological, so an ambiguous clip's MCP candidates are bounded by
its resolved neighbors' Pt numbers — 6 t3 and 10 t4 deuce-recurrence
ambiguities resolved, matched-Pt sequence verified monotonic. The
score bug can't tell the first 40-AD from the second, but the
timeline can.

**Scorecards, same frozen chart loop, before -> after:**

    t3 (clay)          before (59)  after (58*)     t4 (grass)     before (36)  after (49)
    mean shots o/MCP   3.9 / 8.9    4.6 / 8.1       mean shots     8.1 / 7.2    7.7 / 6.7
    server end         10/59        7/58            server end     10/36        17/49
    rally len ±1       20/59        24/58           rally len ±1   15/36        21/49
    serve zone         1/6          1/4             serve zone     1/4          8/10
    letters (all)      46/75        58/100          letters (all)  70/127       80/153
    letters (aligned)  16/25        22/32           letters (algn) 11/22        10/18
    ending type        11/34        9/30            ending type    1/12         5/19
    (*t3_point_06: track too thin, not charted)

Where the merges had footage to work with, they paid: the 16-shot
rally charts 12/16 as one clip (its four fragments charted 1-3 each),
and t3_point_24 charts 12/12 exact. t3 rally ±1 34% -> 41%, letters
(aligned) 64% -> 69%. t4 serve zone 1/4 -> 8/10 and endings 8% -> 26%
— splits put the serve at the clip start where the gates can see it.
Honest losses on the same table: t3 server end fell 10/59 -> 7/58
(merged clips start mid-rally, so no serve anchor and alternation has
to vote alone), t3 endings 11/34 -> 9/30, t4 letters (all) 55% ->
52%, letters (aligned) 50% -> 56% on a smaller base.

**What still fails, named:** (1) t3's mean-shots gap (4.6 vs 8.1)
is mostly NOT recoverable by segmentation — the RG editor cuts INTO
long rallies, so the first shots were never broadcast: point_01
charts 2/13 and the 29-shot rally 8/29 with the film starting
mid-rally. The bug proves those clips are single points; it cannot
conjure footage. (2) t4 still over-counts 19/49 clips by >1 — but
these are now single-point clips per the plateau timeline, so the
excess is chart-level: the Wimbledon feed's long dead time lives
INSIDE the point's plateau, the dead ball stays inside the court-y
envelope on grass, and the freeze-#3 gate can't touch it. That's the
t1 dead-ball coda in a new costume, and it's the next chart-level
fix, not a segmentation one. (3) t4_point_23 charts 21/5: the bug
holds 40-40 through 28 unbroken seconds, which pixel-reads as one
point — likely a challenge/replayed let (the score really doesn't
change), and score identity is blind to it by construction.
- New shared pass: point_boundary.py (bug scan cached per reel,
  plateau labeling, merge/split/no-bug/dead-air, receipts in
  outputs/*/bug_checks/ + bug_timeline.png). Extract twins prefer
  segments_v2.csv; align twins grow the order pass. WASB re-tracked
  all 108 new clips (t3 82%, t4 72% coverage), $0. Frozen t1/t2/m3
  trees, scripts, and every chart constant untouched.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — Serve detection: the ball adjudicates on clay, the window slides on grass

**The 4-match benchmark's weakest stage, re-baselined post-boundary-fix
and then read from the tracks: three failure modes, none of them the
gates' constants.** The t3 detector committed 10/59 calls (7 right);
t4 committed 37/49 and was right on 33 — and then the chart's striker
vote overrode 16 of those calls and got EVERY ONE wrong (server end
7/58 and 17/49). Named from pixels and track scans:

1. DEAD-AIR STANCE (both feeds). Point-boundary clips start at the
   score-bug plateau, so the first second is the between-points
   shuffle. The t3 near player idles at court y 28.2-28.9 — metres
   BEHIND the baseline gate — in ~24 clips and steps up 1-3 s in; on
   t4, 10 of the 12 no-call clips form a legal stance 0.5-9.6 s in,
   where a window nailed to frame 0 (and a toss window capped at 4 s)
   never looks.
2. THE TAPE GHOST OWNS THE FAR TRACK (t3). In ~20 clips the far
   "player" is a 100%-coverage blob frozen at court y 6-8 that never
   visits a baseline: the net-tape ghost out-areas the motionless far
   server. No gate can be tuned around a player the tracker never saw.
3. TOUCH VOTES OUTVOTE THE SERVE (t4). W_TOUCH 3 vs W_SERVE 2, and
   the white-on-white boxes feeding the touch votes are junk exactly
   when it matters. The override that point_53 justified went 0/16
   here.

**The ball adjudicates — measured before wired.** The serve is the
WASB track's first sustained net crossing: a monotone court-y run
spanning > 4 m inside 0.6 s. Its direction gives the server end, its
start frame the serve (the run opens at the toss/contact, which
projects far beyond the server's baseline — the same ground-plane
distortion freeze #3 exempts). Scored against parity truth BEFORE any
wiring: t3 48/56 (86%); t4 26/49 — a coin flip, REFUTED on grass
(white ball, white lines, 72% coverage, dead-air ball-kid throws).
So t3's v3 leads with the ball and t4's keeps the players; the
surfaces picked their own signals. t3's 8 ball misses are mostly
clips whose footage starts mid-rally — no launch-shape gate separated
them (cy0 and pre-launch speed both overlap), so the call commits and
the miss rate is the honest price. Deuce/ad still needs a stance:
readable on 22/56 (the ghost blinds the rest, which refuse a side and
the chart degrades their zone to '?').

**t4 v3: same gates, the window slides.** Each side's first 1-s window
passing coverage+center+baseline, toss wanted within 4 s of SETTLING,
ties to the side that settled first — the server's stance forms while
the receiver is still wandering, 21/25 dual-candidate clips against
truth vs 13/25 for v2's toward-the-center-mark rule — and simultaneous
settles to the stronger toss (3/4). 46 commits, 38 right.

**And the chart now trusts the detector it measured to be better:** a
confident serve call LOCKS the alternation chain's first striker (flip
slots stay in play); the serve-vs-touch vote market stays for
everything else. Both changes in the t3w/t4w twins only.

**Scorecards, same frozen chart loop, before -> after:**

    t3 (clay)          before (58)  after (59*)     t4 (grass)     before (49)  after (49)
    server end         7/58         48/59           server end     17/49        38/49
    rally len ±1       24/58        25/59           rally len ±1   21/49        21/49
    serve zone         1/4          9/18            serve zone     8/10         16/27
    letters (all)      58/100       70/105          letters (all)  80/153       69/129
    letters (aligned)  22/32        19/24           letters (algn) 10/18        7/16
    ending type        9/30         9/30            ending type    5/19         5/19
    (*t3_point_06 charts again: the synth serve gives its thin track a shot)

Server end 12% -> 81% and 35% -> 78%, rally intact. The regressions,
on the record: t4 serve zone's RATE fell (80% -> 59%) while its count
doubled — the 8 wrong-end commits carry wrong sides and their zones
pay; t4 letters (aligned) 10/18 -> 7/16 on a shrinking base — locking
chains to truer servers reshuffled which coin-flip box letters get
counted, and both numbers are the same coin. Receipts:
serve3_t{3,4}_point_*.png strips in charts_wasb/ — point_29's frames
show Ruud mid-toss at the called f14-19 with the bug's serve marker
on RUUD; point_19's shows the far server tossing at 8.5 s, deep in
dead air the old window never reached.

**What still fails, named:** (1) rally length barely moved (t3
undercount is missing footage, t4 overcount is the dead-ball coda
living inside the plateau — the chart-level fix the boundary entry
already queued). (2) endings byte-identical, same reason. (3) t3
deuce/ad is blind wherever the tape ghost is the far track — the
bgsub far-half cut, not the serve detector, owns that fix.
- t3_serve_detect v3 (ball launch + sliding settle + stance-tol
  table), t4_serve_detect v3 (sliding settle + settle-first
  tie-break, ball refutation on the record), t3w/t4w chart twins grow
  the serve lock + no-side zone guard. serves.csv gains src/launch_cy
  columns. Frozen t1/t2/m3 trees, scripts, and constants untouched.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — The dead-ball coda: the point's spine is the net-crossing sequence

**The queued chart-level fix, taken from pixels first.** Six
over-counting t4 clips rendered at their trailing events
(coda_t4_point_*.png in charts_wasb/) tell one story: after the true
final shot the ball KEEPS GOING — bouncing at the near player's feet
while she collects it (48, an ace charted as six shots), drifting past
the far baseline to the kids (28, 01, 24), circling the net after a
net death (05), swatted back across between points (45) — and both
players break rally posture and walk. Wimbledon never cuts, so all of
it lives inside the point's own score-bug plateau and inside freeze
#3's playable envelope. Named mechanism: the in-plateau dead-ball coda.

**Everything per-event was measured and refuted before the rule was
chosen, and the refutations are the finding.** Landing positions:
LIVE rally shots read 5-45 m deep out (the collapse position is
projection garbage on a rising ball) — an out-landing trigger
truncated 15 correct clips. Player walking speed: the coda boxes go
rogue and read FASTER than live walking (coda p90 up to 26 px/f vs
live 5-8). Post-anchor track coverage: 0.74-0.94 both classes.
Static-lock runs: 11 frames both classes. point_28's true coda and
point_20's blind-track live rally are byte-similar in every feature
we can compute. There is no per-event tell.

**What holds is sequence-level tennis: a live rally sends the ball
across the net every shot.** The track's net crossings — smoothed
court-y monotone runs, >= 4 samples, span 5-40 m (real deep-lob
flights project to 32 m; track teleports onto the crowd run 43-53 m,
and the span cap is what separates them), 4-90 m/s, no 6 m/frame
step — are the point's spine. Two truncation rules in the t4 twin:
(1) DEAD-GAP: a shot followed by > 3.0 s with no crossing inside the
gap ended the point (longest live inter-shot gap on the reel: 2.5 s;
codas idle 3.1+). Not applied at a synth serve — that gap is the
between-points shuffle the serve entry already charted. (2) ANCHOR:
shots more than 0.6 s after the LAST crossing's start didn't launch
it; the first within 1.2 s of its end is kept (the receiving shot — a
net error never crosses); the rest are coda. 24 of 49 clips truncate,
53 phantom shots drop.

**t4-only, by measurement:** the same pass on t3 scores 25/59 ->
23/59 — clay's failure mode is undercounting (the RG editor cuts into
rallies), and the anchor charges a crossings-recall tax t3's
fragmented footage can't pay. The t3 twin doesn't grow it.

**Endings, part two: re-derived free, then extended to the near half.**
With the coda gone the "last shot" is the true last shot and its
landing is the real final bounce: t4 endings 5/19 -> 10/31 without
touching the ending code. Then the investigation the boundary entry
queued: near-half landings are invisible to the collapse detector by
construction, but the dense WASB track holds the bounce as an image-y
V-cusp below the net line — and AT the cusp the ball is ON the ground
plane, so the projection is honest exactly there. The trap is the
second bounce: winners' late cusps read cy 25.6-30 (freeze #3's own
boundary — the dead ball at the collector's feet starts at 26.4) and
miscode deep. The true first bounce arrives within flight time, so
the fill searches 1.2 s and no further: at 2.0 s it commits 6 and
misses 3; at 1.2 s it commits 3 and misses 0 (t4_point_23 w@ at
cy 14.6 x -3.4, t4_point_24 d@ at cy 26.6, t3_point_08 d@ at 26.2 —
all against MCP truth). Fill-only: it never overrides a far-half
landing or a net death. Also refuted on the way: re-reading far-half
landings at the cusp frame instead of the post-bounce median made
endings WORSE (10/31 -> 7/29) — winners' far-half bounces read -2.7
to -38.6 m deep at either sample point; far-half in/out is not
recoverable from this homography, which is why ending v1 was 5/19.

**Scorecards, before -> after:**

    t3 (clay)          before (59)  after (59)      t4 (grass)     before (49)  after (49)
    server end         48/59        48/59           server end     38/49        38/49
    rally len ±1       25/59        25/59           rally len ±1   21/49        26/49
    serve zone         9/18         9/18            serve zone     16/27        16/27
    letters (all)      70/105       70/105          letters (all)  69/129       67/123
    letters (aligned)  19/24        19/24           letters (algn) 7/16         16/32
    ending type        9/30         10/31           ending type    5/19         12/33

t4 over-counting >1 fell 17 clips -> 9, rally ±1 43% -> 53%, letters
(aligned) doubled its base at a higher rate (44% -> 50%), endings 26%
-> 36% on nearly double the commits. The regressions, on the record:
t4 points 12, 18, 20 were rally-correct and now aren't — long rallies
whose late track wanders the far run-off with no crossings, which is
exactly what point_28's true coda looks like; three separators were
tried and all three refuted, so the pass eats them and says so.
letters (all) lost two matches with the coda letters it deleted
(69/129 -> 67/123 — a higher rate on an honest base). t4_point_23
still charts long (19 shots against 5): the 28-second 40-40 plateau is
one pixel-point holding a replayed let, and score identity stays blind
to it by construction.
- t4w_chart_point grows net_crossings + truncate_coda (constants
  documented in-file: DEAD_GAP_S 3.0, LAUNCH_SLACK_S 0.6, RECV_S 1.2,
  CROSS_* gates) and the near-half ending fill; t3w_chart_point grows
  the near-half fill only (NEAR_BOUNCE_WIN_S 1.2, NEAR_DEEP_M 0.5,
  NEAR_CY_CEIL 8.0). match_chart_v2.csv gains n_coda/coda_why.
  Receipts: coda_t4_point_{01,05,24,28,45,48}.png strips in
  charts_wasb/. Frozen t1/t2/m3 trees, scripts, and constants
  untouched; ground truth and point-boundary outputs consumed, not
  regenerated.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — Event detector v5: the skeleton becomes the design, and the chart gets a north star

**Diagnosed first, on all 57 rally-count misses, not a sample.** The
M2-era detector (image-y cusps classified hit-or-bounce, tuned on
sparse 25fps SAM tracks, running on dense WASB tracks it was never
designed for) was wrong on rally length on 34 t3 and 23 t4 clips.
Categorized from event-stream dumps and strips, with counts:

    category                              t3   t4   recoverable?
    FOOTAGE (editor cut into the rally)   14    3   no — film's gone
    OVER (dead-ball coda / dup cusps)      4    9   yes — structure
    UNDER (far-end hits the swing gate
           can't see; crossings exist)    ~7   ~4   yes — structure
    UNDER (crossing-recall holes)         ~9   ~7   partially

Two findings inside those numbers. Clay HAS codas — the coda pass was
shipped t4-only, and t3 points 05/10/21/29 chart +2..+4 phantom shots
each (05's strip: four "hits" while Ruud walks to the towel). And the
far-end miss is systematic: a far hit at 250+ px is a ~1 px/frame
image-y wiggle the near-tuned gate can't feel, but its FLIGHT crosses
the net at full projected speed — t3_point_33's five missed far hits
all launch clean crossings.

**The keystone measurement, taken before any design was written: raw
crossing count + 1 already beats the whole detector.** Rally ±1 from
just counting the track's strict net crossings: t3 36/59 vs the
chart's 25/59, t4 36/49 vs 26/49. The t4 coda entry's insight — a
live rally sends the ball across the net every shot — wasn't a
truncation rule; it was the design, inverted. So v5
(experiments/events_v5.py, shared by all four t*w twins behind an
EVENTS flag, default v5): crossings partition the rally, each
partition contains EXACTLY ONE hit, and the cusp detector is demoted
from counting shots to locating each hit inside its slot (strongest
envelope-valid cusp near the crossing start, ANY speed class — the
"bounce"-classified far hits are exactly the recoveries — else a
synth at the crossing start). The serve anchors the chain's front;
a trailing no-cross shot within RECV_S of the last crossing's end is
the net-error/failed-return slot; dead gaps cut the chain ONLY when
the track observed the gap AND the ball never passed the net inside
it (a hole says "didn't see", not "nothing crossed" — t3_point_29's
41-49 m deep-lob excursions fail the frozen 40 m span cap while being
perfectly alive, and its mid-rally gaps must bridge while t4's
tracked codas must cut). Partitions longer than any single flight
(> 2.0 s, incl. the pre-chain front) admit hidden hits from weak net
passes and strong leftover cusps. And the spine adjudicates the serve
detector: t4_point_15's stance call at f406 of a 441-frame clip sits
AFTER the entire crossing story — refuted, twin falls back to its
no-serve path (the old chart had amputated the clip to 1 shot).

**Train/test, declared before scoring: every new constant tuned on t3
+ dev-reel spot checks only; t1/t2/t4 scored untouched.** Tuned:
LAUNCH_BACK/FWD 0.5/0.2 s, EXTRA_PART_S 2.0 (swept 1.2/1.5/2.0 on t3:
34/35/36 rally ±1), EXTRA_SEP_S 0.8, SERVE_SNAP_S 1.0, GAP_COV 0.5,
suspect-serve gate. Inherited frozen: all CROSS_* gates, DEAD_GAP_S,
RECV_S, and the entire cusp/collapse/envelope machinery. m3 spot
checks: sane ordered streams, no explosions. One honest chart-level
consequence, fixed and named: v4 charts had been handing the serve
the RETURN's far-half landing (the serve's own near-half bounce is
invisible to the collapse detector BY CONSTRUCTION, and the missing
return let the serve steal the next bounce). v5 places the return at
its true launch, the theft stopped, and t3 serve zones collapsed to
4/12 — recovered honestly with a near-half serve-landing fill: a ball
flying INTO the camera never reverses image-y at the bounce, it
DECELERATES, so the bounce is the first big descending-velocity kink
(t3_point_25: viy 34 -> 9 px/f at f68, cy 18.5, dead on the service
line). t3 8/17 with real serve bounces. t4's truncate_coda stands
down under v5 — the chain already excluded the coda structurally.

**Scorecards, all four matches, before -> after (regressions bolded
in prose, not hidden):**

    t3 clay (59, TUNED)   before   after      t4 grass (49, HELD OUT)  before   after
    server end            48/59    48/59      server end               38/49    37/49
    rally len ±1          25/59    36/59      rally len ±1             26/49    28/49
    serve zone             9/18     8/17      serve zone               16/27    15/32
    letters (all)        70/105  132/173      letters (all)           67/123   97/159
    letters (aligned)     19/24    65/85      letters (aligned)        16/32    15/28
    ending type           10/31    19/42      ending type              12/33     9/33

    t1 night (22, HELD OUT)  before  after    t2 ctrl (5, HELD OUT)  before  after
    server end                9/22   10/22    server end                2/5     2/5
    rally len ±1             16/22   13/22    rally len ±1              4/5     5/5
    serve zone                 3/3   11/12    serve zone                1/1     1/3
    letters (aligned)          2/2    9/11    letters (aligned)       11/14   12/12
    ending type               6/11    9/16    ending type               2/3     2/3

The tuned tree: rally 42% -> 61%, and letters (aligned) — the honest
letter metric — more than tripled its base at a higher rate (79% ->
76% on 24 -> 85). Held-out t4 rally 53% -> 57% with letters (all) 54%
-> 61%. The regressions, on the record: t1 rally 16/22 -> 13/22 — the
skeleton needs crossings and the night reel's low-recall tracks
amputate what the old cusp counter padded; t4 endings 12/33 -> 9/33
and server end -1 — endings inherit final-shot identity, which v5
reshuffled; t2/t4 serve-zone rates fell as commits rose (the old
zone numbers were partly the return-landing artifact wearing a serve
costume — some of those "rights" were never real). t4_point_23 still
charts 19/5: the replayed let is played ball on both sides of the
net, and the spine is blind to it by the same construction as score
identity.

**And the north star, drafted and wired: token-level acceptance.**
experiments/mcp_accept.py tokenizes machine and MCP strings as
[serve+zone][letter+direction]*[ending] and accepts a point iff the
lists are within ONE token edit — strict equality, '?' matches
nothing, refusing to commit costs the same as being wrong. Every t*w
eval now prints it. Acceptance before v5: 0/135. After: 3/135 (2.2%)
— t1 2/22, t2 0/5, t3 1/59, t4 0/49; mean token distance 7.54 ->
7.18. Two percent is the honest read: rally length was the keystone,
but the metric demands the zones and letters stacked on top of it,
per point, near-perfectly. That's the mountain, measured.

Receipts: v5_t3_point_33.png (five recovered far-end hits, 9/15 ->
15/15 exact), v5_t3_point_05.png (clay coda cut, 9/5 -> 5/5),
v5_t4_point_05.png (held-out grass coda excluded by the chain itself,
14/8 -> 8/8) in charts_wasb/.
- New shared modules: events_v5.py (crossing-skeleton detector, full
  constants provenance in-file), mcp_accept.py (tokenizer + token
  Levenshtein). All four t*w chart twins wire v5 as default (EVENTS
  flag reruns v4), drop the pre-serve event filter under v5, and grow
  the near-half serve-landing kink fill; t4w's truncate_coda is
  v4-path-only now. All four t*w evals report acceptance. Frozen
  non-w trees, ground truth, and point-boundary outputs untouched.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — Acceptance decomposition: the 7.18 edits get names and addresses

**Before choosing what to build next, the mean token distance was
opened up, not guessed at.** experiments/mcp_decompose.py backtraces
the acceptance metric's own Levenshtein to an alignment and bins every
edit on all 135 scored points — pure computation on the existing v5
chart CSVs and MCP strings; no detector, chart, or eval logic touched,
and the baseline replicates exactly (3/135, mean 7.18). Mean edits per
point, overall: deletions (shots never charted) 1.95, letter+direction
both wrong 1.61, direction-only 1.20, ending 0.70, insertions
(phantom shots) 0.54, cross-type 0.44, letter-only 0.43, serve zone
0.30. The top sinks, named: DIRECTION DIGITS (in 2.81 edits/pt — 39%
of the whole budget), STRUCTURE (ins+del 2.49/pt, but split honestly:
t3's 2.68 deletions are mostly the editor's film, t4's 1.24 insertions
are our phantoms), ENDINGS (0.70/pt on a single token per point).
Direction is the weakest component by every cut: attempted on only 70%
of rally shots (the landing detector is far-half-only BY CONSTRUCTION)
and 48% right when attempted — barely above the 33% floor — while MCP
commits a direction on 848/850 strokes. And it's not a refusal
problem alone: 221 committed-wrong vs 159 refused.

**The edit-effort curve says the skeleton is no longer the wall.**
Acceptance at ≤1/2/3/5 edits: 2.2% / 6.7% / 14.8% / 41.5%. The same
thresholds on shot-count + letters only: 23.7% / 42.2% / 58.5% /
75.6%. Structure alone would accept 10x more points than the full
metric — the annotations stacked on v5's spine (zones, directions,
endings) are now the binding constraint.

**The headroom table — fix ONE component to MCP truth at aligned
positions, re-score, the deliverable:**

    counterfactual                       accept ≤1      mean dist
    baseline (v5 as charted)             3/135 ( 2.2%)  7.18
    directions perfect (all shots)       9/135 ( 6.7%)  5.88
    endings perfect                      7/135 ( 5.2%)  6.47
    serve zone perfect                   5/135 ( 3.7%)  6.87
    letters perfect                      5/135 ( 3.7%)  6.63
    directions perfect (attempted only)  4/135 ( 3.0%)  6.33
    letters + directions                21/135 (15.6%)  3.93
    letters + dirs + endings            53/135 (39.3%)  3.23
    all components (structure residual) 59/135 (43.7%)  2.93

No single fix rescues acceptance — the mean point is wrong on several
axes at once, so singles top out at 6.7%. But the compounding is steep
and ordered, and structure caps the whole game at 43.7%.

**Recommendation: build a real shot-direction model next.** Directions
are the largest single sink, the largest single headroom, and a
prerequisite for every compound ceiling above 7%. The tell is in the
counterfactuals: perfecting only the ATTEMPTED directions is worth
almost nothing (3.0%) — sparsity and accuracy are the same disease,
the far-half-only landing detector wearing a direction costume. A
direction model that works on BOTH halves (receiver contact geometry
as the where-did-it-go signal, with landing as corroboration when it
exists) attacks the 2.81 edits/pt head-on; endings ride second (one
token, 0.70/pt, 30% right); t4's phantom insertions are the fixable
half of structure. t3's deleted shots stay footage-capped — that
ceiling is the film's, not ours.

- New: experiments/mcp_decompose.py (edit attribution, effort curves,
  direction spot check, counterfactual headroom — reuses mcp_accept.py
  tokenizers verbatim). docs/benchmark.md grows an "Acceptance
  decomposition" section with the same tables. Frozen trees, ground
  truth, charts, and all t*w eval/detector logic untouched.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — Shot direction v2: the digit was naming the wrong court

**Semantics before code: the direction digit was being read in the
wrong coordinate frame, and a calibration table proved it before
anything was built.** MCP's instructions define direction 1 as "a
right-hander's forehand side (a lefty's backhand)" — the parenthetical
is the tell: that sentence names a FIXED SIDE of the receiving half,
mirrored by which END is receiving, NOT flipped by who's holding the
racquet. Our zone() mapped landing court-x into ABSOLUTE image
thirds — correct for a far righty receiver by coincidence, mirrored
for every near receiver, and 48% when attempted looked like noise when
half of it was a sign error. Every plausible mapping, tested on all
150 committed aligned landings on length-matched points across the 4
matches (experiments/dir_calibrate.py — t1 both-lefty, t2/t3/t4 all
righty, so the matches jointly disambiguate):

    mapping                        t1(L)  t2(R)  t3(R)  t4(R)   overall
    abs asc (the shipped zone())    7/24   6/17  32/79  12/30   57/150 (38%)
    abs desc (mirror)               9/24   6/17  33/79  14/30   62/150 (41%)
    RECV-END MIRROR, no handedness 11/24   8/17  46/79  14/30   79/150 (53%)
    recv-end + handedness flip      5/24   8/17  46/79  14/30   73/150 (49%)
    handedness only                 9/24   6/17  32/79  12/30   59/150 (39%)

The both-lefty t1 match adjudicates the row that matters: 11/24 vs
5/24 AGAINST the handedness flip — the naive reading of the spec loses
to its own parenthetical. And a split hiding inside the winner:
near-half landings under the mirror score 19/22 on t3 while far-half
score 27/57 — the collapse detector's far-half landing x (the ONLY
direction signal the chart had) is the noisy measurement, and the
near-half kink fills are nearly clean. The complementary serve table
came free and PASSED: shipped serve_zone() 11/16 vs 3/16 for its 4<->6
swap — wide/body/T stands, serves untouched.

**The estimator: every shot in the dense-WASB era has a
where-did-it-go signal now.** shot_direction.py, shared by all four
t*w twins, replacing zone(lx). Signal quality measured on t3 ONLY
(n=113 aligned pairs; dir_signals_dev.py); t1/t2/t4 held out:

    signal                       coverage      accuracy (t3)
    near-half landing            22   (19%)    19/22  (86%)
    receiver contact             91   (81%)    70/91  (77%)
    crossing + slope            105   (93%)    58/105 (55%)
    far-half landing             57   (50%)    27/57  (47%)

The receiver's contact point for the NEXT shot is the workhorse —
where the ball was received IS where this shot went, and the airborne
projection's x degrades gently even where its y is garbage. The
crossing signal (net-crossing x plus flight dx/dy, the v5 skeleton
moonlighting) improved 55% -> 67% under a depth sweep that rose
monotonically 0-12 m and plateaued at the receiver's BASELINE — the
flight line evaluated where the receiver actually stands. Precedence =
the measured ladder. The refusal question got an experiment instead of
a principle: vetoing on runner-up disagreement kept precision 78% vs
77% but cut net-right tokens 85 -> 62 — acceptance charges a refusal
and an error the same token edit, so a 77% guess strictly beats an
honest shrug. Commit-always won; '?' survives only for the signal-less
shot (2/113 on t3).

**Scorecards — the direction component in isolation, before -> after,
tuned and held-out separated** (attempt rate / accuracy-when-attempted
on aligned pairs, from mcp_decompose):

    match                 attempted            right when attempted
    t3 clay (TUNED)       71% -> 262/267 (98%)   48% -> 192/251 (76%)
    t1 night (HELD OUT)   80% ->  92/95  (97%)   36% ->  59/89  (66%)
    t2 ctrl (HELD OUT)    73% ->  25/26  (96%)   42% ->  22/25  (88%)
    t4 grass (HELD OUT)   64% -> 261/272 (96%)   56% -> 153/203 (75%)
    overall               70% -> 640/660 (97%)   48% -> 426/568 (75%)

Both diseases treated at once, as the decomposition demanded: attempt
rate 70 -> 97 AND accuracy 48 -> 75, held-out gains matching the tuned
tree (t2's 88% is small-n grace; t1's 66% is the night reel paying its
usual geometry tax). Strict positional direction accuracy ('?' counts
wrong) went 28% -> 69%; direction refusals 159 -> 12.

**And the north star moved:** acceptance 3/135 -> 7/135 (t1 2->3, t2
0->1, t3 1->3, t4 0->0), mean token distance 7.18 -> 6.19, effort
curve at <=1/2/3/5 edits 2.2/6.7/14.8/41.5% -> 5.2/11.1/23.7/57.0%.
Every OTHER metric on all four scorecards is byte-identical (server
end, rally +-1, serve zone, letters, endings) — the change touches
only rally direction digits and the evals confirm nothing else moved.
On the record: t4 acceptance stays 0/49 — its disease is structure
(1.29 phantom insertions + 1.43 deletions per point), and no direction
digit can fix a token that shouldn't exist. The lever is now spent by
its own measure: direction-only edits fell 1.20 -> 0.47/pt, and the
re-run counterfactual says "directions perfect" is worth +1 point
(8/135). The new top sinks, named for next time: structure 2.57/pt,
letters 1.71/pt across their two bins, endings 0.70 — with
letters+dirs+endings-perfect now reading 35.6% and all-components
44.4%. Endings ride next: one token per point, 30% right.

- New: experiments/shot_direction.py (receiver-mirrored mapping + the
  signal-ladder estimator, constants provenance in-file),
  dir_calibrate.py (the semantics table), dir_signals_dev.py (t3
  signal harness). All four t*w chart twins swap zone(lx) for
  shot_direction.annotate() and drop the orphaned zone(); serve-zone
  logic untouched. docs/benchmark.md: header table + a "Shot direction
  v2" section. Frozen non-w trees, ground truth, and point-boundary
  outputs untouched.
- Session cost: $0.00. Project total: ~$4.15 of $9.

## 2026-07-10 — The letter sink: audit the boxes, clean the boxes, then buy better boxes for one tree

**Quantified before built, per the house rule.** Letters were the
largest substitution sink after direction v2 (1.71 edits/pt in their
two bins; 55% strict positional on length-matched points), and the
suspected root cause has been in this LOG since the pilot: the $0
bgsub player boxes go rogue at exactly the wrong moments — spectators,
net-tape ghosts, shadows, player+shadow merges — and the letter is
ball-x vs box-center-x at the contact frame, gated on the ball
reaching the box. experiments/box_letter_audit.py binned every aligned
rally letter (all 4 matches, pre-fix charts) by the condition of the
box that fed it — sane / implausible (court-half + size sanity) /
absent:

    outcome         sane   implaus  absent   total
    right             70      31       0      101
    wrong             17      18       0       35
    refused (gate)    11      23       0       34
    refused (no box)   0       0       8        8
    total             98      72       8      178

45% of aligned letters read off a bad box; 39% right there vs 71% on
sane boxes. Box-quality ceiling ~+26 strict letters. The premise held,
so the work proceeded — cheap fixes first.

**Box hygiene (player_boxes.py, shared by all four twins; constants
tuned on t3 ONLY, t1/t2/t4 held out), with the dead ends on the
record.** What shipped: court-half plausibility in court coordinates
(foot minus clip offset through H_img_to_court, generous slack — the
serve detector's stance geometry), x-only teleport rejection (center-x
vs the temporal-neighbor median, radius in meters converted at the
box's own depth), linear interpolation through short dropouts, and a
letter gate widened to max(observed, clip-typical) body height — a
legs-only partial blob under-gates by half a body. What was BUILT,
MEASURED, and REMOVED: a height-vs-depth gate (k = h_px x
meters-per-pixel(foot) is depth-invariant for a standing human, but
bgsub box size is bimodal — full-body vs legs-only — on BOTH ends, so
every clip reference lands between the modes and the gate killed real
boxes producing right letters at contact distance 0: t3 65 -> 62); a
y-term in the teleport gate (partial-blob foot_y flicker projects to
fake multi-meter far-end teleports; it deleted the far side wholesale);
a multi-frame letter vote (±1 frame 67 -> 66, ±3 -> 63 — post-contact
flight frames poison the median). Net: t3 strict letters 65 -> 67
(tuned), t4 17 -> 18 (held out), t1/t2 unchanged; overall 114/209 ->
117/209, letter edits 1.71 -> 1.59/pt, mean token distance 6.19 ->
6.05, acceptance 7/135 unchanged. Every other metric on all four
scorecards byte-identical (server end, rally ±1, serve zone, endings —
the evals confirm). That is a plateau at ~1/8th of the audit ceiling,
and the diagnosis of the residual is unambiguous: on the failing far
shots the far player is NOT IN the bgsub CSV anywhere near contact
(t3_point_25: six far letters refused, the "far box" hundreds of px
from the ball on every window frame). No temporal hygiene conjures a
player the tracker never saw. That's the gate for authorized spend.

**SAM-3 on the worst box-bound tree: t3, ~$12, one tree only.** t4 is
worse by letter RATE, but its wrong letters sit on SANE boxes (11/13)
— striker/geometry disease, not boxes; t3 holds the box-driven mass
(33 of 46 failures on implausible/absent boxes) and the most absolute
misses (47). experiments/t3_sam_players.py re-ran the M3 recipe over
the 59-clip tree: box prompts derived AUTOMATICALLY from the bgsub
boxes (tallest hygiene-passing far box; median-height near box),
split-and-stitch for the 8 clips past fal's ~490-frame chunk limit,
masks split by the M3 component recipe, rows mapped back to the
stabilized frame the ball CSV lives in. Two API landmines, receipts
cached in outputs/t3/sam_raw/: prompts on DIFFERENT frames silently
track one object; and even same-frame two-box calls sometimes drop one
(the mask is empty inside the second prompt box AT the prompt frame) —
fixed with per-side repair calls. ~97 video-rle calls ≈ $12 at the M3
rate. Coverage: 51/59 clips ≥85% both ends; 6 clips keep a far-less
segment because bgsub never produced ANY promptable far box there —
the auto-prompt inherits bgsub's blindness at bootstrap, a real cost
of the buy option. bgsub CSVs untouched in players/; SAM CSVs in
players_sam/; the t3 twin switches via PLAYERS_DIR.

**The A/B, same eval, same hygiene, same everything else:**

    t3 letters                bgsub+hygiene   SAM-3
    strict positional          67/114 (59%)   76/114 (67%)
    committed-aligned (eval)   67/85  (79%)   75/89  (84%)
    letters (all)             138/174        162/205

+21 gains / -13 losses: 16 of the gains are far-side, 15 are refusals
turned right (t3_point_25 alone returns 7 — the far player finally
exists at contact), and the losses concentrate in exactly the
unpromptable far-less segments (t3_point_33 returns 4). Directions
ride along 86 -> 89 (receiver-contact is a box consumer). Acceptance
stays 3/59 — letters alone don't cross the ≤1-edit bar, as the
decomposition's counterfactuals predicted (letters-perfect is worth
+6 points overall). Verdict for the consolidation decision: SAM buys
roughly a third of the remaining letter deficit on the tree where
boxes are the disease, at ~$0.20/clip all-in, and its failure mode is
the same bootstrap blindness — it needs a prompt from somewhere. The
shipped default stays bgsub at $0; the delta is now a measured number
instead of a suspicion.

- New: experiments/box_letter_audit.py (the quantification),
  player_boxes.py (shared hygiene, dead-ends documented in-file),
  t3_sam_players.py (auto-prompted SAM fleet + repair pass). All four
  t*w chart twins: player_boxes.load() swap-in, widened letter gate,
  PLAYERS_DIR config (t3). docs/benchmark.md: header letters row, a
  "Letters" section with the audit and the A/B. Frozen non-w trees,
  ground truth, serves.csv, and point-boundary outputs untouched.
- Session cost: ~$12 (SAM A/B, under the one-off $30 authorization).
  Project total: ~$16.

## 2026-07-10 — Consolidation with a gate, and the confidence layer that makes a draft usable

**The sprawl becomes the tool.** Twenty-odd experiment scripts — four
chart twins kept in sync by hand, four evals, the shared modules they
imported by sys.path adjacency — are now the `courtvision/` package:
one chart assembler, one eval, one CLI (`chart` / `eval` / `draft`),
and every per-broadcaster difference the twins carried in code moved
into `data/matches/<id>.yaml` (lefty, clip offsets, serve lock,
side-gated serve zones, near-half ending fill, coda reporting, the
changeover-parity set priors). events_v5, shot_direction,
player_boxes, and mcp_accept lifted verbatim; the upstream stages
(WASB ball, bgsub players, serve v3, score-bug boundaries, MCP align,
the decomposition) came along as config-driven modules. Adding a
match is now a YAML, not a fork.

**The gate, because "refactor" is where numbers go to drift:** the
package had to reproduce the benchmark EXACTLY before anything new
was allowed on top. It did — all 138 chart CSVs byte-for-byte
across the four trees, all four scorecards byte-identical down to
the per-clip lines, acceptance 7/135 untouched. The experiments/
directory stays frozen as history (divergence policy in
courtvision/README.md: v5-only in the package, serve v3-only for
t1/t2 reruns, no SAM fleet lift — the receipts stay where they were
written).

**Then the layer the MVP actually needed.** A 57%-within-5-edits
draft is useless if the charter has to discover per point whether
it's the 5%-acceptance kind or the 25-edit kind. Per-point signals
the pipeline already computes — serve commit + stance margin, striker
-chain conflicts, ball coverage and max hole, letter/direction
refusal fractions and contact distances, the direction signal-tier
ladder, ending commit, crossings-vs-shots consistency, shot count —
feed a numpy logistic, LOMO-calibrated against token edit distance.
One bug found on the way in: aces were charged 100% letter refusal
for having no rally letters to refuse — the cleanest points in the
set, punished for being short. One genuinely new signal found on the
way in: the mid-rally-start signature. The clay editor cuts INTO
rallies, and on those clips the ball-launch serve detector fires on
the clip's first crossing — a "serve" 0 s into the clip whose launch
cy sits INSIDE the court (a real serve's toss projects 20+ m beyond
the baselines). That one mechanistic gate took the held-out t3 fold
from 65% to 84% high-tier precision.

**The discipline, and the tier that died of it.** Fit on 3 matches,
score the held-out 4th, all four ways. The tier I wanted — "sign off
at a glance", within 2 token edits at >=85% precision — was built
FIRST and failed the audit: 50% held-out precision at 1.5% coverage.
135 points at an 11% base rate cannot support it; two tiers ship,
not three, and the failed one is on the record. What survives is the
usable-draft bar the effort curve already named:

    LOMO (held out)   high prec (<=5 edits)   coverage   low <=5 rate
    t1 night              11/11 (100%)          50.0%        27%
    t2 ctrl                 3/3 (100%)          60.0%        50%
    t3 clay               16/19  (84%)          32.2%        43%
    t4 grass              11/11 (100%)          22.4%        50%
    pooled                41/44  (93%)          32.6%        44%

    flag x edits (pooled)   0-1    2   3-5   6+
    high                      6    6    29    3
    low                       1    2    37   51

**What the MVP can promise today, said plainly.** Given a broadcast
match with a fitted homography and a transcribed score-bug alignment:
a draft chart for every tracked point at $0, and a HIGH flag on ~a
third of them that means "start from the draft" with 93% reliability
(3 disasters in 44 flags, held out). What it cannot promise: that a
high-flagged point is RIGHT (27% within 2 edits), that a low-flagged
point is chartable at all without the video (56% are 6+ edits out),
or anything about the footage the editor never broadcast — t3's
deletions stay invisible to every signal except the serve forensics.
The draft-and-triage loop a charter would actually run now exists:
`courtvision draft <match>` -> outputs/<t>/export/<t>_mcp_draft.csv,
MCP points schema, machine string in 1st, confidence + clip +
jump-to timestamp appended. 138 draft points, 46 flagged high.

- New: courtvision/ (18 modules; README with the divergence policy),
  data/matches/t{1-4}.yaml, data/confidence_model.json (all-data fit;
  the honest numbers are the LOMO table), docs/USAGE.md.
  docs/benchmark.md: consolidation + confidence section. Frozen
  non-w trees, ground truth, serves.csv, and point-boundary outputs
  untouched; experiments/ untouched.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-11 — Three matches in one pass: the calibration layer gets the n it asked for

**The premise.** Every honest caveat in the confidence layer's writeup
was a sample-size caveat: 135 points, an 11% base rate, a top tier
that died of n. So this pass optimizes for aligned points first and
axis coverage second. Candidate pool: MCP-charted finals cross-refed
against YouTube, where the winner format turned out to be the
"condensed match" upload — 20-40 minute re-cuts that keep nearly
every point and none of the studio filler; they beat extended
highlights for alignment yield everywhere they exist. Picked: t5
Sinner-Zverev AO F 2025 (hard, night, AO feed, 190 MCP points), t6
Sabalenka-Pegula US Open F 2024 (hard, WTA, USO feed, 167), t7
Djokovic-Sinner ATP Finals RR 2023 (indoor hard, Tennis TV feed,
218). 575 MCP points on the table; 372 clips extracted; 356 scored.

**Staging is configs now, and the configs earned three new knobs.**
Everything ran through the package stages — fitcourt, probe, extract,
boundaries, align — plus per-match YAML; no script twins. Each feed
broke exactly one thing, and each fix went into `court_detect` as
config, not fork: t5's bright near-court coating starved the tophat
default (knob: `tophat_t`), t6's green apron can't share an HSV band
with the blue court (knob: second hull band, OR'd), t7's light-blue
apron rides V~252 (open the ceiling). Two staging lessons re-learned
the hard way: the hull must include the APRON or the boundary lines
erode away with it (t3 already knew this; t5 made me re-learn it),
and the homography fit window must be motion-scanned first — the
first t5 window had 26.5 px of pan smearing the plate; a static 5 s
serve setup landed the fit at <=0.2 px residuals. t7's far baseline
bows 7.7 px of lens distortion, documented and accepted like t3's.

**The bugs hide their zeros, and the plateau machinery pays for it.**
All three broadcasters drop zero-valued trailing columns — at 0-0 the
points column simply isn't rendered. The era window then contains
live background, and game-start points split into duplicate-score
fragment groups (t6: 4 groups incl. one trio, t7: 7). The adjudication
that emerged is better than the one I planned: MCP's own 1st/2nd
columns arbitrate most groups — when the row reads `4w` + a
second-serve rally, the fragment pair IS the fault and the point,
filmed either side of a cutaway, and the serve lives in the later
piece. The rest fell to changeover parity plus eyeballed frame
strips; two groups were mid-rally cutaway splits (net-crossing counts
across the pair sum to MCP's strike count — the receipt). Losers get
blanked in the alignment CSV so the eval skips them instead of
double-charging one MCP row. Also caught this way: an IBM stats card
that passed the presence NCC as a "score bug" (t6_point_78).

**A tiebreak set is not 13 games.** The set-3 fold of t7's server-end
parity vote came back 17 agree / 37 disagree — systematically flipped.
The staged prior counted set 2 as 12 games + TB = 13 swap-units (odd).
Wrong: the 7-4 TB is 11 points, which is ONE internal end-change at 6
points, and the set-end change cancels it — net EVEN parity. Prior
25 -> 24, verified on video (Sinner opens set 3 serving from the far
end), vote flips to 138/27 for start_end=near. The general rule, for
the next TB match staged: a TB set's swap parity depends on the TB's
point total, not on calling it a 13th game.

**Scorecards, same constants, nothing re-tuned:** t5 acceptance 2/71
(server end 75% — the AO night feed's serve-end detection is the
noisiest of the three; parity vote 53/17), t6 10/128 with server end
121/128 and the best single-match acceptance on record (7.8% — WTA +
the USO feed's stable wide camera chart cleanly), t7 9/157 with rally
+-1 at 83%, the best structural fold yet. Pooled: 28/491 (5.7%) from
7/135 (5.2%) — the bar held under a 3.6x bigger, five-feed test.

**The recalibration, which was the point.** LOMO across seven folds,
491 points: pooled HIGH precision 88% (92/104) at 21.2% coverage,
from 93% (41/44) at 32.6%. Both numbers moved toward the truth, not
away from it — 44 flags was a small-n flattery, and the new folds
name t4 as the weak match (18/26, 69%: its phantom-insertion disease
flies under signals built to catch missing data, not invented data).
t6 held 97% at 27%. The strict <=2-edit tier was re-attempted at
n=491 and still dies in LOMO — 0% coverage at a 19% base rate; still
on the record, still not shipped. Shipped scorer refit on all 491
(t_high=0.731): 102/491 flagged high at 94% in-sample. Exports
regenerated for all seven matches: 508 draft points, 104 high.

**Process failures, on the record.** The run stalled overnight: the
background-job watcher died silently between t6 ball tracking and t7
players — both jobs finished fine; nothing was listening. The rest of
the run polled long jobs from the driving loop instead of trusting
completion callbacks. And the t5 players stage reported 100%/100%
coverage, which was flagged as suspicious and turned out to be real —
the AO feed's static wide camera plus a clean bgsub plate; sometimes
the number is just good.

- New: courtvision/{fitcourt,probe,extract}.py (staging stages behind
  the CLI), data/matches/t{5,6,7}.yaml, data/mcp/points_* + alignment
  + map CSVs for the three matches, outputs/t{5,6,7}. Modified:
  boundaries CFG (+3 matches, constant-width era windows), config/cli
  (video + court_detect), confidence_model.json (7-match refit),
  benchmark.md, USAGE.md, courtvision/README.md. Frozen experiments/,
  t1-t4 ground truth, and t1-t4 charts untouched (chart CSVs never
  re-run; only their exports carry the new confidence column).
  Marginal cost of the three matches: $0.

## 2026-07-11 — The t4 autopsy: what a clean-looking fragment costs, and the gates that catch it

**The weak fold gets its autopsy.** The recalibration named t4 (grass,
Wimbledon 2024 WTA F) the drag: 69% held-out HIGH precision against
everyone else's ≥84. Eight false-highs, every one pulled apart against
its chart CSV, MCP string, and — house rule — frame strips, because the
mechanism gets named from pixels or it doesn't get named. Two diseases,
neither of them the phantom-insertion story I'd been telling. First:
the half-cadence chart. t4_point_02 charts 8 shots at 1.5 s spacing;
the pixels show the real exchange running 0.9 s (near hit f245, far
hit ~f272, near winding up again by f293 — a stroke our chart simply
never had). Every other stroke missing, and the striker chain
alternates right over the top of it: conflicts 0, crossings_gap 0,
because the ball track is blind in exactly the same places the hit
detector is. Second: the dissolve-cut mid-rally join. t4_point_35
opens on a crowd cutaway and then FIVE SECONDS of live rally before
the stroke our stance detector blessed as a "serve" at f148. The grass
editor cuts into points like the clay editor — but t4's serves are all
stance-called (src=players), and the launch-plausibility gate only
inspects src=ball serves. It defaulted to pass on every t4 point,
+0.59 logit of unearned trust, a signal that structurally cannot fire
on this feed.

**The dead-ends, kept.** Weak-gated crossings do NOT recover the
missing strokes — point_02's 17-stroke rally yields 7 weak crossings;
the track never told the story, so no crossings-vs-shots residual can
either. And charted cadence (mean inter-shot gap) is real but soft:
AUC 0.62 on t4 AND cross-feed, yet its marginal LOMO flags ran 50/50 —
it gets a sidecar column for the charter's eyes, not a model seat.

**What shipped instead: two more rules from physics, one feature.**
`xr_pre_serve` — weak crossings that end before the charted serve;
two or more means the clip joined the rally mid-flight and the chart
can't be the whole point (also a fitted feature, where one crossing
can whisper). `rally_spineless` — a 3+-shot chart whose window holds
zero weak crossings charted a rally the ball never played
(t4_point_49: four shots, zero crossings, flagged high at p=0.747
against a 0.745 threshold). Both live beside the launch gate, outside
the fit. LOMO, all seven folds:

    t4    18/26 (69%) @ 53.1%  ->  17/20 (85%) @ 40.8%
    pooled 92/104 (88%) @ 21.2% ->  90/96 (94%) @ 19.6%

No other fold below 92%; t5's one disaster gated away (3/3). High-tier
6+-edit disasters halve, 12 -> 6. The trade is 1.6 pts of coverage,
and it's the honest direction: t4 was being flagged at 53% of the
match against a 61% base rate — the model wasn't discriminating, it
was liking t4's rosy serve signals. The residue is on the record too:
point_11 stays flagged at d=10 (its dissolve-cut leaves no pre-serve
crossings — the track is blind there as well), and 43/46 sit one edit
over the bar. Fragments the track never saw remain invisible to every
signal; that ceiling is t4's chart quality, not the calibration's.

- Modified: courtvision/confidence.py (3 signals, 2 gates, model
  feature +1), courtvision/export.py (docstring number),
  data/confidence_model.json (refit on 491, t_high=0.753, 97 high at
  96% in-sample), docs/benchmark.md (autopsy table + before/after).
  Exports regenerated for all seven matches (508 points, 99 high).
  Charts, ground truth, experiments/, and staging outputs untouched.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-11 — cv-17: the drafts go public, addressed to the 32

**Nothing in the pipeline ran.** The session turned outward — the
first since t1 with zero perception, charting, or eval work. The
seven draft exports are now published artifacts: all
`t*_mcp_draft.csv` hosted at trmccormick.com/exports/ (stable URLs via
Docusaurus `pathname://`, dodging the hashed-asset pipeline), and
cv-17 ("The 33rd Charter") is the cover letter, addressed to the
Match Charting Project's 32 active charters via Sackmann's Jan 2026
plea. Reply channel: GitHub issues on this repo. (Dead-end kept: the
CSVs went to static/data/ first — gitignored for build-time PageSpeed
metrics, so they built locally and would have 404'd in CI. They live
at static/exports/.) This is the public
probe on landscape open question #1 (has anyone attempted assisted
charting?) — if a charter answers, the question answers itself.

**Framing decisions worth keeping.** The ask is a stopwatch, not
praise: chart ten points cold vs correct ten green drafts, measured
by someone who holds the pencil — the ratio the whole project bets
on and has never measured. The grammar honesty went in the post
instead of under it: the strings are MCP-style, not MCP-legal (`s`
prefix, `?` direction tokens, none of MCP's depth/position/volley
vocabulary), converted from a gap into question 3 for the charter
(strict-legal with blanks vs visible uncertainty). And the license
note is explicit: the exports carry MCP-joined columns, so the files
inherit CC BY-NC-SA 4.0 with attribution to Tennis Abstract and the
volunteer charters.

**The hero is the file.** render_cv17_hero.py scrolls all 134 rows
of t6's draft as rendered pixels (39 HIGH in green, summary card
pinned), plays two HIGH points with the WASB comet and their draft
row in the bar, then the honest-split card — 39 start-from-draft /
95 re-chart / 6 unplaced / faults invisible — before the ask card.
Per-file split published in the post: t1 10/24, t2 4/6, t3 12/59,
t4 18/49, t5 3/71, t6 39/134, t7 13/165 = 99/508 high. t4's 37%
flag rate got the honest aside (its answer key runs half-cadence on
some rallies; "if you open one file to hunt over-confidence, open
that one").

- New: experiments/render_cv17_hero.py, outputs/cv17_hero{,_raw}.mp4
  (untracked). Blog repo: cv-17 post, static/exports/*.csv x7,
  static/img/blog/the-review-copy.mp4, social card. Pipeline, charts,
  exports, model: untouched.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-11 — cv-18 build: the review tool, by subagent assembly line

**The charter-assist MVP exists.** `courtvision review <match> --mode
review|cold --session NAME` serves a localhost three-pane UI — row
list with confidence colors, clip player that jumps to the serve,
keyboard-first edit bar with live MCP-legality lint — and logs every
action to events.jsonl with a pause key that stops the clock. The two
modes differ by exactly one thing (cold strips the draft string,
confidence, and serve timestamp server-side), which is the whole
stopwatch experiment's validity argument. `review-analyze` turns
three sessions into the cv-18 tables: timing with a 180s idle-hole
rule, accuracy vs MCP truth, the anchoring check, the correction
histogram, triage honesty. Zero new runtime deps — stdlib http.server
with a hand-rolled Range handler; the only addition is pytest (dev),
and the repo now has a test suite: 34 tests.

**Built by an assembly line, and the reviews earned their keep.**
Seven tasks, each a fresh implementer subagent gated by a fresh
reviewer; every blocking finding was real: (1) the plan's own lint
sample code would have failed the plan's own test — implementer
caught it; (2) an implementer's "helpful" setuptools build-system
flipped the uv project to editable-install and left egg-info litter —
reverted for pytest's pythonpath ini; (3) malformed POSTs crashed the
handler thread, Range-past-EOF returned broken 206s — hardened with
socket-level regression tests; (4) accept() could silently lose a
correction on a failed save, and the pause overlay hid behind an
open cheat sheet, a two-keystroke trap — both fixed; (5) the big one,
CRITICAL: the analyzer tokenized corrected strings with the
MCP-native tokenizer, so rubber-stamping a correct s-prefixed HIGH
draft scored a phantom serve edit — every review-arm number cv-18
would publish was biased against the draft. Fixed by tokenizing
corrections with the s-tolerant draft grammar, projection imported
from frozen mcp.py so the drift class is dead, regression tests
pinning all of it. A naive median (upper-of-two) also died there.

**The live smoke found what unit tests can't.** Playwright drove the
real UI against t3: blank serve_s rows threw NaN into currentTime and
silently killed autoplay — fixed. And a proper spook: phantom
row_opens accepted a row I never reviewed. One tab, idle-quiet,
events only during tool activity — the automation layer itself was
the second driver. The tool's own telemetry reconstructed the whole
incident to the millisecond, which is precisely the property the
experiment needs. A real session has one human and no phantom.

**Final review: merge, with one guard first.** The whole-branch pass
(which independently re-verified the rubber-stamp fix empirically:
43/128 exact on stamped drafts, ≈0 under the old code) demanded
export_sha256 verification at analysis load — if the export is
regenerated between charting and analysis, the numbers would
silently grade against a draft the charter never saw. Wired, with
corrupt-event-line tolerance, before Trevor spends the afternoon.

- New: courtvision/{notation,review,review_analysis}.py,
  courtvision/review_ui.html, tests/ (34), docs/cv18-protocol.md
  (frozen seeds cv18-a/cv18-b), USAGE review section. Modified: cli
  (+review, +review-analyze), pyproject (pytest dev-group), plan doc
  kept byte-identical to shipped UI. Deferred to cv-19: notation↔mcp
  vocab reconciliation ('g' foot-fault lint gap included), e2e
  analyze() test, UI polish list in the SDD ledger.
- Next: the experiment itself — docs/cv18-protocol.md, blocks 0-4.
  Machine time is done; the stopwatch is Trevor's.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-19 — The launch gate learns what a serve looks like on other feeds

**The HIGH tier was starving on the newest matches, and the blocker
had a name.** A workflow pass over the whole record (4 readers +
synthesis + adversarial critique) ranked the model-quality levers;
the top implementable one was pure diagnosis: t5+t7 hold 228 scored
points and yielded 6 LOMO HIGH flags between them, while t7 posts the
best structural scorecard on record. The autopsy
(experiments/conf_coverage_autopsy.py) instrumented the LOMO loop and
blamed each good-but-LOW point: on t7, 69 of 71 gate-blocked good
points died on the launch-plausibility gate alone. Pass rates: t7
11.5%, t5 16.9% vs t4 89.8%. The gate zeroed the tier before the
model ever voted — and serve_launch_plausible sits in MODEL_FEATURES
too, so it double-punished as a feature.

**The mechanism is track acquisition, not mid-rally joins.** launch_cy
is the ball's court-y at the START of the first sustained crossing
run. On feeds where WASB picks the serve up late — AO night ball,
Turin's small far-end ball — the run starts over the court and a real
serve reads "inside." The absolute band (cy <= -10 or >= 30) was read
off t3, where launch-inside genuinely meant the clay editor had cut
into a rally. The original insight was a conjunction — "a serve
called 0 s into the clip whose launch cy sits INSIDE the court" — and
the shipped gate kept only the cy half. Dead end, kept: re-adding the
TIME clause does not transfer either — Tennis TV cuts condensed clips
tight to the serve (t7 median serve_s 0.60 s), so serve_s < 1.0 still
blocks 59 good t7 points. What transfers is the STANCE read: 116 of
t7's 138 launch-inside clips carry a verified pre-launch baseline
stance (margin_m — the server settled at the baseline for 1.2 s
before the launch, which a mid-rally join cannot produce).

**The fix is one clause, and it survived its own discipline.**
implausible = launch inside the band AND no stance read
(courtvision/confidence.py). LOMO A/B over four candidate rules
(experiments/launch_gate_repair.py): the stance rule wins — pooled
94%@19.6% -> 92%@47.7%, t7 7.6% -> 58.6% at 93%, t6 28.1% -> 54.7%
at 97%, and the LOW tier's stranded good points (<=2 edits) fall
66 -> 13. Costs on the record: disasters in HIGH 6 -> 19 (of 234),
and two folds under-deliver — t5 77% (genuinely weak feed; part of
its old 4.2% coverage was honest triage) and t3 80% (the cy-only band
was accidentally protective against the editor's cuts). A
precision-target sweep (0.93, 0.95) starves healthy folds without
rescuing either — their disease is invisible to the signals, which is
the t4 lesson again from the other side. `courtvision calibrate`
reproduces the harness numbers exactly; 54 tests pass. Exports NOT
regenerated — the cv-18 artifacts are frozen behind export_sha256, so
the drafts stay at the 99-HIGH edition until the stopwatch closes.

- Roadmap (workflow synthesis + critique, full trace in session):
  1. finish charting app + run cv-18 (the gate for pipeline work) —
  in flight on the other branch; 2. THIS session's gate repair;
  3. WASB crossing recall (deletions 1.95/pt, the ceiling-raiser);
  4. player boxes for letters (bgsub far-half cut first, SAM-3 only
  where box-driven); 5. bounce-first endings (far-end ~1 px/frame
  resolution risk named); 6. t4 phantom insertions; 7. MCP-legal
  grammar (awaits cv-17 charter answer); 8. calibration n via the
  app's ground-truth factory. Unowned diagnoses parked: t4's
  sane-box letter disease, t5 serve-end on the AO night feed.
- New: experiments/{conf_coverage_autopsy,launch_gate_transfer,
  launch_gate_repair}.py. Modified: courtvision/confidence.py (one
  clause + docstring), data/confidence_model.json (t_high=0.744,
  flags 250/491 in-sample), docs/benchmark.md (+section). Exports,
  charts, review tool: untouched.
- Session cost: $0.00 (no SAM-3 needed). Project total: ~$16.

## 2026-07-20 — Two parked diagnoses, and the answer key confesses again

**The assignment was two unowned diseases; both dissolved under
pixels into something else.** Parallel diagnosis agents (analysis
only, house rule: mechanisms named from pixels) took the two parked
items from the roadmap critique.

**t4's "sane-box letter disease" is mostly not a letter disease.**
The stance serve detector fires 20-40 frames before true contact on
grass (serves.csv frame 64 vs pixel-true ~93 on point_30; frame 4 —
during a broadcast dissolve — vs ~20 on point_42). chart.py then
appends a SYNTH serve and charts the real serve as rally shot 1; a
late stroke drops off, total length coincidentally re-matches MCP,
the eval's length-match gate passes, and every rally letter is
scored against the MCP stroke one slot ahead — the OTHER player's
stroke, since strokes alternate. 13 of the 14 wrong aligned letters
sit in the 3 clips with this defect; 6 of them were pixel-CORRECT
reads charged as wrong by the index shift. It also owns a slice of
t4's outlier insertion bin (1.24 phantom edits/pt). The genuine
residue: 4-5 letters read against motion-smeared or sliver boxes
whose CENTER sits tens of px off the torso — boxes the audit calls
sane because it checks size and location, never center-vs-body.
Refuted with receipts: striker misattribution (every checked machine
striker was pixel-correct), handedness, two-hander geometry. Fix
leads (serve absorption when the first hit lands 15-45 f after a
synth serve with no net crossing between; smear-aware letter
refusal) are pipeline — cv-18 gate holds.

**t5's weak server end was half answer key.** The eval's
changeover-parity prior for set 2 was 9; set 1 went 6-3 = 9 games,
ODD, and after an odd set the set-end change and the game-1 change
are consecutive — prior 9 hands set-2 games 0 and 1 the same end.
The crosstab tell: set-2 odd-game-sum clips scored 2/10 while even
scored 12/3 — truth inversion, not noise (detector noise dilutes
both cells). Pixel receipts: scorebug + serve end on points 40/44,
plus point_45 as the masked class (detector wrong, scored right).
Fixed 9 -> 10: t5 server end 53 -> 59/71. Then the audit swept all
seven matches (experiments/parity_audit.py) and found the class in
three more staging files: t3 "*,1" 13 -> 12 (the TB set contributes
EVEN parity — t7's own lesson, unapplied one file over),
pixel-verified on the 4-2 Djokovic service game; t1 9/9/19 ->
10/10/20; t2 9 -> 10. Pooled server end 404 -> 414/491 (84.3%);
every other metric byte-identical (p1_end feeds only the server
tally). t4/t6/t7 audit clean — t4's cold cell is prior-free state
0,0, which really is the detector (the dissolve-cut class from the
letter diagnosis, same clips).

**The genuine t5 detector residual has a measured fix, parked.** 11
clips: the night track misses the serve's own crossing (hole-
truncated runs fail LAUNCH_SPAN/MONO/GAP), so find_launch's first
PASSING run is the RETURN — end flips. Guarded earliest-crossing
arbitration reaches t5 62/71, t6/t7 no-damage; unguarded reaches 91%
but pays t6 -6 on clip-start teleport junk. Also measured: a stance-
settle vote lifts t7 133 -> 147/157 but DAMAGES t5 (56%) and t6 —
per-feed staging or nothing. All pipeline; all wait for cv-18.

- Dead ends kept: naive first-crossing recovery (teleport junk),
  stance vote as a global rule, serve_s-based launch-gate repair
  (condensed feeds cut tight to the serve — yesterday's lesson
  reconfirmed from the other side).
- New: experiments/{t4_letter_diag,t4_letter_frames,
  diag2_t5_server_end,diag2_t5_launch_forensics,diag2_render_launch,
  diag2_end_recovery,diag2_first_crossing,
  diag2_first_crossing_guarded,parity_audit}.py; frame receipts in
  outputs/diag/ (untracked). Modified: data/matches/{t1,t2,t3,t5}
  .yaml (priors + receipts), docs/benchmark.md (+truth-correction
  section). Pipeline, charts, exports, model: untouched.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-20 — cv-19 build: the charting app ("courtvision charter")

**Excel hell has an exit.** One page, three flavors: `courtvision
charter <match>` (staged — pre-cut clips, machine drafts pre-loaded
as editable chips), `courtvision charter --new <id> --video <file>`
(chart-along any match), and `courtvision charter --emit-static
chart.html` (one 173KB self-contained file that charts offline in a
browser and could sit at trmccormick.com/chart unchanged). The
palette IS the cheat sheet: every legal MCP mark is on screen with
its meaning, as key and button; the swipe-to-instructions loop that
produced 6/134 review rows in a week is dead. The scoreboard runs
itself — winner derived from each string's parity+ending, games/
sets/tiebreaks/rotation automatic — and challenges the charter at
every game boundary ("chart says 1-0, 0-0 — screen agree?"). Missed
points get one-key unseen rows; cut clips get `?` endings with a
who-won prompt. Every match exports the training bundle: MCP points
CSV + segments CSV + manifest — a new benchmark match with zero
alignment work.

**The score engine is pinned to reality, twice.** score.py replayed
three real MCP points files — t3 (bo5), t6, t7 (two tiebreaks, rows
filed non-chronologically) — 592/592 points column-for-column, and
the winner-from-string derivation matched PtWinner on 592/592. The
oracles taught us Gm# never resets and that t7's file wraps at Pt 72.
The page's JS scorer then proves conformance by replaying the same
592 in the browser: SELFTEST PASS n=592, verified live. Grammar
lives once (grammar.json) — palette, lint, and static build all read
the same file; the foot-fault lint gap died as a side effect.

**The assembly line earned its keep again.** Eight tasks, reviewer
gates, opus final pass. Caught before any human charted a point:
plan-inherited keyboard collisions (x/m/u/y/p are notation AND were
action keys — one keypress mid-rally navigated away; actions moved
to uppercase), a quirks-mode-forcing script inject (proven with a
compatMode probe), a fault-undo that corrupted the entry, localStorage
resume that lost the match pointer, an unseen flag silently dropped,
and — final review — the browser export reading stored score columns
instead of replaying (the exact stale-score disease the raw-inputs
rule exists to prevent) plus a segments filter that disagreed between
flavors. Export parity is now verified byte-for-byte.

**Two collisions with a concurrent session, on the record.** Another
Claude session committed pipeline work (launch-gate repair; parity
priors) mid-build twice, and fix-amends entangled with its commits
both times. Both repaired losslessly (their work stands as its own
commits: 72eef74, 6be5342, 1a01217), and the assembly line now runs
a no-amend policy. Lesson filed: two agents, one repo, zero locks —
verify HEAD or don't amend.

- New: courtvision/{grammar.json,score.py,httpkit.py,chartapp.py,
  chart_ui.html}, tests (35 -> 66), docs/specs+plans for the app,
  USAGE charter section. CLI: `charter` added, pipeline `chart`
  restored after a collision ruling. Frozen and untouched:
  review.py, review_ui.html, mcp.py, benchmark, all charts.
- Parked for next cycle: cv-18 restart on this app (fresh seeds, all
  arms, --contaminated unions the practice era), static publish to
  trmccormick.com/chart, lint error-letter-before-*, review-flavor
  notes UI, headless selftest in CI-of-one.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-20 — cv-19 charting live: the editor and the auditor, born mid-match

**Six points into the first real chart (AO 2026 F,
Rybakina-Sabalenka, picked by an 18-candidate scouting fan-out and
proven uncharted against the MCP's own lists) the append-only fence
fell.** Click-to-edit shipped hot mid-session: chips reload, Enter
saves, Esc cancels, replay rescores everything downstream; the
server validates every edit by replay and reverts rather than brick.
The review loop went three rounds on controller-written code and
earned each one. Round 1's Critical was adjudicated into the app's
best feature: attested winners are TRUTH (re-deriving them from
replay would silently reassign real points), so contradictions
between a string and its attested winner under the replayed server
are surfaced, not resolved — amber ⚠ rows, a conflict count in the
score strip, export gated until the chart reconciles. The auditor's
first two catches were its own author and then its own reviewer:
my regression test asserted 1 conflict where it found 2 (my string
had server-wins parity), and the reviewer's verification script made
the same class of error and got flagged too. Two humans-worth of
parity mistakes in one day; none reached disk. Round 2 closed the
guard gaps: edit mode could leak through the unseen-point path and
fresh entries could be clobbered by row clicks — beginEdit now
confirms-and-clears ALL mid-flight work before loading anything.
- Suite 68. Charting continues toward the first MCP submission.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-20 — The doc set: Court Vision becomes legible as a data product

**Trevor's steer: "follow data product management best practices,
research them if you need to."** Three research agents swept the
2024-2026 practitioner literature (data-as-a-product / DATSIS, Model
Cards and Datasheets lineage, data-quality SLOs and scorecards); the
converged advice, scaled to a solo repo: consumer-first product
statement, the calibrated operating point AS the SLO, traffic lights
only with declared thresholds and prescribed actions, a gaps register
with a closed status vocabulary, an automation-bias warning as "the
single most consequential sentence," versioned benchmarks with
non-comparable markings, and no enterprise ceremony.

**Three new docs plus a README refresh.** docs/scorecard.md (what to
trust / verify / re-key, per component and per feed, with the gaps
register), docs/model-card.md (system card: intended use, explicit
out-of-scope topped by "never submit uncorrected drafts to MCP,"
factors, limitations with mechanisms, license chain), and
docs/data-product.md (consumers and jobs, data contracts, SLO-style
quality table with a correction-budget promotion rule, DATSIS
self-audit, lifecycle gates, risk register). The benchmark's
truth-corrected state is now formally **benchmark-v2** (v1→v2 =
today's parity correction; pre-correction server-end numbers
non-comparable). README updated from the stale M0-M4/SAM-ball era.

**Every claim adversarially verified before commit.** Four checker
agents audited 217 claims against benchmark.md/LOG.md/source; 18
discrepancies found and fixed, including three of my own inventions
the process caught red-handed: a "~92% real t4 letters" number the
record doesn't support (it supports ~84-87%), a "designed scorecard
generator" that was only ever my proposal, and a 10-15-minute
hand-charting estimate contradicted two paragraphs later by my own
"2 hours" (the record says ~1 min/point). Also caught: the shipped
cv-17 exports still carry the frozen 99-HIGH flags — the 92%@48%
contract exists only in the recalibrated scorer until cv-18 closes —
and the probe count (67) I'd quoted was the retired t1-era detector
(shipped: 50 interior + 96 line probes). The docs that preach
"hand-typed numbers eventually lie" were themselves the proof.

- Open decisions surfaced for Trevor: no LICENSE/NOTICE file (code
  all-rights-reserved by default; data implies CC BY-NC-SA — needs an
  explicit license map), and whether to build the scorecard-table
  generator so the numbers can never drift.
- New: docs/{scorecard,model-card,data-product}.md. Modified:
  README.md. Pipeline, charts, exports, model: untouched.
- Session cost: $0.00. Project total: ~$16.

**Addendum — the doc set ships as HTML.** Trevor's rule: anything
user-facing in markdown should be HTML. The three docs converted to
self-contained, theme-aware HTML (pandoc gfm -> html5 + shared style
header docs/_doc_style.html, title-block stripped, cross-links
repaired, render verified headless in light mode with dark handled by
CSS vars); the .md sources retired — the .html files are the record.
README stays markdown (GitHub's front door), as do the working
records (LOG, benchmark, USAGE, plans/specs) pending a wider ruling.

## 2026-07-20 — Three builds by assembly line while Trevor charts the AO final

**All three parallel tracks approved by their reviewers with zero
blocking findings; committed separately.** Constraints held: the live
ao26f charting session, the running app's files, the frozen exports,
and the shipped model were untouched throughout (Track B proved the
model JSON byte-identical after its calibrate rerun).

**import-bundle (d08d1cd).** The post-MVP command the cv-19 bundle
format was designed for: a charted match becomes benchmark-match
scaffolding — MCP points file, 1:1 alignment (the charter stamped
every window; the score-bug adjudication machinery has nothing to
do), stream-copied clips, and a match yaml whose set_priors are a
loud TODO(parity) citing this morning's odd-set lesson. Dry-run
receipt against the live session: 7 points, 6 windows, 1 unseen,
zero conflicts. When the AO final chart is done, benchmark-v3
material is one command away.

**The scorecard generator (ec62390).** calibrate dumps a LOMO
sidecar; gen_scorecard.py rewrites the marker-delimited regions of
docs/scorecard.html from the eval run — idempotent, prose preserved,
diff-containment mechanically verified. It promptly earned its keep:
four cells of the hand-built page were carrying 4-match-era numbers
under the benchmark-v2 heading (zone 26→40 strict, endings 30→20,
dirs 75→72, letters 60% pooled). The page that shipped preaching
"hand-typed numbers eventually lie" lied in four cells for two
commits. Now they cannot.

**WASB step-1: refuted, kept (bd9c5c2).** The roadmap's cheapest
crossing-recall idea — overlapping triplet windows — raises TRACK
recall on the t1 night reel (coverage 89.8→93.3%, holes shrink) and
LOWERS strict crossing recall (82→78): the recovered frames insert
non-monotone samples inside runs the holes had kept clean, and the
run gates split or refuse them. Twin scorecard flat-to-worse (rally
±1 13→12/22, mean distance 5.55→6.00). Dead end on the record for
the price of a pilot, before the cv-18 gate ever had to rule on it.
The per-detection scores the harness now retains are the input for
the surviving idea: down-weight wander instead of inserting it.

- Also this session: on-screen forced/unforced guidance in the
  charter UI at the err-mark decision (8cb1137, hot-swapped into the
  live session), and the charter server was found alive on :8766
  when it seemed lost — the tab had closed, not the process.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-20 — The pivot: training on the corpus beats hand-tuned rules (proof), and hand-tuning letters is a dead end

**Strategic reset, on Trevor's steer.** The north star is FULL
AUTOMATION — the robot charts matches, humans don't. Not "help a
charter go faster." The Match Charting Project corpus is ~11,600
charted matches (7,567 M + 4,081 W, public), and the tools to fetch
video (yt-dlp) + process it (ffmpeg) are already installed. So the
data was never the limit; hand-cranked per-broadcast staging was.
The human's role shrinks to eye-testing the specific calls the model
is least sure about (active learning), not fixing drafts.

**Proof the pivot is right (courtvision/learn.py, LOMO, all 7
held-out matches):** a numpy random forest trained on the aligned
corpus beats the hand-written rules on both weak components —
- SERVE ZONE: 59.7% vs 31.7% heuristic (+28pp), beats even the
  committed-only baseline (48.9%) by +11pp; confidence is calibrated
  (top-25% -> 71.7%), so "auto-file the confident, ask Trevor about
  the rest" is viable for serve NOW. Clean win, n=479.
- ENDING TYPE: 59.3% vs 22.8% (+36.5pp) — BUT honest catch: 0% recall
  on wide (0/63) and deep (0/86). The forest lumps every out-error
  into "winner"; the win is winner(97%)+net(69%). Root cause is the
  INPUT, not the model or data volume: WASB loses the ball before the
  out-of-bounds bounce, so wide-vs-deep isn't in the tracks. Fixing
  endings needs bounce-through-tracking, not more matches.
Reviewer confirmed: no leakage (heuristic outputs never fed as
features), true leave-one-match-out, apples-to-apples baseline on
identical held-out points, deterministic (seed=0, bit-identical
reruns), additive-only, suite 80 green.

**Dead end, kept (experiments/letter_anchor_cf.py, REFUTED):** the
counterfactual test of body-anchored letter references vs the shipped
blob-center read — ref1 (foot_x) is byte-identical to center-x on all
60,442 player rows (tautology), ref2 (neighbor-median) moves the
reference but flips ZERO committed letters, ref3 (abstain on smears)
only loses right answers (t4 -3, held-out t3 -18). Baseline ref0
reproduced the shipped 67/85 (t3) and 17/31 (t4) exactly. The lesson
reinforces the pivot: stop hand-tuning the letter rule — train shot
type from the corpus instead. (experiments/wasb_recall_gated.py, the
confidence-gated crossing-recall successor to the refuted blind
step=1, is included as history; its measurement was inconclusive and
is superseded by the training approach.)

- New (committed): courtvision/learn.py (the training harness),
  experiments/learn_components.py, experiments/letter_anchor_cf.py,
  experiments/wasb_recall_gated.py. NOT committed here (still being
  produced by the running ingestion workflow): courtvision/ingest.py,
  data/corpus/, the first auto-ingested new match. Shipped pipeline,
  charts, exports, model, benchmark: untouched.
- Session cost: $0.00. Project total: ~$16.

## 2026-07-20 — Corpus pulled, ingestion pipeline proven: labels are infinite, video is the constraint

**The full Match Charting Project corpus is now local** (data/corpus/,
gitignored — 189MB public CSVs): 11,646 matches, 1,853,115 charted
points, ~9.4M shots. Turning a chart into per-shot training labels
needs NO video — proven on a brand-new match (Swiatek-Paolini, RG F
2024): 88 points -> 494 labeled shot rows.

**The honest scaling finding (courtvision/ingest.py, reviewer-clean):**
the LABEL half of a training pair is free and effectively infinite;
the FEATURE half (ball track, player boxes, court coords) exists only
after the video is staged and clips are joined to chart rows. Video
availability: recent finals ~100% findable but mostly as SHORT
highlights that don't align 1:1 to a full chart; full/condensed
matches exist for the marquee head, thin fast, absent for the long
tail. So the bottleneck is FEATURES (video + staging), not labels.

**Staging is only partly by-eye — better news than feared.** The three
knobs: (1) court-colour HSV band + a static fit-window; (2) score-bug
crop config; (3) clip<->chart alignment. Crucially, the HSV band
PORTED directly from t3 (also Roland Garros) to the new RG match and
produced a clean court mask — so the colour band is transferable
within a broadcaster family, not a per-match hand-tune. The genuine
last-mile is reading the score bug to build the alignment. Path to
scale: a small library of ~6-8 broadcaster bands + an automated
score-bug reader would cover the findable head.

**ingest.py automates end-to-end today:** find match -> extract labels
(video-free, scales to 1.85M now) -> find+rank video (correctly
preferred the 84-min full match over the 3-min highlight) -> download
-> normalise fps -> scaffold match yaml -> report which knobs remain.

- Committed: courtvision/ingest.py, .gitignore (corpus), this LOG.
  Not committed: data/corpus/ (gitignored), the WIP match-8 scaffold.
- Session cost: $0.00 (light proofs only; heavy run was deferred to
  spare the live charting session, now lifted). Project total: ~$16.

---

## cv-18 · Track A closed: benchmark match #8 (Swiatek–Paolini, RG-F 2024, clay) — auto score-bug alignment is the unblock

Took the deferred heavy run all the way to an evaluated match #8, fully
automatic wherever the pipeline allowed. Verdict: **aligned-and-evaluated**,
but the last-mile human knob is STILL fitcourt corners, and the chart
accuracy is clay-poor.

Chain, with the honest per-stage verdict:
- **download** AUTO — ingest.find_video top hit `i7Kfe0JaGdw` (84-min full
  match), yt-dlp 720p → `clips/g1_swiatek_paolini_30fps.mp4` (30fps h264).
- **fitcourt** MANUAL — auto window pick worked
  (`experiments/g1_scorebug/fitwindow.py`: lowest-drift court-view run, frame
  137880, 0.61px drift), but **auto_fit FAILED** with the t3-ported clay band
  (rms 74px, visibly skewed overlay — the RG world-feed's faint far baseline +
  reddish stands defeat the horizontal assignment). By-eye doubles corners
  (symmetric about center x=618) → clean fit, all residuals ≤7px. **The
  MANUAL 1/3 knob survives; a human still places 4 corners.**
- **SCORE-BUG READ** AUTO — the headline. `read_bug.py` reads the RG-2024 WTA
  bug by anchoring on the light points-box (dark-on-light), games digit
  (white-on-green immediately left), completed-set columns, and the "//" serve
  marker. `sweep3.py` two-pass (cheap pixel plateaus → OCR one frame per
  plateau, bounded to ~374 OCR calls) over the whole 84-min reel. `match_mcp.py`
  LCS-aligns the observed key-sequence to the 88 MCP points (greedy-first was
  sabotaged by OCR-glitch 0-0s grabbed out of order; LCS finds the optimal
  monotonic match, resolving deuce recurrence by order like align.py).
  **Result: 65/88 MCP points auto-aligned (74%), server-bug read 64/65 correct
  on matched points.** Shipped `align` then matched **65/65 clips 1:1** →
  `g1_mcp_map.csv`, monotonic. Misses are dominated by game-start 0-0 points
  the broadcast never dwells on, plus residual set-2 games-digit OCR noise.
  **This is the scaling unblock: the "MANUAL 3/3" by-eye transcription is now
  machine-written.**
- **players / track-ball / serve / chart / eval** AUTO on a 24-point subset —
  players 100% coverage; WASB clay tracks thin (8–41%, holes); chart 20 pts.
- **eval scorecard (match #8, 20 pts):** server-end 11/20, rally-len±1 7/20,
  serve-zone 5/7, letters-aligned 6/8, ending 2/13, **acceptance ≤1edit 0/20.**
  Clay-poor, as t3 warned: thin WASB tracks + NO clip_offsets (RG camera
  wander uncorrected, I skipped the probe/offsets stage) + a single manual
  homography. The loop CLOSED; the accuracy did not.

Lessons / dead-ends:
- OCR sweep died twice at ~13min on inline per-frame OCR — the leak was the
  long pre-match intro + replays generating false-positive OCR fires. Fix:
  ffmpeg-crop the bug region at 5fps (123× realtime), then cheap pixel-plateau
  segmentation with OCR bounded to one frame per plateau. Decouple segmentation
  from reading.
- points OCR: psm-8 reads "AD"; psm-7 fails on it — try 8 then 7. Repair the
  invalid tennis-points vocabulary at match time ("25"→"15", "320"→"30", …)
  off the raw plateau CSV — no re-sweep needed.
- The score bug is broadcaster-family-stable: this RG-2024 WTA layout is a
  cousin of t3's RG-2023 world feed, but the columns/eras differ enough that
  the crop windows are per-feed. The READER generalizes; the geometry doesn't.

## 2026-07-20 — Track B: the video-free notation prior (1.85M points, no pixels)

The other half of the scale run: how much of a chart is predictable
from NOTATION CONTEXT ALONE, trained on the corpus, tested on
match-disjoint held-out matches (experiments/notation_prior.py; sklearn
+pandas via `uv run --with`; reviewer reproduced byte-identical, no
leakage, match-disjoint split verified).

- NEXT-SHOT SIDE (fh/bh): 72.9% vs 50.9% base (+22pp), balanced recall
  — the big win. Side is exactly what the pixel letter-reader fumbles,
  so a confident context prior can override an ambiguous fh/bh call.
- ENDING TYPE: 43.5% vs 33.1% base (+10pp). Winner recall 79% (gates
  the pixel classifier's one working axis). But wide 17% / deep 26% —
  barely above base.
- NEXT DIRECTION (1/2/3): 46.4% vs 38.1% base (+8pp).

The honest limit, and it lines up with Track A's pixel finding: the
wide-vs-deep split the ball tracks are BLIND to is ALSO largely
unpredictable from notation context. A prior can't substitute for the
missing bounce signal. Two independent lines of evidence now say the
same thing — wide/deep needs ball-tracking through the out-of-bounds
bounce, not more data and not a cleverer prior.

Use: multiply each prior P(token|context) against the pixel likelihood
to re-rank Bayesian-style — biggest payoff on next-side and
winner-gating. New: experiments/notation_prior.py (+ report in
outputs/diag/, gitignored).
- Session cost: $0.00. Project total: ~$16.

## 2026-07-21 — Sharpening the axe: the deep-research pass and the Blueprint

**Strategic session, on Trevor's steer: stop chopping, sharpen.** No pipeline
code changed. Two asks: (1) a holistic view — deep research on computer-vision
best practices, because "it's not really clear why you're doing what you're
doing"; (2) a devlog reset — ELI5 shouldn't be a paragraph per post, it should
BE the post. Behind it: a sophisticated, modular, understandable model hardened
enough to build an MVP business on.

**What ran:** one research workflow, 41 agents. 3 mapped current state (all 33
package modules, all 17 posts + primer, benchmark docs). 8 ran technical
deep-dives (ball, court, players/pose, event spotting, sequence models,
training strategy, hardening, foundation models). 14 load-bearing
recommendations were then attacked by paired adversarial verifiers — one
fact-checking citations against fetched sources, one refuting feasibility
against this repo/Mac/budget (several RAN the code: rtmlib measured at 87 fps
on Apple Silicon feeding our existing boxes; librosa onset detection ran on a
staged match in seconds). Verdicts: 12 evidence-CONFIRMED, 2 PLAUSIBLE, 0
refuted. A completeness critic flagged what the dives missed (licensing,
full-match segmentation, scope contract, review-loop economics).

**The deliverable: docs/blueprint.html** — the machine as a picture (5
stations, every block graded green/yellow/red/manual), station-by-station
upgrades with receipts, a 12-item roadmap where items 1–6 are independent
days-scale builds, a risk register, and the new devlog contract.

**The three reframes worth remembering:**
1. **You don't need to see the ball land.** Wide-vs-deep was framed as a
   tracking problem; the field says infer the landing spot from the flight
   path (Hawk-Eye's own principle — "Where Is The Ball," CVPR-W 2025: 0.63 m
   landing error on broadcast tennis) and get the bounce MOMENT from pixel
   event spotters (E2E-Spot: 96 mAP@1 on broadcast tennis, pretrained tennis
   checkpoints, BSD, bounce classes included) or audio. Physics + spotter +
   single-frame heatmap replaces "perfect tracking."
2. **Few labels favors stick figures.** UMEG-Net (AAAI 2026): skeleton+ball+
   court-corner graph models beat raw-video models 2–3x in the few-shot
   regime — and the graph's input nodes are exactly the modules we already
   compute. The modular architecture is the sample-efficient choice, not a
   compromise.
3. **The corpus becomes training data the moment it's pinned to the clock.**
   MCP charts are transcripts; score-bug alignment (74% auto) + shot-to-event
   snapping mints frame-level labels at scale (the soccer MatchTime/ATBA
   playbook). Confirmed guardrail: video LLMs stay OUT of the per-frame loop
   (GPT-4o 57.8% on hard sports-video tasks; ~1 fps sampling; $12–50/pass vs
   ~$0 local).

**Dead ends / corrections, on the record:**
- First research pass called TT3D's code CC BY-SA — verifier corrected it
  (the license is the dataset's). Check repos, not summaries.
- "Court net deletes the last manual knob" — overstated; HSV band + fit
  window survive as per-match config. Knob shrinks, doesn't vanish.
- The fusion decoder's inputs don't exist in per-token shape yet: the
  notation prior is three task classifiers (side/direction/ending), and
  letters/directions emit decisions, not probabilities. Retraining into
  probability form is real work the roadmap now owns.
- Our own shorthand had drifted: the shipped confidence model is a LOGISTIC
  REGRESSION (the random forest is the unshipped corpus prototype), and the
  score-bug OCR lives in experiments/, not the package. The blog audit caught
  the author's memory disagreeing with the author's code — half the case for
  the ELI5 reset in one anecdote.
- License landmines mapped: MCP corpus is CC BY-NC-SA (existential for the
  commercial path — talk to Sackmann before revenue); TennisCourtDetector,
  F3Set, TennisProject ship NO license (email NUS for F3Set); Ultralytics
  YOLO is AGPL (avoided); Sapiens is non-commercial (avoided). Chosen stack
  is permissive throughout: WASB MIT, rtmlib Apache, E2E-Spot BSD, ByteTrack
  MIT, MAPIE BSD.
- Bonus find from verification: F3Set's annotations ship in-repo with player
  names, handedness, and per-clip scores — those 114 matches need NO score-bug
  OCR to align. Bigger win than the research pass claimed.

**Devlog reset (binding from cv-18):** plain English is the post, not a box in
it. One concept per post; "Previously" recap instead of codename callbacks;
define on first use every post; decode every notation string token by token;
numbers restated in plain terms ("19 of 49 — call it 2 points in 5"); receipts
move to a collapsed appendix; frontmatter written last as a ≤25-word headline;
acceptance test = a cold reader can say what problem, what concept, and
whether it worked, from this post + primer alone. Primer to gain ~15 missing
entries (HSV, Hough, ECC, residuals, F1, logit, AUC, base rate, notation
legend...). The roadmap doubles as the editorial calendar, one concept per
build.

- New: docs/blueprint.html; README links it. Nothing else touched.
- Session cost: $0 API (research ran on session compute). Project total: ~$16.

## 2026-07-21 — Roadmap #1: the boundary race — wide/deep recall off zero (and one refutation, kept)

First chop with the sharpened axe. experiments/landing_spot.py implements
the blueprint's landing-spot idea, v1: isolate the FINAL FLIGHT SEGMENT
(contiguous track run after the last shot's contact, flight-capped at
2.0s, cut at the first >6-frame hole), fit straight lines cx(t)/cy(t) to
the tail in court coordinates, and RACE two crossings — sideline first
means wide, baseline first means deep, both-within-2-frames means x,
neither within a 0.7s extrapolation cap means abstain. Pure geometry,
zero training, no constants fitted to truth (windows chosen a priori
from flight time; margins reuse endings.py's OUT_MARGIN).

**Headline — the 0% is broken.** On the 169 benchmark points whose true
ending is w/d/x (population per learn_components + g1, truth =
mcp_ending_type):
- wide: strict 30/69 (43%), called-any-out 56/69 (81%)
- deep: strict 35/90 (39%), called-any-out 65/90 (72%)
- when the racer commits to w-or-d (not x/abstain), it separates the two
  at 70% (65/93; balanced 71%) — genuine discrimination, coin is 50%.
Both the shipped rules and the pivot forest score 0% on these classes
(baseline reproduced this run: 0/69, 0/90).

**Refuted, kept: naive fusion.** Stage 2 appended the racer's outputs
(call one-hot, t_side, t_base, n_seg) to the SAME 4-class LOMO forest
(learn_components, seed 0). Result: wide 0/69 -> 0/69, deep 0/90 ->
2/90, overall 59.0% -> 55.5%. Diagnosis: base-rate pull — winners are
45% of points and the racer false-fires on 62% of */n truths (the
shadow-depth inflation makes in-balls extrapolate out), so a flat
accuracy-driven forest keeps buying majority classes. The racer's
signal is CONDITIONAL: P(wide vs deep | out), not P(out).

**The architectural lesson (feeds roadmap #6/#7):** decide OUT-ness from
independent evidence (net-death, an observed landing, later the pixel
event spotter + audio), then let the racer name the flavor. That is
exactly the per-slot likelihood shape the grammar-constrained fusion
decoder wants. Do not ask one flat classifier to do both jobs — that
experiment is now on the record twice.

**Clay confirms the blueprint's coupling:** per-match strict-correct on
true outs — t7 27/58, t6 20/44, t5 12/29, t4 6/14 vs t3 1/10, g1 1/11.
Thin clay tracks leave no final segment to fit; the WASB clay fine-tune
(roadmap #10) multiplies this module for free.

Known v1 biases, stated not tuned: depth inflation fires the baseline
crossing early (d-bias on airborne balls); the 2-frame "both" window
over-calls x on true wides (18/69). Both belong to the racer's
confusion matrix, which the fusion decoder can consume as-is.

- New: experiments/landing_spot.py (stage 1 racer + stage 2 LOMO A/B);
  report at outputs/diag/landing_spot_report.txt (gitignored). Shipped
  pipeline untouched; suite 80 green. Run:
  PYTHONPATH=.:experiments uv run python experiments/landing_spot.py
- Session cost: $0. Project total: ~$16.

## 2026-07-22 — Roadmap #1 lands v1.1, the hero render catches two bugs, cv-18 ships in the new voice

**The racer grew up under the camera's gaze.** Rendering the cv-18 hero
video (experiments/render_landing_hero.py — two US Open points, the
final flight frozen, the extrapolation racing the sideline vs the
baseline) caught two real defects the scoreboard had absorbed:
1. **Teleporting tracks.** t6_point_110's last "detection" jumped 37 m
   in 3 frames — the tracker latching onto junk after the real ball
   left frame, poisoning the physics fit. Fix: end the flight segment
   at the first physically impossible jump (>3.0 m/frame court motion,
   ~90 m/s — a physics bound, not a tuned constant; boxes.py already
   does this for players).
2. **The striker column lies.** t6_point_117 charts its last striker as
   'far' while the ball demonstrably died 2 m beyond the FAR baseline.
   The racer now infers the target half from the flight itself
   (already-out position wins, else y-slope sign) and trusts nothing
   upstream. Eyes beat derivatives, again — both bugs were invisible in
   the tables and obvious in one rendered frame.

**v1.1 numbers (final, on the record; three variants measured, all
kept):** striker-target/no-gate: w 30/69, d 35/90 (41% overall strict).
+teleport gate: w 31/69, d 33/90 (41%). +flight-inferred target:
**w 20/69, d 55/90, x 8/10 — 83/169 (49%) strict; loose any-out w 84% /
d 86%; committed w-vs-d discrimination 75/90 (83%)**. The deep jump and
wide dip are the same shadow-depth inflation now concentrating in the
race window (27 true wides call 'x'); v2 owns that bias. Stage-2 naive
forest fusion stays refuted (0-1% w/d) — the racer is P(flavor | out),
to be consumed by the fusion decoder behind independent out-ness
evidence.

**The devlog reset shipped.** cv-18 ("Nobody Sees the Ball Land") is
the first post where plain English IS the post — video-first hero
(clips + racing overlay), one concept, Previously recap, decoded
numbers, receipts in a collapsed appendix. cv-01–cv-17 retrofitted to
the same voice by a 17-agent pass against a written brief (facts
preserved verbatim, insider density relocated to per-post appendices),
and /primer gained ~9 new entries incl. the full notation legend and
the boundary race. Blueprint (docs/blueprint.html) + this experiment
committed and pushed with the posts.

- New/changed: experiments/landing_spot.py (v1.1), 
  experiments/render_landing_hero.py, docs/blueprint.html (yesterday's
  axe-sharpening), README link, this LOG. Suite 80 green.
- Session cost: $0. Project total: ~$16.

## 2026-07-22 — Roadmap #4: the machine grows ears (audio hits v1, t6+g1)

**Staging find #1:** every reel on disk is SILENT — the ingest pipeline
always downloaded video-only streams, and extract.py cuts clips with
-an. Ears required re-fetching audio-only tracks of the SAME uploads:
g1 by known ID (duration 5039.40s vs reel 5039.45s — same file), t6 by
duration fingerprint (condensed-match upload, 2412.22s vs 2412.13s).
t3's edit didn't surface in search; later. New: clips/audio/ (gitignored).

**Staging find #2, recovered:** g1's clip cutter never persisted its
reel windows (the score-bug pipeline is still experiments/-grade). Its
geometry was reverse-engineered: every clip measures exactly 1.6s
longer than its alignment window (verified points 01/02/05/10), so
start = f0/30 − a constant pre-pad, and any constant guess folds into
the per-match A/V offset. Lesson for the package promotion: cutters
must persist their windows.

**experiments/audio_hits.py (numpy STFT spectral flux, 1-8 kHz, 11.6ms
hops; THRESH swept on t6 per declared protocol, frozen at 5.0 for g1):**
- A/V offset locks: +95ms (t6), +310ms (g1) — sharp single-offset
  peaks, i.e. the fingerprint-matched audio really is the same edit.
- Agreement with video hits: t6 44% within 2 frames / 57% within 3;
  g1 49% / 66% — the CLAY match agrees MORE (closer court mics?).
- THE HEADLINE: onsets inside ball-track holes — 156 events in 67s of
  blind time (t6), 196 in 78s (g1). The mic keeps working exactly
  where the eyes fail.
- The physics receipt: sweeping THRESH shows onsets-per-true-shot
  median hitting exactly 2.00 at THRESH=12 (racquet + bounce per
  shot). At the frozen 5.0 the ratio is 3.27 (crowd/calls/grunts ride
  along) and naive rally length (onsets/2) is only 30% within ±1 —
  audio v1 is a WITNESS, not yet a counter. v2 is the Sony-AI-pattern
  mel-CNN classifying racquet vs bounce vs other, trained on pseudo-
  labels minted from aligned video hits (blueprint roadmap #8).
- Session cost: $0. Project total: ~$16.

## 2026-07-22 — Roadmap #2: the court fits itself (7/8 plates, incl. the clay that broke auto-fit)

**experiments/court_autofit.py:** the public pretrained 14-keypoint
TennisCourtDetector (third_party/, no license — research use, blueprint
risk register) run on each match's fit-window plate; homography solved
via the author's 12-configuration cross-validated search (reimplemented
numpy-only); converted to our image→meters convention; graded against
the 8 HAND-FIT homographies at 6 landmarks (doubles corners + net posts).

**Results (landmark delta vs hand truth, mean/max px):**
t5 2.7/3.1 · t6 2.4/4.0 · t7 4.7/8.6 · t2 7.3/13.9 · t4 8.0/15.2 ·
**g1 8.0/12.2 — the clay plate classical auto-fit missed by 74 px rms;
the pretrained net fits it first try** · t3 15.7/29.0 (see below) ·
t1: 2/14 keypoints detected → NO FIT — the detector ABSTAINS on the
night feed, which is exactly the accept/reject gate behavior the
blueprint wants (abstain → human clicks corners).

**Open question, kept honest:** the t3 overlay shows the NET's orange
lines sitting on the painted doubles paint while the hand-fit green is
visibly inset on the far side — the 15.7 px "error" may partly be the
hand fit's. Arbitration belongs to the line-mask judge (fitcourt's
step-5 scorer) when the gate is wired; do not assume the hand fits are
perfect truth.

**Dead end caught by eyes (kept):** first run showed all fits
displaced by a consistent ~750 px — their postprocess() upscales
heatmap coords by 2 internally (assumes 720p input) and I scaled
again. One overlay render found it instantly; the metric alone never
named the cause. Eyes beat derivatives, third time this week.

Next for the knob: wire the gate (auto-init → line-mask judge →
accept/abstain), try per-clip plates or gamma boost for the night
feed, and retrain a licensed net on self-labeled frames before the
business phase (weights license: none).
- New: experiments/court_autofit.py; third_party/TennisCourtDetector
  clone + weights (gitignored). Session cost: $0. Total: ~$16.

## 2026-07-22 — Roadmap #3: skeletons for the smudges (pose v1.1, t3+t4)

**The week-one gate from the blueprint, run:** rtmlib (Apache-2.0,
auto-downloaded ONNX, CPU) posed at every graded contact frame on the
two letters-benchmarked matches. Verdict: PASS with caveats.

**Numbers (MCP truth letters, length-matched clips, identical shots per
method):**
- t3 clay: ball-side-of-torso (pose anchor) 42/50 (84%) vs shipped blob
  rule 36/44 (82%) — parity, exactly what the refuted re-anchoring
  counterfactual predicted. Wrist-cross 37/50 (74%).
- t4 grass (the blob rule's disaster surface): pose ball-side 23/35
  (66%) vs blob 16/30 (53%) — +13pp where the boxes smear worst.
- Far-player keypoint confidence: shoulders 0.61-0.65, wrists
  0.45-0.57 — shaky wrists, usable shoulders, exactly the research
  prediction at ~90px player height.

**Bug caught by the audit stills (kept, with the frame):** v1 matched
detections to the striker by nearest-blob-center, and a rogue blob box
(the known box disease) dragged the match to a BALL KID at the back
fence — outputs/diag/pose_blooper_ballkid.png. Some v1 shots graded
the wrong human's skeleton and pose STILL beat the blob on grass.
Fourth eyes-beat-derivatives of the week.

**v1.1 fix + its honest cost:** candidates must stand in the striker's
court half (ankles projected through the homography); highest
confidence wins; the blob box demoted to tiebreak hint. Identity now
clean — but coverage halved (t3 93→50 graded shots) because bad ankles
fail the court test with no fallback. v2 recovers coverage with
temporal tracking (pose every frame, continuity), and the real letters
lift remains roadmap #9's keypoint-graph model over shot windows —
single-frame geometry was only ever the gate test.

- New: experiments/pose_letters.py; audits in outputs/diag/pose_*.png.
- Session cost: $0. Project total: ~$16.

## 2026-07-23 — The scorecard becomes the site's front door

Trevor's design, workshopped live: kill the stat tiles (only "how much
we've tested" survives), every component becomes a CARD — a looping
clip of what the machine is looking at, a plain question, a grade chip
(Trust/Verify/Re-key), the number, and a trend line with a dot per
grading run and date labels. trmccormick.com's homepage now IS this
grid (hero carousel retired), with the devlog wall below.

Plumbing: gen_scorecard.py now also emits docs/scorecard.json and
appends each run to data/scorecard_history.json (backfilled ONLY with
documented benchmark figures — some lines have 4 dots, some 1; uneven
is honest). The site reads the JSON via scripts/sync-scorecard.sh.
Three purpose-made card loops rendered (serve ring, direction arrow,
service-box thirds + landing dot: experiments/render_scorecard_clips.py);
five more are hero-video cuts pending purpose-made versions. Clips
live on the site, NOT in this repo (no broadcast video committed).

Caught on the way: the first "usable drafts" tile said 83% — a
denominator lie (aligned-only records); the true bin-derived number is
66%. Fixed with a comment naming the trap. The page whose job is
preventing quiet lies almost shipped one.
- Session cost: $0. Total: ~$16.
