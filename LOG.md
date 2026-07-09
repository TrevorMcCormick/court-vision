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
