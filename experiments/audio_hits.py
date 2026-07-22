"""Audio hit detection v1 — the machine grows ears (blueprint roadmap #4).

HYPOTHESIS. The ball tracker's eyes blink at exactly the wrong moments
(track holes, lost out-balls). Broadcast microphones don't: the racquet
impact is a sharp broadband transient, and audio timing (11.6 ms hops)
is finer than a video frame (33 ms). An onset detector on the soundtrack
is an independent witness for the event layer — one that keeps working
when the track has holes.

METHOD (v1, pure numpy, no new deps).
- Audio: the source video's own soundtrack, re-fetched audio-only via
  yt-dlp (the staging pipeline always downloaded video-only). Identity
  verified by duration fingerprint: g1 5039.40s vs reel 5039.45s (same
  upload, known ID); t6 2412.22s vs 2412.13s (condensed-match upload
  found by search, duration-exact).
- Onsets: STFT (1024/256 hann, 22.05 kHz mono), band-limited 1-8 kHz,
  log-magnitude spectral flux, normalized by a sliding median/MAD
  (~2 s), peaks above THRESH with 0.1 s min separation.
- A/V offset: one constant per match (audio and reel share a timeline;
  container offsets don't). Grid search over ±0.5 s maximizing video
  hits matched within 40 ms — self-verifying: a wrong audio file or a
  drifting timeline cannot produce a sharp single-offset peak.
- THRESH is picked on t6 (the calibration match, best video hits) and
  FROZEN before grading g1 — the repo's tune-on-one-match rule.

QUESTIONS GRADED.
1. Agreement: after the offset, what fraction of video-detected hits
   have an audio onset within 1/2/3 video frames?
2. Independence: how many onsets land inside ball-track HOLES — moments
   the eyes recorded nothing?
3. Shot counting vs MCP truth: onsets after the serve vs the human
   chart's shot count. Expect ~2 impacts per shot (racquet + bounce);
   the ratio distribution is itself a finding, reported raw.

Run:  PYTHONPATH=.:experiments uv run python experiments/audio_hits.py
"""

import csv
import wave

import numpy as np

from courtvision import config
from courtvision.config import ROOT
from courtvision.mcp import parse_mcp, mcp_point_tokens

SR = 22050
N_FFT, HOP = 1024, 256
BAND = (1000.0, 8000.0)          # Hz — racquet impact transient range
NORM_WIN_S = 2.0                 # sliding median/MAD normalization window
MIN_SEP_S = 0.10                 # peaks closer than this merge (one impact)
THRESH = 5.0                     # flux z-score; calibrated on t6, frozen
OFF_GRID = np.arange(-0.5, 0.5, 0.005)
MATCH_TOL_S = 0.040              # offset-search match tolerance
FPS = 30.0

MATCHES = ["t6", "g1"]           # audio staged for these (see module doc)


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return x


def spectral_flux_onsets(x_i16):
    """Band-limited log spectral flux -> (times_s, zscores) of peaks."""
    x = x_i16.astype(np.float32) / 32768.0
    n = (len(x) - N_FFT) // HOP
    if n < 4:
        return np.array([]), np.array([])
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(N_FFT).astype(np.float32)
    S = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    L = np.log1p(S[:, band])
    flux = np.maximum(L[1:] - L[:-1], 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])

    # sliding median/MAD normalization (strided windows, decimated)
    w = max(3, int(NORM_WIN_S * SR / HOP) | 1)
    pad = np.pad(flux, w // 2, mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(pad, w)
    med = np.median(sw, axis=1)
    mad = np.median(np.abs(sw - med[:, None]), axis=1) + 1e-6
    z = (flux - med) / mad

    sep = max(1, int(MIN_SEP_S * SR / HOP))
    peaks = []
    for i in range(1, len(z) - 1):
        if z[i] >= THRESH and z[i] >= z[i - 1] and z[i] >= z[i + 1]:
            if peaks and i - peaks[-1] < sep:
                if z[i] > z[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    t = (np.array(peaks) * HOP + N_FFT / 2) / SR
    return t, z[peaks] if peaks else np.array([])


def match_data(mid):
    """Per-clip: reel window, video hit times (clip-rel s), serve_s,
    true shot count, ball-track frames."""
    cfg = config.load(mid)
    seg_csv = cfg.out_dir / "segments_v2.csv"
    if seg_csv.exists():
        segs = {int(s["seg"]): {"start_s": int(s["start_frame"]) / FPS,
                                "dur_s": float(s["dur_s"])}
                for s in csv.DictReader(open(seg_csv))}
    else:
        # g1 layout: the clip cutter's windows weren't persisted, but its
        # geometry is recoverable — every clip measures exactly 1.6 s
        # longer than its alignment window (f0..f1), so start = f0/FPS
        # minus a constant pre-pad. The pre/post split is unknown; any
        # constant guess folds into the per-match A/V offset. Verified on
        # points 01/02/05/10 (pad 1.60-1.67 s, within a frame or two).
        PAD_PRE, PAD_TOTAL = 0.8, 1.6
        segs = {}
        align = csv.DictReader(
            open(ROOT / "data" / "mcp" / f"{mid}_clip_alignment.csv"))
        for r in align:
            if not r.get("f0"):
                continue
            k = int(r["clip"].rsplit("_", 1)[1])
            f0, f1 = int(r["f0"]), int(r["f1"])
            segs[k] = {"start_s": f0 / FPS - PAD_PRE,
                       "dur_s": (f1 - f0) / FPS + PAD_TOTAL}
    serves = cfg.load_serves()
    mcp = {r["clip"]: r for r in
           csv.DictReader(open(ROOT / "data" / "mcp" / f"{mid}_mcp_map.csv"))
           if r.get("status") == "matched"}
    clips = {}
    for k, s in segs.items():
        clip = f"{mid}_point_{k:02d}"
        entry = {"start_s": s["start_s"], "dur_s": s["dur_s"],
                 "hits": [], "serve_s": None, "true_shots": None,
                 "track_frames": None}
        ch = cfg.charts_dir / f"chart2_{clip}.csv"
        if ch.exists():
            rows = list(csv.DictReader(open(ch)))
            entry["hits"] = [float(r["contact_frame"] or r["frame"]) / FPS
                             for r in rows if not r.get("synth") == "True"]
        srow = serves.get(clip)
        if srow and srow.get("serve_s"):
            try:
                entry["serve_s"] = float(srow["serve_s"])
            except ValueError:
                pass
        m = mcp.get(clip)
        if m:
            _, _, played = parse_mcp(m.get("first", ""), m.get("second", ""))
            entry["true_shots"] = len(mcp_point_tokens(played)) - 1
        bf = cfg.ball_dir / f"ball_{clip}.csv"
        if bf.exists():
            entry["track_frames"] = np.array(
                [int(r["frame"]) for r in csv.DictReader(open(bf))])
        clips[clip] = entry
    return clips


def run_match(mid, out):
    wav = load_wav(ROOT / "clips" / "audio" / f"{mid}.wav")
    clips = match_data(mid)
    out.append(f"\n## {mid} — {len(clips)} clips")

    # onsets per clip (clip-relative seconds)
    onsets = {}
    for clip, e in clips.items():
        a = int(e["start_s"] * SR)
        b = int((e["start_s"] + e["dur_s"]) * SR)
        t, _ = spectral_flux_onsets(wav[a:b])
        onsets[clip] = t
    n_on = sum(len(t) for t in onsets.values())
    out.append(f"onsets in point windows: {n_on}")

    # constant A/V offset by grid search over pooled video hits
    pairs = [(clip, h) for clip, e in clips.items() for h in e["hits"]]
    best_off, best_n = 0.0, -1
    for off in OFF_GRID:
        n = sum(1 for clip, h in pairs if len(onsets[clip]) and
                np.min(np.abs(onsets[clip] - (h + off))) <= MATCH_TOL_S)
        if n > best_n:
            best_n, best_off = n, off
    out.append(f"A/V offset: {best_off*1000:+.0f} ms "
               f"({best_n}/{len(pairs)} hits matched at 40 ms in search)")

    # 1. agreement at video-frame tolerances
    for tol_fr in (1, 2, 3):
        tol = tol_fr / FPS
        n = sum(1 for clip, h in pairs if len(onsets[clip]) and
                np.min(np.abs(onsets[clip] - (h + best_off))) <= tol)
        out.append(f"  video hits with an onset within {tol_fr} frame(s): "
                   f"{n}/{len(pairs)} ({n/max(1,len(pairs)):.0%})")

    # 2. onsets inside ball-track holes (after the serve, within the rally)
    holes_hit = holes_total_s = 0.0
    n_hole_onsets = 0
    for clip, e in clips.items():
        tf, sv = e["track_frames"], e["serve_s"]
        if tf is None or sv is None or len(e["hits"]) < 1:
            continue
        end_f = max(e["hits"]) * FPS + FPS          # last hit + 1 s
        lo = sv * FPS
        gaps = []
        prev = None
        for f in tf[(tf >= lo) & (tf <= end_f)]:
            if prev is not None and f - prev > 3:
                gaps.append((prev, f))
            prev = f
        for a, b in gaps:
            holes_total_s += (b - a) / FPS
            ta, tb = a / FPS - best_off, b / FPS - best_off
            k = np.sum((onsets[clip] >= ta) & (onsets[clip] <= tb))
            n_hole_onsets += int(k)
            holes_hit += 1 if k else 0
    out.append(f"  onsets inside ball-track holes (>3 frames, in-rally): "
               f"{n_hole_onsets} events across {holes_total_s:.0f}s of "
               f"blind time — heard where the eyes saw nothing")

    # 3. shot counting vs MCP truth
    ratios, close = [], [0, 0]
    for clip, e in clips.items():
        if e["true_shots"] is None or e["serve_s"] is None:
            continue
        t = onsets[clip]
        lo = e["serve_s"] + best_off - 0.15
        # bound by the rally's end where video hits exist — the clip tail
        # is applause and crowd, not tennis
        hi = (max(e["hits"]) + best_off + 1.2) if e["hits"] else e["dur_s"]
        n_audio = int(np.sum((t >= lo) & (t <= hi)))
        r = n_audio / max(1, e["true_shots"])
        ratios.append(r)
        est = max(1, round(n_audio / 2))
        close[0] += 1 if abs(est - e["true_shots"]) <= 1 else 0
        close[1] += 1
    if ratios:
        out.append(f"  onsets-per-true-shot ratio: median "
                   f"{np.median(ratios):.2f} (IQR {np.percentile(ratios,25):.2f}"
                   f"-{np.percentile(ratios,75):.2f}, n={len(ratios)} clips) — "
                   f"~2 would mean 'hears racquet AND bounce'")
        out.append(f"  naive rally length (onsets/2) within ±1 of truth: "
                   f"{close[0]}/{close[1]} ({close[0]/max(1,close[1]):.0%})")
    return onsets, best_off


def calibrate_thresh(out):
    """Declared calibration: sweep THRESH on t6 only, pick by hit-match
    rate at 2 frames with the onset budget shown, freeze for g1."""
    global THRESH
    out.append("\n## THRESH calibration sweep (t6 only, declared protocol)")
    results = []
    for th in (5.0, 8.0, 12.0, 16.0, 20.0):
        THRESH = th
        sub = []
        run_match("t6", sub)
        m2 = next(l for l in sub if "within 2 frame" in l)
        ratio = next((l for l in sub if "ratio" in l), "")
        frac = int(m2.split(":")[1].split("/")[0]) / 645
        results.append((th, frac))
        out.append(f"  THRESH={th:4.1f}: {m2.strip()} | {ratio.strip()}")
    peak = max(f for _, f in results)
    THRESH = max(th for th, f in results if f >= 0.95 * peak)
    out.append(f"  -> frozen THRESH={THRESH} "
               f"(highest threshold within 5% of the peak match rate)")


def run():
    out = ["# Audio hits v1 — the machine grows ears",
           f"(band {BAND[0]:.0f}-{BAND[1]:.0f} Hz; hop {HOP/SR*1000:.1f} ms; "
           "THRESH calibrated on t6, frozen for g1)"]
    calibrate_thresh(out)
    for mid in MATCHES:
        run_match(mid, out)
    report = "\n".join(out)
    print(report)
    dest = ROOT / "outputs" / "diag" / "audio_hits_report.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report + "\n")
    print(f"\n[saved] {dest}")


if __name__ == "__main__":
    run()
