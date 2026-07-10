"""Per-match configuration — the sanctioned per-broadcaster staging knobs.

Every number and flag that legitimately varies between matches lives in
one YAML file per match (data/matches/<id>.yaml); everything else in the
package is frozen shared logic. The flags mirror the divergences the
four t*w experiment twins carried in code:

  lefty                     freeze-#2's one allowed knob (t1 both-lefty)
  clip_offsets              wandering-camera correction (t3/t4 only)
  staging.lock_serve        confident serve call locks the striker chain
                            (measured on the t3/t4 staging pair)
  staging.serve_zone_requires_side
                            no deuce/ad stance read -> no zone claim
  staging.near_ending_fill  near-half V-cusp ending recovery (t3/t4)
  staging.coda_report       n_coda/coda_why columns in the match chart
                            (t4; the in-plateau dead-ball coda)

Paths are stored repo-relative and resolved against the repo root.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MATCH_DIR = ROOT / "data" / "matches"


@dataclass
class Staging:
    lock_serve: bool = False
    serve_zone_requires_side: bool = False
    near_ending_fill: bool = False
    coda_report: bool = False


@dataclass
class EvalCfg:
    mcp_map: Path = None
    alignment: Path = None
    mcp_points: Path = None
    mcp_match_id: str = ""
    title: str = ""
    start_end: str = "far"
    # (set1, set2) -> games-already-played prior for changeover parity;
    # keys may use '*' as a wildcard on either side
    set_priors: dict = field(default_factory=dict)
    # set states in which a 6-6 game is a tiebreak (end swap every 6
    # points): list of "s1,s2" strings, the string "any", or [] for none
    tiebreak_states: object = field(default_factory=list)

    def prior(self, s1, s2):
        for key in (f"{s1},{s2}", f"*,{s2}", f"{s1},*"):
            if key in self.set_priors:
                return int(self.set_priors[key])
        return 0

    def is_tiebreak_state(self, s1, s2):
        if self.tiebreak_states == "any":
            return True
        return f"{s1},{s2}" in (self.tiebreak_states or [])


@dataclass
class MatchConfig:
    id: str
    title: str
    match: str
    clips_dir: Path
    out_dir: Path
    ball_dir: Path
    players_dir: Path
    charts_dir: Path
    homography: Path
    serves: Path
    clip_offsets: Path            # or None: camera holds, no correction
    lefty: dict
    staging: Staging
    eval: EvalCfg
    serve_detect: dict
    players_detect: dict

    def clip_path(self, stem):
        return self.clips_dir / f"{stem}.mp4"

    def load_offsets(self):
        """{clip: (dx, dy)} from the probe's shift search, or {}."""
        import csv
        if self.clip_offsets is None or not self.clip_offsets.exists():
            return {}
        return {r["clip"]: (float(r["dx"]), float(r["dy"]))
                for r in csv.DictReader(open(self.clip_offsets))}

    def load_serves(self):
        import csv
        return {r["clip"]: r for r in csv.DictReader(open(self.serves))}

    def ball_stems(self):
        """Clips that have a ball track — the chartable set, sorted."""
        return sorted(p.stem.replace("ball_", "")
                      for p in self.ball_dir.glob("ball_*.csv"))


def load(match_id):
    """Load data/matches/<match_id>.yaml into a MatchConfig."""
    path = MATCH_DIR / f"{match_id}.yaml"
    raw = yaml.safe_load(open(path))
    out_dir = ROOT / raw["out_dir"]
    ev = raw.get("eval", {})
    return MatchConfig(
        id=raw["id"],
        title=raw.get("title", raw["id"]),
        match=raw.get("match", ""),
        clips_dir=ROOT / raw["clips_dir"],
        out_dir=out_dir,
        ball_dir=out_dir / raw.get("ball_dir", "ball_wasb"),
        players_dir=out_dir / raw.get("players_dir", "players"),
        charts_dir=out_dir / raw.get("charts_dir", "charts_wasb"),
        homography=out_dir / raw.get("homography", "H_img_to_court.npy"),
        serves=out_dir / raw.get("serves", "serves.csv"),
        clip_offsets=(out_dir / raw["clip_offsets"]
                      if raw.get("clip_offsets") else None),
        lefty={"near": bool(raw["lefty"]["near"]),
               "far": bool(raw["lefty"]["far"])},
        staging=Staging(**raw.get("staging", {})),
        eval=EvalCfg(
            mcp_map=ROOT / ev["mcp_map"] if ev.get("mcp_map") else None,
            alignment=ROOT / ev["alignment"] if ev.get("alignment") else None,
            mcp_points=ROOT / ev["mcp_points"] if ev.get("mcp_points") else None,
            mcp_match_id=ev.get("mcp_match_id", ""),
            title=ev.get("title", raw.get("title", raw["id"])),
            start_end=ev.get("start_end", "far"),
            set_priors=ev.get("set_priors", {}) or {},
            tiebreak_states=ev.get("tiebreak_states", []),
        ),
        serve_detect=raw.get("serve_detect", {}),
        players_detect=raw.get("players_detect", {}),
    )


def match_ids():
    return sorted(p.stem for p in MATCH_DIR.glob("*.yaml"))
