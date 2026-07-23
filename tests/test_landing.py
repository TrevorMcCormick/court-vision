"""Landing-spot boundary racer — synthetic flights with a known verdict.

Court coords (meters): far baseline y=0, near baseline y=L_C=23.77, net at
y=11.885, singles sidelines x=1.372 and x=9.598. Each flight is a clean
straight run after a contact at frame 100; the racer fits the tail and
races the sideline crossing against the baseline crossing.
"""

import numpy as np

from courtvision import endings, landing
from courtvision.court import NET_Y


def _track(cx_end, cy_end, cx0=5.5, cy0=6.0):
    """A 15-frame flight from (cx0,cy0) at f=101 to (cx_end,cy_end) at
    f=115, with 10 pre-contact frames the racer must ignore."""
    frames = np.arange(90, 116)
    cxc = np.concatenate([np.full(11, cx0),
                          np.linspace(cx0, cx_end, 15)])
    cyc = np.concatenate([np.full(11, cy0),
                          np.linspace(cy0, cy_end, 15)])
    return frames, cxc, cyc


def test_wide_flight_calls_w():
    # sails out the far-right sideline (x past 9.598), depth barely changes
    frames, cxc, cyc = _track(cx_end=10.6, cy_end=5.0, cy0=5.0)
    assert landing.race_final_shot(100, frames, cxc, cyc) == "w"


def test_deep_flight_calls_d():
    # sails long over the far baseline (y through 0), stays centered in x
    frames, cxc, cyc = _track(cx_end=5.5, cy_end=0.4, cy0=3.5)
    assert landing.race_final_shot(100, frames, cxc, cyc) == "d"


def test_in_flight_abstains():
    # decelerated, crosses no boundary within the extrapolation cap
    frames, cxc, cyc = _track(cx_end=5.6, cy_end=5.9, cy0=6.0)
    assert landing.race_final_shot(100, frames, cxc, cyc) is None


def test_too_short_segment_abstains():
    frames = np.array([98, 99, 100, 101, 102])   # only 2 post-contact samples
    cxc = np.full(5, 5.5)
    cyc = np.full(5, 6.0)
    assert landing.race_final_shot(100, frames, cxc, cyc) is None


def test_endings_gate_off_by_default():
    """A wide flight with no recorded landing stays '?' unless the flag
    is on — the racer must never fire on its own."""
    frames, cxc, cyc = _track(cx_end=10.6, cy_end=5.0, cy0=5.0)
    ys = np.zeros_like(frames, dtype=float)
    shots = [{"frame": 100, "is_serve": False, "landing_y": None}]
    off = endings.infer(shots, frames, ys, cyc, cxc, 30.0, landing_race=False)
    on = endings.infer(shots, frames, ys, cyc, cxc, 30.0, landing_race=True)
    assert off == "?"
    assert on == "w@"


def test_endings_gate_never_overrides_real_landing():
    """With a real far-half landing present, the racer stays out of it."""
    frames, cxc, cyc = _track(cx_end=10.6, cy_end=5.0, cy0=5.0)
    ys = np.zeros_like(frames, dtype=float)
    # landing_y in [far baseline .. net] and landing_x in-bounds -> '*'
    shots = [{"frame": 100, "is_serve": False,
              "landing_y": 5.0, "landing_x": 5.5}]
    got = endings.infer(shots, frames, ys, cyc, cxc, 30.0, landing_race=True)
    assert got == "*"


def test_infer_target_far_reads_slope():
    assert landing.infer_target_far(y0=6.0, by=-0.5) is True      # heading far
    assert landing.infer_target_far(y0=18.0, by=0.5) is False     # heading near
    assert landing.infer_target_far(y0=-1.0, by=0.0) is True      # already out far
