"""Server-half tests for the chart app. This file starts with the
conformance plumbing; Task 7 appends the route tests."""

from pathlib import Path

import courtvision.chartapp as chartapp


def test_conformance_path_points_at_generated_fixture():
    p = chartapp.CONFORMANCE_PATH
    assert p.name == "score_conformance.json"
    assert p.exists() and p.stat().st_size > 10000
