"""Court Vision — auto-charting tennis matches from broadcast video.

The living consolidation of the experiments/ era (the scripts stay
frozen as history; divergence policy in the package README): WASB ball
tracking, $0 bgsub player boxes, serve detection v3, score-bug point
boundaries, the v5 crossing-skeleton event detector, the
receiver-mirrored direction estimator, letters, endings, the chart
assembler, MCP alignment + eval + acceptance, and the calibrated
per-point confidence layer that tells a charter WHICH drafted points
to trust.

    uv run python -m courtvision chart t3
    uv run python -m courtvision eval t3
    uv run python -m courtvision draft t3
"""

__version__ = "0.1.0"
