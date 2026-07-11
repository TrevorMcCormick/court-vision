"""Command-line entrypoints.

    uv run python -m courtvision chart <match> [clips...] [--charts-dir D]
    uv run python -m courtvision eval  <match> [--charts-dir D]
    uv run python -m courtvision draft <match>       # chart + confidence + export
    uv run python -m courtvision calibrate           # fit + report the confidence layer
    uv run python -m courtvision export <match>      # charting-ready MCP-schema CSV
    uv run python -m courtvision review <match> --mode review --session r1

<match> is a config id from data/matches/ (t1, t2, t3, t4) or 'all'.
"""

import argparse
from pathlib import Path

from . import config


def _matches(arg):
    return config.match_ids() if arg == "all" else [arg]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="courtvision")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("chart", help="video + config -> per-point chart CSVs")
    p.add_argument("match")
    p.add_argument("clips", nargs="*")
    p.add_argument("--charts-dir", type=Path, default=None)

    p = sub.add_parser("eval", help="charts vs MCP ground truth -> scorecard")
    p.add_argument("match")
    p.add_argument("--charts-dir", type=Path, default=None)

    p = sub.add_parser("calibrate",
                       help="fit the confidence tiers (LOMO) and report")

    p = sub.add_parser("export", help="charting-ready MCP-schema draft CSV")
    p.add_argument("match")

    p = sub.add_parser("draft", help="one-shot: chart + confidence + export")
    p.add_argument("match")

    p = sub.add_parser("track-ball", help="WASB ball tracks -> ball_*.csv")
    p.add_argument("match")
    p.add_argument("clips", nargs="*")

    p = sub.add_parser("players", help="bgsub player boxes -> players_*.csv")
    p.add_argument("match")
    p.add_argument("clips", nargs="*")
    p.add_argument("--pass-a", action="store_true")
    p.add_argument("--pass-b", action="store_true")

    p = sub.add_parser("serve", help="serve detection v3 -> serves.csv")
    p.add_argument("match")

    p = sub.add_parser("align", help="clips -> MCP rows by score-bug join")
    p.add_argument("match")
    p.add_argument("--no-order-pass", action="store_true")

    p = sub.add_parser("fitcourt", help="reel + court_detect -> homography")
    p.add_argument("match")
    p.add_argument("--manual", action="store_true")

    p = sub.add_parser("probe", help="court-view detection -> segments.csv")
    p.add_argument("match")

    p = sub.add_parser("extract", help="segments -> point clips + offsets")
    p.add_argument("match")
    p.add_argument("--skip-clips", action="store_true")

    sub.add_parser("decompose", help="edit-distance decomposition report")

    p = sub.add_parser("review",
                       help="correct drafts against clips (local web UI)")
    p.add_argument("match")
    p.add_argument("--mode", choices=["review", "cold"], required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--seed", default=None)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "chart":
        from . import chart
        for mid in _matches(args.match):
            cfg = config.load(mid)
            chart.chart_match(cfg, stems=args.clips or None,
                              charts_dir=args.charts_dir)
    elif args.cmd == "eval":
        from . import evaluate
        for mid in _matches(args.match):
            cfg = config.load(mid)
            evaluate.evaluate(cfg, charts_dir=args.charts_dir)
    elif args.cmd == "calibrate":
        from . import confidence
        confidence.calibrate_and_report()
    elif args.cmd == "export":
        from . import export
        for mid in _matches(args.match):
            export.export_match(config.load(mid))
    elif args.cmd == "draft":
        from . import chart, export
        for mid in _matches(args.match):
            cfg = config.load(mid)
            chart.chart_match(cfg)
            export.export_match(cfg)
    elif args.cmd == "track-ball":
        from . import ball
        for mid in _matches(args.match):
            ball.track_match(config.load(mid), stems=args.clips or None)
    elif args.cmd == "players":
        from . import players
        for mid in _matches(args.match):
            players.track_match(config.load(mid), stems=args.clips or None,
                                do_a=args.pass_a or not args.pass_b,
                                do_b=args.pass_b or not args.pass_a)
    elif args.cmd == "serve":
        from . import serve
        for mid in _matches(args.match):
            serve.detect_match(config.load(mid))
    elif args.cmd == "align":
        from . import align
        for mid in _matches(args.match):
            align.align_match(config.load(mid),
                              order_pass=not args.no_order_pass)
    elif args.cmd == "fitcourt":
        from . import fitcourt
        for mid in _matches(args.match):
            fitcourt.fit_match(config.load(mid), manual=args.manual)
    elif args.cmd == "probe":
        from . import probe
        for mid in _matches(args.match):
            probe.probe_match(config.load(mid))
    elif args.cmd == "extract":
        from . import extract
        for mid in _matches(args.match):
            extract.extract_match(config.load(mid), skip_clips=args.skip_clips)
    elif args.cmd == "decompose":
        from . import decompose
        decompose.report()
    elif args.cmd == "review":
        from . import review
        review.run(config.load(args.match), args.mode, args.session,
                   seed=args.seed, n=args.n, port=args.port,
                   open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
