# Match Charting Project ground truth

Point-by-point human charts from Jeff Sackmann's Match Charting Project
(github.com/JeffSackmann/tennis_MatchChartingProject), CC BY-NC-SA 4.0.
Used as held-out TEST labels for the auto-charting pipeline — see LOG.md
for the train/dev/test discipline.

- `points_20170810_nadal_shapovalov.csv` — Canada Masters R16, night,
  both LEFT-handed. Test set 1. Footage: clips/nadal_shapo_r16_2017.mp4
  (TennisTV highlights, 720p60).
- `points_20170812_federer_haase.csv` — Canada Masters SF, day, both
  right-handed. Test set 2 (nearest transfer), footage TBD.
- `points_20230611_djokovic_ruud.csv` — Roland Garros F 2023, CLAY,
  both right-handed. Test set 3 (adversarial: surface + broadcaster).
  Footage: official Roland-Garros extended highlights, 720p50.
- `points_20240713_krejcikova_paolini.csv` — Wimbledon F 2024, GRASS,
  WTA, both right-handed. Test set 4 (adversarial: surface + tour +
  broadcaster). Footage: official Wimbledon extended highlights, 720p25.

Columns: Pt, set/game scores, Pts (score at point start), Svr,
1st/2nd (serve + rally in MCP notation), PtWinner.
