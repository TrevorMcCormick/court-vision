# Court Vision as a data product — spec

*The product-management view of the repo: who consumes what, under
which contract, measured how, governed by which gates. Companion to
[scorecard.md](scorecard.md) (current status) and
[model-card.md](model-card.md) (system + limitations). Last updated
2026-07-20.*

## Product statement

**For** tennis charters (the author today; Match Charting Project
volunteers tomorrow) **who** spend on the order of two hours charting
a match's points by hand, **Court Vision delivers** machine-drafted
MCP-style notation
with a calibrated trust flag per point and a keyboard-first review
tool, **so that** correcting beats transcribing. **Unlike** raw CV
demos, every number ships with an out-of-sample receipt — and the core
value hypothesis (correcting beats transcribing) is explicitly
unproven until the cv-18 stopwatch experiment measures it.

## Consumers and their jobs

| consumer | job to be done | interface |
|---|---|---|
| A charter correcting a match | "Turn 2 hours of charting into 30 minutes of review" | `courtvision review` UI + draft export CSV |
| The MCP community | "More charted matches without lowering chart quality" | corrected charts in MCP points format; cv-17 draft exports for evaluation |
| The project itself | "Know if the pipeline got better or worse" | `courtvision eval` scorecards vs frozen baselines |
| Future training consumers | "Ground truth with per-point video timestamps (which MCP lacks)" | the charting app's training bundle (shipping since cv-19: MCP points CSV + segments CSV + manifest per charted match; import-as-benchmark-match queued) |

## Data contracts

**Draft export** (`outputs/<t>/export/<t>_mcp_draft.csv`): MCP points
schema (match_id, Pt, Set/Gm/Pts, Svr, 1st/2nd) + machine string,
`confidence` (high/low), `conf_p`, `clip`, `serve_s` (jump-to
timestamp), `n_shots`. Grammar honesty: strings are MCP-*style*, not
MCP-legal (`s` prefix, `?` refusal tokens) — whether to emit
strict-legal-with-blanks is question 3 to the charter community
(cv-17), unresolved.

**The trust contract** (the product's core promise):
- HIGH ⇒ ≥90% of flagged points within 5 token edits, held-out —
  currently delivering 92% at 48% coverage (per-fold floor 77%;
  disclosed in the scorecard).
- LOW carries no promise beyond "the draft may still help".
- No sign-off tier is offered: it has failed leave-one-match-out at
  every attempted n, and shipping it anyway would be a false promise.

**Experiment integrity:** review sessions grade against
sha256-verified frozen exports — pipeline improvements can never
silently regrade a human's completed session.

**License:** exports join MCP columns ⇒ CC BY-NC-SA 4.0 with
attribution to Tennis Abstract and the volunteer charters.
Noncommercial, share-alike, always.

## Quality model (SLO-style, mapped to standard dimensions)

| dimension | commitment | current | measured by |
|---|---|---|---|
| Accuracy (triage) | HIGH precision ≥90% held-out | 92% | LOMO table, benchmark.md |
| Accuracy (content) | tracked, no floor promised | 67% of points ≤5 edits; 5.7% ≤1 | acceptance metric, every eval |
| Completeness | every broadcast-visible point extracted & aligned | 508 drafted / 491 scored on 7 matches; faults structurally excluded | alignment yield per staging |
| Validity | every string tokenizes under the draft grammar | 100% (grammar-enforced) | mcp/notation tokenizers + lint |
| Consistency | truth corrections are republished, never patched silently | 2 corrections on record (transcription 2026-07-10 in LOG.md; parity 2026-07-20 in benchmark.md) | LOG.md + benchmark.md |
| Auditability | every number traces to a script + LOG entry | receipts policy | LOG.md freezes; experiments/ |
| Cost | ~$0 marginal per match | $0 (WASB local); SAM 3 optional at ~$12/match | LOG session footers |

Error-budget thinking, adapted: the HIGH tier's 8% imprecision is the
budget; the two feeds spending it fastest (t5, t3) are named in the
gaps register rather than averaged away. **Promotion rule:** a change
that blows the budget on the frozen benchmark does not ship —
coverage is sacrificed before precision (both shipped examples: the
t4 gates traded 1.6 pts of coverage to halve disasters; the strict
sign-off tier is withheld entirely because it can't hold its number).

**Benchmark versioning:** the frozen 7-match set with corrected truth
is **benchmark-v2** (v1→v2 boundary: the 2026-07-20 changeover-parity
truth correction; pre-correction server-end numbers are
non-comparable and marked as such in benchmark.md). Constants are
never tuned against the frozen set silently — every re-tune is a
numbered freeze with before/after tables. Adding matches or further
truth corrections increments the version.

**Self-audit against the standard data-product characteristics**
(DATSIS, scaled to a git repo):

| characteristic | mechanism here |
|---|---|
| Discoverable | README indexes the doc set; devlog narrates it publicly |
| Addressable | stable repo paths; exports keyed by match id + sha256 freeze |
| Trustworthy | published held-out numbers + receipts policy + this quality table |
| Self-describing | docs travel in-repo; USAGE + contract sections define every field |
| Interoperable | MCP notation is the interchange format (grammar caveat open, cv-17 q3) |
| Secure/compliant | CC BY-NC-SA honored; video never redistributed |

Known contract gap (designed, unbuilt): exports do not yet carry a
provenance stamp (pipeline git SHA, schema version, benchmark
version, run date) and the scorecard tables are hand-assembled rather
than emitted by the eval run. Both are queued as cheap hardening —
generated numbers can't drift; hand-typed ones eventually lie.

## Lifecycle and gates

1. **Now → cv-18:** pipeline frozen for chart content; only truth
   corrections, diagnosis, and triage-layer fixes ship. The charting
   app shipped 2026-07-20 (`courtvision charter`) and is already
   charting its first from-scratch match toward an MCP submission;
   the cv-18 stopwatch restart on it is parked for the next cycle.
2. **cv-18 result:** the correction histogram + minutes-per-point
   ratio decide everything: if correcting beats charting, build in
   edit-cost order (structure → letters → endings); if not, the
   review tool + confidence tier are the product and the roadmap
   reorders.
3. **Scale loop (post-cv-18):** each corrected match feeds the
   training bundle → more calibration n → tighter tiers → eventually
   a shippable sign-off tier — the compounding flywheel the app's
   ground-truth factory exists to start.

## Operating practices already in force

- **Out-of-sample by default** — constants tuned on a named match,
  frozen, then scored elsewhere; every re-tune is a numbered freeze.
- **Receipts, not claims** — mechanisms are named from pixels;
  refuted hypotheses stay on the record as dead ends.
- **Honest denominators** — e.g., letters are only scored where rally
  lengths agree; footnotes say why.
- **Truth is auditable too** — the answer key gets the same
  adversarial treatment as the pipeline (two corrections shipped).

## Top risks

| risk | exposure | mitigation |
|---|---|---|
| Value hypothesis fails (correcting ≯ charting) | the whole product thesis | cv-18 measures it before further build spend |
| Small n (491 pts) | tier thresholds fragile | staging playbook proven config-only; grow matches |
| License scope (BY-NC-SA) | no commercial path for exports | accepted; product is community contribution |
| **No repo LICENSE/NOTICE file** | code is all-rights-reserved by default; data license implied but not declared | **open decision for the owner**: license code separately (e.g. MIT/Apache) from data (CC BY-NC-SA), add NOTICE with the MCP attribution string |
| Single maintainer | bus factor 1 | receipts + this doc set make the state legible |
| Footage variance (new feeds break assumptions) | staging cost per feed | config-not-code precedent held on t5/t6/t7 |
