# Desurvey: the contract between the Python and JavaScript implementations

`geomodeltools.desurvey.desurvey_intervals` (Python) is the reference; it was moved here from geoview, where its minimum-curvature
core was checked against Leapfrog Geo 2026.1.2 (notebook `desurvey_vs_leapfrog_anonymised` in geoview: under a quarter of a
millimetre in every hole of the anonymised Centinela set). The browser tool `geoplots_local` has a JavaScript copy so colleagues
without Python can desurvey drillholes before drawing contact plots. Its build is only allowed to go ahead if the copy reproduces the
golden data below; if it does not, the JavaScript is fixed.

**Contract version: 1.** A change to a rule below bumps the number, regenerates `tests/golden/desurvey/` and is reviewed as one commit.

## Conventions
* Azimuth: degrees clockwise from north. Coordinates: X east, Y north, Z up.
* **Dip**: `dip_positive_down` (default **True** in this package): the table stores dip positive when the hole points down, as
  Leapfrog Geo writes it in its own survey export. False for tables that store negative-down dips (an acQuire export, for example,
  and geoview's own default). The same tables read with the right flag give the same positions; read with the wrong flag, holes
  point up. The low-level functions (`directions`, `desurvey`) take the elevation angle, negative downward.

## Rules
* **Method**: minimum curvature. Between two survey stations the direction is a spherical interpolation of the station directions,
  and the position steps along the arc with ratio factor `2 / dogleg * tan(dogleg / 2)` (1 for a straight step). Before the first
  station the first direction applies, after the last the last one. Stations are taken in depth order (a stable sort).
* **Positions** of each interval: start (`FROM`), middle (`(FROM + TO) / 2`, on the arc, not the chord) and end (`TO`).
* **A hole that cannot be placed is left out and reported, never guessed**: two survey rows at the same depth (unless a de-duplication
  rule is chosen), or no survey rows and no azimuth/dip on the collar. A collar with its own azimuth and dip and no survey uses them.
* **De-duplication** of survey rows sharing a depth: `skip` (the hole is left out), `first` or `last` (that row of each depth, in table
  order). The Python function also accepts a function; the browser tool offers the three named rules.
* **Intervals** of a hole with no collar are dropped; so are those of left-out holes. `_row` is the 0-based row of the interval in
  the table given. Without intervals, each hole is cut every `trace_step` metres (the last cut at the hole length).
* **Report** (`check_tables`): `duplicate_survey_depth`, `survey_without_collar`, `no_survey`, `interval_without_collar`,
  `no_intervals`, `interval_beyond_depth`, each with `type`, `kind`, `hole`, `table`, `row` (counted from 1), `conflicting_row`,
  `depth`, `message` (exact text), in the order the Python lists them.
* Columns are named by the caller; finding them by their usual names is a convenience of the Python function, not of the contract.

## Tolerance
Positions: 1e-8 m absolute plus 1e-12 relative. Everything else (holes, depths, row numbers, report rows and messages, skipped holes
and reasons) must be identical.

## Not in the contract (version 1)
A distance that falls exactly on a bin edge (neighbouring intervals of one hole are often exactly 2, 3 or 4 m apart): two correct implementations can put it in different bins because they differ by 1e-12, so the golden pipeline case uses a bin width (1.37 m) that keeps the edges away from such values. De-duplication by a function, the dogleg-severity and trace-vertex helpers (`dogleg_severity`, `trace_depths`), and the choice of
column names by pattern.

## Golden data
`tests/golden/desurvey/`: synthetic `collars.csv`, `surveys.csv`, `intervals.csv` (and `collars_neg.csv`, `surveys_neg.csv` with the
dips negated), five cases (positive-down with skip, first and last de-duplication; the negative-down tables with the flag off, which
must give the same positions; traces every 25 m), and `expected/<case>.json` written by the Python reference. The tables contain
curved holes, an up-hole, a survey out of depth order, a repeated survey depth, a hole with its own collar direction, one with no
direction at all, one with no intervals, rows of a hole with no collar, and an interval ending below its hole. `MANIFEST.json` holds the
SHA-256 of every file and the commit that wrote them; `generate.py` regenerates all of it and should be run only on a contract change.
Synthetic data only. The Leapfrog comparison on real (anonymised) data is a manual check before a release, not part of the tests.
