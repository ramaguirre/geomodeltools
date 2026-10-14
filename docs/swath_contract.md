# Swath analysis: the contract between the Python and JavaScript implementations

`geomodeltools.swath` (Python) is the reference. The browser tool `swath_local` runs the same calculation in JavaScript, and its build
is only allowed to go ahead if its engine gives the same answers on the golden data below. If it does not, the JavaScript is fixed.

**Contract version: 1.** A change to any rule below bumps the number in `swath.CONTRACT_VERSION`, regenerates the golden data and is
reviewed as one commit.

## What is calculated
Blocks (a block model) are compared with composites (drill samples), panel by panel and per estimation unit (UE).

* **Panel grid** from a Vulcan header: parent block size = largest minimum block size over the schemas; offsets = component-wise
  extremes over the schemas; panels hold `multiplier` parent blocks per axis; rotation = Rz(90 - bearing) Rx(dip) Rz(plunge).
  Panel `(i, j, k) = floor((xyz - corner) . axes / size)`; panel number is X fastest. Edge panels are kept.
* **Row rules**, in this order, each excluded row counted once under the first rule it fails: no value, value outside
  `[value_min, value_max]`, no coordinates, no UE, UE excluded by `ue_filter`, invalid weight (not finite or <= 0), outside the grid.
* **Weights**: blocks = dx * dy * dz (times density if given; 1 without dimensions); composites = length (1 without it).
* **UE codes** are compared as clean text: 22, "22", "22.0" and " 22 " are the same code; empty and NA-like values are no code.
* **Grades** are weight-averaged (`ley_*`); `media_simple_*` are unweighted means. A panel is compared only with at least
  `min_blocks` blocks and `min_samples` composites. Swaths add the compared panels along each axis.
* **Summary per UE**: global grades and difference, mean of panel grades, correlation and slope of block against composite grade
  over panels (population statistics), and a status when a UE cannot be compared.

Output names are those of the JavaScript engine (Spanish): `panel, i, j, k, x_centro ... diferencia_pct` for panels,
`indice, coordenada, ... media_simple_muestras` for swaths, `qa` counts per table.

## Not in the contract (version 1)
Building the grid from the block extents when there is no header, the grid check (blocks crossing panel edges), file parsing
(CSV, zip, gzip) and the figures. The JavaScript tests cover those on their own.

## Golden data
`tests/golden/swath/`: a small synthetic model and composites (unrotated and rotated 65 degrees), four cases (defaults, other
multiplier, composite and block minimums, value maximum, UE filter), and the expected results written by the JavaScript engine of
`swath_local` (validated against the original desktop tool on real models). `MANIFEST.json` records the `swath_local` commit and
the SHA-256 of every file; `generate.py` regenerates everything and should be run only on a contract change.

**Tolerance**: relative 1e-12 plus absolute 1e-13 (the percentage differences amplify a subtraction of two nearly equal grades).
Counts, panel indices and QA numbers must be identical.

Synthetic data only. Real-model comparisons (4.3 M and 16.6 M blocks) are run by hand before a release and are not in the repository.
