# EDA and contact plots: the contract between the Python and JavaScript implementations

`geomodeltools.report.build_report` (Python) is the reference. It turns a table into a **payload**: plain numbers and labels that a
renderer only has to draw (`static/report.js`, which never recomputes a statistic). The browser tool `geoplots_local` has its own
JavaScript engine that builds the same payload from a CSV, so colleagues without Python can use it. Its build is only allowed to
go ahead if that engine reproduces the golden payloads below. If it does not, the JavaScript is fixed.

**Contract version: 1.** A change to any rule below bumps the number, regenerates `tests/golden/plots/` and is reviewed as one commit.

## The table
CSV, header row, one delimiter (`,` `;` or tab). Missing values: empty cell, `NA`, `NaN`, `nan`, `NULL`, `null`, `None`, `n/a`, `N/A`.
A column is **numeric** when every non-missing cell is a number, and **text** otherwise (labels are then the cell text, exactly).
A numeric column used as a category is labelled as Python does: integers as `1`, `2`, `10` when no cell is missing; with any missing
cell the column is float and the labels are `1.0`, `2.0`, `10.0` (a float with a fraction keeps its shortest form, `2.5`).
Categories are sorted numerically when numeric, otherwise by character code.

## Rules (names as in `eda.py`, `contact_plots.py`, `report.py`)
* **`weight`**: a numeric column; without it every row weighs 1. Rows with a missing weight are not part of the contract.
* **`quick_eda_data` overall**: rows with a value; `values` and `weights` in file order; `probability.v` the sorted values and `z`
  the normal quantile of the Hazen position `(cumulative weight - weight/2) / total weight` (inverse normal CDF to 1e-12);
  `summary`: `count` rows, weighted `mean`, unweighted `median`, weighted population `std`, `cov = std / mean` (NaN when mean is 0).
* **by category**: rows with value and category; categories ordered by median (ties are not part of the contract); per category the
  same `values`, `probability` and `summary`.
* **`categorical_eda_data`**: rows with the category; weighted counts per label sorted by count, descending; `pct` of the total.
* **`cat_vs_cat_data`**: rows with both categories; weighted sums per (x, y); each x row as a percentage of its row total.
* **`distance_to_contact`**: signed distance in the coordinate columns. A row whose unit is in `inside` gets `+` the distance to the
  nearest `outside` or `contact` row; a row in `outside` gets `-` the distance to the nearest `inside` or `contact` row; other rows
  get no distance. Euclidean distance, exact (not approximate).
* **bins**: `n_bins` bins of `bin_width`, centred on 0, each `(left, right]` (a value equal to the lowest edge is outside). Without
  `bin_width`: `nice_step(2 * P90(|d|) / n_bins)`, with `P90` the linear-interpolation 90th percentile of the absolute distances of
  the rows used (those with a distance and a value or category), `nice_step` rounding up to 1, 2 or 5 times a power of ten.
  Rows outside the outer bin edges are left out. `mid` is the bin midpoint, `n` the row count (profile) or weight (composition).
* **`contact_profile`**: weighted mean of the value per bin; `null` in an empty bin.
* **`contact_composition`**: weighted sum per bin and category; percentages of the bin total, `null` in an empty bin; categories
  sorted as above.

NaN and infinity are written as `null`. Floats are compared with relative tolerance 1e-12 plus absolute 1e-13; labels, counts and
list lengths must be identical.

## Not in the contract (version 1)
Missing weights, ties in the median or count order, tables with more than one type in a column, files that are not CSV (the browser tool
also reads `.gz`), and the drawing itself.

## Golden data
`tests/golden/plots/`: `data.csv` (600 synthetic rows, with missing cells of both kinds), `cases.json` (four cases: weighted and
unweighted EDA with text and integer categories, a contact with a contact unit and weights, a contact with several codes, a fixed bin
width and rows beyond the outer bins) and `expected/<case>.json` (the payloads). `MANIFEST.json` holds the SHA-256 of every file
and the geomodeltools commit that wrote them; `generate.py` regenerates all of it and should be run only on a contract change.
Synthetic data only.
