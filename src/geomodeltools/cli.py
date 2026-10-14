"""Command-line entry points for quick EDA and contact plots.

Lets the director run these from a terminal against a CSV, no notebook needed. Installed
as the `geomodeltools-eda`, `geomodeltools-contact` and `geomodeltools-report` console
scripts (see pyproject.toml's `[project.scripts]`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .contact_plots import contact_plot  # noqa: E402
from .eda import cat_vs_cat, categorical_eda, quick_eda  # noqa: E402
from .report import write_html  # noqa: E402


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix!r} (expected .csv, .txt or .parquet)")


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def eda_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="geomodeltools-eda",
        description="Quick EDA plot for a numeric column (optionally split by a category), "
        "or for one or two categorical columns.",
    )
    parser.add_argument("csv", type=Path, help="Input CSV/parquet file.")
    parser.add_argument("--value", help="Numeric column for a quick-EDA figure.")
    parser.add_argument("--category", help="Categorical column: splits --value by it, or plots it alone.")
    parser.add_argument("--category-y", help="With --category and no --value: plots --category vs --category-y.")
    parser.add_argument("--weight", help="Optional column of per-row weights.")
    parser.add_argument("--log", action="store_true", help="Log-scale the numeric EDA plots.")
    parser.add_argument("-o", "--out", type=Path, default=Path("eda.png"), help="Output image path.")
    args = parser.parse_args(argv)

    df = _read_table(args.csv)

    if args.value:
        fig = quick_eda(df, args.value, category=args.category, weight=args.weight, log=args.log)
    elif args.category and args.category_y:
        fig = cat_vs_cat(df, args.category, args.category_y, weight=args.weight)
    elif args.category:
        fig = categorical_eda(df, args.category, weight=args.weight)
    else:
        parser.error("Pass --value for a numeric column, or --category (optionally with --category-y).")
        return

    _save(fig, args.out)


def contact_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="geomodeltools-contact",
        description="Contact plot: a numeric or categorical variable vs. distance from a unit boundary.",
    )
    parser.add_argument("csv", type=Path, help="Input CSV/parquet file.")
    parser.add_argument("--coords", nargs="+", required=True, help="Coordinate column names, e.g. X Y Z.")
    parser.add_argument("--unit", required=True, help="Column with the geological unit codes.")
    parser.add_argument("--inside", nargs="+", required=True, help="Unit code(s) on the inside of the contact.")
    parser.add_argument("--outside", nargs="+", required=True, help="Unit code(s) on the outside of the contact.")
    parser.add_argument("--contact", nargs="+", default=None, help="Optional unit code(s) exactly on the contact.")
    parser.add_argument("--value", help="Numeric column to profile across the contact.")
    parser.add_argument("--category", help="Categorical column whose composition to track across the contact.")
    parser.add_argument("--bin-width", type=float, default=None, help="Distance bin width (auto-picked if omitted).")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of bins, split evenly either side of 0.")
    parser.add_argument("--weight", help="Optional column of per-row weights.")
    parser.add_argument("-o", "--out", type=Path, default=Path("contact.png"), help="Output image path.")
    args = parser.parse_args(argv)

    df = _read_table(args.csv)
    fig = contact_plot(
        df,
        args.coords,
        args.unit,
        args.inside,
        args.outside,
        value=args.value,
        category=args.category,
        contact=args.contact,
        bin_width=args.bin_width,
        n_bins=args.n_bins,
        weight=args.weight,
    )
    _save(fig, args.out)


def report_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="geomodeltools-report",
        description="Build a self-contained, interactive HTML EDA/contact-plot report: open it in any "
        "browser, no server or Node needed.",
    )
    parser.add_argument("csv", type=Path, help="Input CSV/parquet file.")
    parser.add_argument("--numeric", nargs="+", default=[], help="Numeric columns to include.")
    parser.add_argument("--categorical", nargs="+", default=[], help="Categorical columns to include.")
    parser.add_argument("--weight", help="Optional column of per-row weights.")
    parser.add_argument(
        "--contacts-json", type=Path, default=None,
        help="Path to a JSON file: a list of contact-plot scenarios, each a dict with coords/unit/inside/"
        "outside and value and/or category (see build_report's docstring for the full set of keys).",
    )
    parser.add_argument("--title", default="geomodeltools — EDA report", help="Page title.")
    parser.add_argument("-o", "--out", type=Path, default=Path("report.html"), help="Output HTML path.")
    args = parser.parse_args(argv)

    df = _read_table(args.csv)
    contacts = json.loads(args.contacts_json.read_text(encoding="utf-8")) if args.contacts_json else None

    out = write_html(
        df, args.out, title=args.title,
        numeric=args.numeric, categorical=args.categorical, contacts=contacts, weight=args.weight,
    )
    print(f"Wrote {out}")
