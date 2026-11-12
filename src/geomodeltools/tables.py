"""Reading a table file: a text table of any extension (delimiter, decimal mark and encoding worked out from the file unless given),
Parquet or Excel. Moved here from geoview's import wizard (2026-11-05) so notebooks and apps read tables exactly as geoview does.

    df, parse = read_table("collar.csv")                     # parse: the ParseOptions actually used (delimiter ";", decimal ","...)
    df, _ = read_table("collar.csv", parse)                  # read again the same way
    df, _ = read_table(raw_bytes, name="collar.csv", nrows=300)   # the start of a big file, from bytes

numpy and pandas only (Parquet needs pyarrow, Excel needs openpyxl, as pandas does).
"""
from __future__ import annotations

import io
import pathlib
import re
from dataclasses import asdict, dataclass, fields

import pandas as pd

#: tables that are not text (read by their own readers)
BINARY_TABLES = (".parquet", ".xlsx", ".xls")
#: extensions that are something else (or not readable): never taken for a text table
NOT_TABLES = {".obj", ".omf", ".dxf", ".mtl", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".pdf", ".zip", ".7z", ".gz", ".rar",
              ".exe", ".dll", ".docx", ".pptx", ".dwg", ".00t", ".dm"}
DELIMITERS = {"auto": "Auto", ",": "Comma ,", ";": "Semicolon ;", "\t": "Tab", "|": "Pipe |", "ws": "Spaces or tabs"}
DECIMALS = {"auto": "Auto", ".": "Point .", ",": "Comma ,"}
ENCODINGS = {"auto": "Auto", "utf-8": "UTF-8", "cp1252": "Windows (cp1252, Latin-1)"}
_HEAD = 8192
_PREVIEW_BYTES = 2_000_000


class TableReadError(ValueError):
    """A file that cannot be read as a table (not text, or not readable with the options given)."""


@dataclass
class ParseOptions:
    """How a table file is read. "auto" is worked out from the file. ``header``: the row with the column names (-1: none, the columns
    are then col1, col2...); ``skip``: rows to ignore first; ``na``: extra words that mean empty, comma separated ("ND, -99")."""
    delimiter: str = "auto"
    decimal: str = "auto"
    encoding: str = "auto"
    header: int = 0
    skip: int = 0
    na: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> ParseOptions:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def _options(parse) -> ParseOptions:
    return parse if isinstance(parse, ParseOptions) else ParseOptions.from_dict(parse)


def is_text_table(name: str) -> bool:
    """Is the table a text file (delimiter, decimal mark and encoding apply), as opposed to Parquet or Excel?"""
    return pathlib.Path(name).suffix.lower() not in BINARY_TABLES


def is_importable_name(name: str) -> bool:
    """Could a file of this name be a table? Only its extension is looked at."""
    return pathlib.Path(name).suffix.lower() not in NOT_TABLES


def looks_like_text(head: bytes) -> bool:
    """Is this the start of a text file (no NUL bytes, almost no control characters)? A cut through a multi-byte character is allowed."""
    if not head or b"\x00" in head:
        return False
    controls = sum(1 for b in head if b < 32 and b not in (9, 10, 13, 12))
    return controls / len(head) < 0.01


def head_of(source, n: int = _HEAD) -> bytes:
    """The first ``n`` bytes of a file (a path, or bytes); empty if it cannot be read."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source[:n])
    try:
        with open(source, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def is_table_file(source, name: str | None = None) -> bool:
    """Is this file (a path, or bytes with its ``name``) a table: Parquet or Excel, or any text file that is not another known format?"""
    name = name or pathlib.Path(source).name
    ext = pathlib.Path(name).suffix.lower()
    if ext in BINARY_TABLES:
        return True
    return ext not in NOT_TABLES and looks_like_text(head_of(source))


def esri_grid(head: bytes) -> dict | None:
    """The header of an ESRI ASCII grid (``ncols``, ``nrows``, ``xllcorner`` or ``xllcenter``, ``yllcorner`` or ``yllcenter``, ``cellsize``,
    optional ``NODATA_value`` lines before the numbers), or None. A grid like this is a surface (a DEM), not a table."""
    keys, n = {}, 0
    for line in head.decode("ascii", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].lower() in ("ncols", "nrows", "xllcorner", "xllcenter", "yllcorner", "yllcenter", "cellsize", "nodata_value"):
            try:
                keys[parts[0].lower()] = float(parts[1])
            except ValueError:
                return None
            n += 1
        else:
            break
    if {"ncols", "nrows", "cellsize"} <= keys.keys() and ("xllcorner" in keys or "xllcenter" in keys) and ("yllcorner" in keys or "yllcenter" in keys):
        return {**keys, "header_lines": n}
    return None


def _bytes(source, head: int | None = None) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        with open(source, "rb") as fh:
            data = fh.read(head + 1) if head else fh.read()
    if head and len(data) > head:  # only the start: drop the row the cut went through
        data = data[:head]
        data = data[: data.rfind(b"\n") + 1] or data
    return data


def _encoding(data: bytes, encoding: str) -> str:
    if encoding != "auto":
        return encoding
    try:
        data[:200_000].decode("utf-8")
        return "utf-8-sig" if data[:3] == b"\xef\xbb\xbf" else "utf-8"
    except UnicodeDecodeError as e:
        if e.start > 199_990:  # the cut went through a character, not a different encoding
            return "utf-8"
        return "cp1252"


def _sniff_delimiter(name: str, text_head: str) -> str:
    if pathlib.Path(name).suffix.lower() == ".tsv":
        return "\t"
    lines = [ln for ln in text_head.splitlines() if ln.strip()][:5]
    line = lines[0] if lines else ""
    if line.count(";") and len({ln.count(";") for ln in lines}) == 1 and all(len(re.findall(r"\d,\d", ln)) == ln.count(",") for ln in lines):
        return ";"  # semicolons separate the fields and the commas are decimal marks (1,5)
    counts = {d: line.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=lambda d: (counts[d], d == ","))
    if counts[best]:
        return best
    return "ws" if len(line.split()) >= 2 else ","  # columns separated by runs of spaces (a fixed-width or .asc / .dat export), or one column


_COMMA_NUMBER = re.compile(r"-?\d+,\d+")


def _is_number(value) -> bool:
    try:
        float(str(value).replace(",", "."))
        return True
    except ValueError:
        return False


def _comma_decimals(df: pd.DataFrame) -> bool:
    """Do the text columns hold numbers written with a decimal comma (``1,5``)?"""
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        v = s.dropna().astype(str).head(200)
        if len(v) and v.str.fullmatch(_COMMA_NUMBER.pattern).mean() >= 0.9:
            return True
    return False


def read_table(source, parse: ParseOptions | dict | None = None, *, name: str | None = None, nrows: int | None = None) -> tuple[pd.DataFrame, ParseOptions]:
    """The table of a file (a path, or bytes with its ``name``) read with ``parse`` ("auto" options are worked out from the file), and the
    `ParseOptions` actually used. ``nrows`` reads only the start of a big text file (a preview). Raises `TableReadError` for a file that
    is not a table."""
    p = _options(parse)
    if not isinstance(source, (bytes, bytearray)):
        source = pathlib.Path(source)
        name = name or source.name
    if name is None:
        raise TypeError("read_table: give the file's name= when reading bytes (its extension says how to read it)")
    ext = pathlib.Path(name).suffix.lower()
    header = None if int(p.header) < 0 else int(p.header)
    if ext == ".parquet":
        df = pd.read_parquet(io.BytesIO(_bytes(source)))
        return (df.head(nrows) if nrows else df), p
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(_bytes(source)), header=header, skiprows=int(p.skip) or None, nrows=nrows)
        return df, p
    data = _bytes(source, _PREVIEW_BYTES if nrows else None)
    if not looks_like_text(data[:_HEAD]):
        raise TableReadError(f"{name} is not a text file, so it cannot be read as a table (text tables of any extension, Parquet and Excel are).")
    enc = _encoding(data, p.encoding)
    sep = p.delimiter if p.delimiter != "auto" else _sniff_delimiter(name, data[:4096].decode(enc, errors="replace"))
    na = [w.strip() for w in str(p.na).split(",") if w.strip()] or None
    regex = r"\s+" if sep == "ws" else sep

    def read(decimal, header=header):
        try:
            return pd.read_csv(io.BytesIO(data), sep=regex, decimal=decimal, encoding=enc, header=header, skiprows=int(p.skip) or None, na_values=na,
                               nrows=nrows, engine="python" if sep == "ws" else None, skipinitialspace=sep == "ws")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
            raise TableReadError(f"{name}: cannot be read with these settings ({e})") from None

    dec = p.decimal
    df = read("." if dec == "auto" else dec)
    if dec == "auto" and sep != "," and _comma_decimals(df):  # semicolon files from Spanish-locale software: 1,5 means 1.5
        dec, df = ",", read(",")
    elif dec == "auto":
        dec = "."
    if header == 0 and df.shape[1] >= 2 and all(_is_number(c) for c in df.columns):  # the first line is numbers: the file has no column names
        df, header = read(dec, header=None), -1
    if header is None or header == -1:
        header = -1
        df.columns = [f"col{k + 1}" for k in range(df.shape[1])]
    return df, ParseOptions(delimiter=sep, decimal=dec, encoding=enc, header=header, skip=int(p.skip), na=p.na)
