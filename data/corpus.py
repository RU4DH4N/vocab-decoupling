import re
from pathlib import Path

import pyarrow.parquet as pq

TEXT_READER_VERSION = "newline-separated-parquet-rows-v2"


def read_text(files: list[str | Path], max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    buf, n = [], 0
    for p in files:
        for batch in pq.ParquetFile(p).iter_batches(columns=["text"]):
            for scalar in batch.column("text"):
                text = scalar.as_py()
                if not text:
                    continue
                separator = "" if not buf else "\n"
                remaining = max_chars - n
                if remaining <= len(separator):
                    return "".join(buf)
                piece = (separator + text)[:remaining]
                buf.append(piece)
                n += len(piece)
                if n >= max_chars:
                    return "".join(buf)
    return "".join(buf)


_UNITS = re.compile(r"\s*\S+|\s+")


def split_words(text: str) -> list[str]:
    return _UNITS.findall(text)


def cap_split(unit: str, max_bytes: int) -> list[str]:
    raw = unit.encode("utf-8")
    if len(raw) <= max_bytes:
        return [unit]
    out, cur, n = [], [], 0
    for ch in unit:
        w = len(ch.encode("utf-8"))
        if cur and n + w > max_bytes:
            out.append("".join(cur))
            cur, n = [], 0
        cur.append(ch)
        n += w
    if cur:
        out.append("".join(cur))
    return out


def to_units(text: str, max_bytes: int | None) -> list[str]:
    words = split_words(text)

    if max_bytes is None:
        out = words
    else:
        out = []
        for w in words:
            out += cap_split(w, max_bytes)

    if "".join(out) != text:
        raise ValueError("segmentation is not lossless")
    return out
