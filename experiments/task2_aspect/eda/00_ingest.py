"""Step 0: stream the raw bz2 CSV into Parquet (data/interim). Raw file is only read.

review_id = 0-based row number in the raw CSV, so ids are stable across reruns.
Also records the raw file's SHA-256 so later steps (and the report) can prove it is unchanged.
"""

from __future__ import annotations

import bz2
import csv
import sys

import edalib as E
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("review_id", pa.int64()),
        ("app_id", pa.int32()),
        ("review_text", pa.string()),
        ("review_score", pa.int8()),
        ("review_votes", pa.int8()),
    ]
)


def main() -> int:
    cfg = E.load_cfg()
    log = E.logger("00_ingest")
    raw = E.raw_file(cfg)
    out = E.interim("reviews.parquet")
    limit = 50_000 if E.SMOKE else None

    with E.timed(log, f"sha256 of {raw.name}"):
        digest = E.sha256(raw)
    stat = raw.stat()
    E.save_json(
        {"file": cfg["input"]["raw_file"], "sha256": digest, "bytes": stat.st_size},
        "raw_file_fingerprint",
    )
    log.info("raw sha256=%s bytes=%d", digest, stat.st_size)

    if E.is_cached(out, log):
        return 0

    n_cols = len(cfg["input"]["columns"])
    batch_rows = int(cfg["input"]["batch_rows"])
    bad_rows = 0
    n = 0
    cols: dict[str, list] = {f.name: [] for f in SCHEMA}

    def flush(writer: pq.ParquetWriter) -> None:
        writer.write_table(pa.table(cols, schema=SCHEMA))
        for v in cols.values():
            v.clear()

    with (
        E.timed(log, "stream CSV -> parquet"),
        bz2.open(raw, "rt", encoding="utf-8", errors="replace", newline="") as fh,
        pq.ParquetWriter(out, SCHEMA, compression="zstd") as writer,
    ):
        for row in csv.reader(fh):
            if limit is not None and n >= limit:
                break
            if len(row) != n_cols:
                bad_rows += 1
                n += 1
                continue
            app_id, text, score, votes = row
            cols["review_id"].append(n)
            cols["app_id"].append(int(app_id))
            cols["review_text"].append(text)
            cols["review_score"].append(int(score))
            cols["review_votes"].append(int(votes))
            n += 1
            if len(cols["review_id"]) >= batch_rows:
                flush(writer)
                log.info("  %s rows", f"{n:,}")
        if cols["review_id"]:
            flush(writer)
    log.info("wrote %s rows (%d malformed skipped) -> %s", f"{n:,}", bad_rows, out)
    E.save_json({"rows_read": n, "malformed_rows": bad_rows}, "ingest_counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
