"""Build a synthetic checks database shaped like real retail data.

    python stonkfly/make_demo_data.py --out data/demo_sales.duckdb

WHY THIS EXISTS. A negative result is only worth publishing if someone else
can reproduce it. Shipping the experiment without data makes it unverifiable;
shipping real shop data would expose a client's revenue. So the repository
carries a synthetic database with the same *shape* as the real one and none of
its figures.

WHAT IS PRESERVED. Weekly seasonality, a growth trend, an opening mid-series,
and plausible noise — the features that make a sales chart look like a sales
chart to the renderer. Absolute levels are invented.

WHAT IS NOT. No real revenue, no real shop names, no real dates from any
business. Nothing here can be reversed into anyone's books.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SHOPS = ("Northgate", "Riverside", "Hillcrest")
START = "2023-09-25"          # a Monday
WEEKS = 150


def build(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(START)
    rows = []

    for shop_i, shop in enumerate(SHOPS):
        # Third shop opens two thirds of the way in, mirroring a real network
        # that added a location — the case that makes gross and like-for-like
        # revenue diverge.
        opens = 0 if shop_i < 2 else int(WEEKS * 0.66)
        base = float(rng.uniform(220_000, 340_000))
        trend = float(rng.uniform(-0.0015, 0.0035))   # per week

        for w in range(opens, WEEKS):
            monday = start + pd.Timedelta(weeks=w)
            # Yearly seasonality: December peak, summer trough.
            doy = monday.dayofyear
            season = 1 + 0.28 * np.cos((doy - 355) / 365 * 2 * np.pi)
            level = base * season * (1 + trend) ** w
            week_rev = float(level * rng.normal(1.0, 0.11))

            # Split the week into days, then days into checks.
            day_share = rng.dirichlet(np.full(7, 9.0))
            for d in range(7):
                date = monday + pd.Timedelta(days=d)
                day_rev = week_rev * day_share[d]
                avg_check = float(rng.uniform(900, 1_500))
                n_checks = max(1, int(day_rev / avg_check))
                amounts = rng.lognormal(np.log(avg_check) - 0.18, 0.6, n_checks)
                amounts *= day_rev / amounts.sum()
                for amount in amounts:
                    rows.append((shop, date, float(amount)))

    df = pd.DataFrame(rows, columns=["Store", "InvoiceDate", "Revenue"])
    df["CheckId"] = np.arange(1, len(df) + 1, dtype=np.int64)
    return df[["CheckId", "Store", "InvoiceDate", "Revenue"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_sales.duckdb")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import duckdb

    df = build(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    con = duckdb.connect(str(out))
    try:
        con.register("t", df)
        # Same column names the real pipeline uses, so --db swaps cleanly.
        con.execute("""
            CREATE TABLE clean_transactions AS
            SELECT CheckId::BIGINT AS CheckId, Store::VARCHAR AS Store,
                   CAST(InvoiceDate AS DATE) AS InvoiceDate,
                   Revenue::DOUBLE AS Revenue
            FROM t""")
        n, lo, hi = con.execute(
            "SELECT COUNT(*), MIN(InvoiceDate), MAX(InvoiceDate) "
            "FROM clean_transactions").fetchone()
    finally:
        con.close()

    print(f"{out}: {n:,} checks, {lo} .. {hi}".replace(",", " "))
    print("synthetic — no real business figures")


if __name__ == "__main__":
    main()
