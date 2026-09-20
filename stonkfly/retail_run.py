"""Run the fly over real retail weeks and over shuffled ones, then compare.

    python -m stonkfly.retail_run --db /path/to/checks.duckdb --weeks 26

WHAT IT REPORTS. For each week the network sees the previous 13 weeks rendered
as a chart and produces QUIET / WORSE / BETTER. The same is done on a shuffled
copy of the series, where the weeks carry identical values but no order.

The comparison is the result. If the network speaks about as often on shuffled
weeks as on real ones, it is responding to the texture of a line, not to the
business behind it — and that is worth stating plainly.

⚠ This is a stunt with a control, not an analyst. Nothing here is validated
for business use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_weeks(db_path: str, metric: str = "revenue", stores=None):
    """Weekly series from a checks database, oldest first, full weeks only."""
    import duckdb
    import pandas as pd

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT InvoiceDate AS date, Store AS store, Revenue AS revenue,
                   CheckId AS check_id
            FROM clean_transactions""").df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sales_display import weekly_series

    return weekly_series(df, metric, stores)


def run_series(brain, decoder, series, window: int, label: str, ms: int):
    """Slide a window over the series, one verdict per step.

    ⚠ Uses brain.rgb_step(), the upstream API: it takes the frame and a
    duration in milliseconds and returns (spike counts, wall seconds). There
    is no observe() on the brain — that name belongs to FlyController, which
    also owns trading concerns we do not want here.

    Learning stays OFF during scoring. With plasticity on, every frame would
    change the weights and the run would measure the order of the weeks as
    much as the weeks themselves — which is exactly what the shuffled control
    is supposed to isolate."""
    from sales_display import sales_frame

    results = []
    for end in range(window, len(series) + 1):
        chunk = series[end - window:end]
        frame = sales_frame(label, chunk, f"week {end}")
        counts, _elapsed = brain.rgb_step(frame, ms, learning=False)
        # decode() expects a rate window in seconds
        r = decoder.decode(counts, ms / 1000)
        r["week_index"] = end
        r["value"] = chunk[-1]
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to checks duckdb")
    ap.add_argument("--metric", default="revenue",
                    choices=("revenue", "checks", "avg_check"))
    ap.add_argument("--weeks", type=int, default=26,
                    help="how many of the most recent weeks to score")
    ap.add_argument("--window", type=int, default=13,
                    help="weeks drawn in each frame (matches MAX_POINTS)")
    ap.add_argument("--ms", type=int, default=500,
                    help="neural time per frame, milliseconds")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="decoder threshold in Hz (upstream default is small)")
    ap.add_argument("--seeds", type=int, default=5,
                    help="shuffled control repetitions")
    ap.add_argument("--out", default="retail_run.json")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from retail_decoder import RetailDecoder, control_shuffled, summarise

    series = load_weeks(args.db, args.metric)
    if len(series) < args.window + 2:
        raise SystemExit(f"нужно минимум {args.window + 2} полных недель, "
                         f"есть {len(series)}")
    # ⚠ +window-1, не +window. Скользящее окно даёт (len - window + 1) шагов,
    # поэтому при срезе на (weeks + window) оценивалось на одну неделю больше
    # запрошенного: --weeks 26 выдавал 27.
    series = series[-(args.weeks + args.window - 1):]
    scored = len(series) - args.window + 1
    print(f"недель в работе: {len(series)} "
          f"(окно {args.window}, оценивается {scored})")

    # Brain and annotations come from the upstream pipeline unchanged.
    # VisualMemoryBrain() loads the connectome itself; annotations() is the
    # helper the upstream controller uses to look up neuron types.
    from stonkfly.neural.common import annotations
    from stonkfly.neural.visual import VisualMemoryBrain

    print("\nзагружаю коннектом…")
    brain = VisualMemoryBrain()
    brain.weights_frozen = True          # scoring must not rewrite the network
    decoder = RetailDecoder(brain.ids, annotations(brain.ids),
                            threshold=args.threshold)
    print(f"нейронов: {brain.n:,}".replace(",", " "))

    print("\nнастоящие недели:")
    real = run_series(brain, decoder, series, args.window,
                      "RETAIL-WEEKLY", args.ms)
    real_sum = summarise(real)
    print("  " + json.dumps(real_sum, ensure_ascii=False))

    print("\nперемешанные недели (контроль):")
    controls = []
    for seed in range(args.seeds):
        shuffled = control_shuffled(series, seed=seed)
        res = run_series(brain, decoder, shuffled, args.window,
                         "SHUFFLED", args.ms)
        s = summarise(res)
        controls.append(s)
        print(f"  seed {seed}: " + json.dumps(s, ensure_ascii=False))

    spoke_real = real_sum["spoke_pct"]
    spoke_ctrl = [c["spoke_pct"] for c in controls]
    mean_ctrl = float(np.mean(spoke_ctrl))
    sd_ctrl = float(np.std(spoke_ctrl))

    print("\n" + "=" * 58)
    print(f"сеть подала сигнал на настоящих неделях: {spoke_real}%")
    print(f"на перемешанных:                        {mean_ctrl:.1f}% "
          f"± {sd_ctrl:.1f}")
    if sd_ctrl > 0:
        z = (spoke_real - mean_ctrl) / sd_ctrl
        print(f"z = {z:+.2f}")
        # Same bar the real analyst uses: two standard deviations.
        print("вывод: " + ("отличается от контроля" if abs(z) >= 2
                           else "НЕ отличается от контроля — реакция не несёт "
                                "информации о бизнесе"))
    else:
        print("вывод: контроль без разброса, сравнивать не с чем")

    Path(args.out).write_text(json.dumps(
        {"real": real_sum, "controls": controls, "detail": real},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nподробности: {args.out}")


if __name__ == "__main__":
    main()
