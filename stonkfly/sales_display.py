"""Render weekly retail metrics into an RGB frame for the fly visual system.

Drop-in replacement for stonkfly.display.market_frame. Same contract: returns
a (180, 320, 3) uint8 array that retinal_samples() reads through the inferred
optic projection. Never reads future weeks.

WHY THIS SHAPE. The fly's optic lobe was measured, not designed for us: the
upstream projection maps 3,335 brightness inputs and 811 R8 colour inputs onto
a fixed 320x180 canvas. Changing the canvas would invalidate that mapping, so
the frame size, the colour encoding and the general layout stay exactly as in
the upstream market renderer. Only the *content* changes: weekly revenue of a
retail network instead of a crypto pair.

WHAT THE COLOURS MEAN. Upstream paints a segment blue when the series rises
and red when it falls. We keep that convention because the downstream R8
colour channels are wired to it. For retail this reads naturally: blue weeks
grew against the previous week, red weeks shrank.

HONEST LIMITS (mirroring upstream AGENTS.md discipline):
  * This is a wiring-constrained spiking experiment, not an analyst. The
    connectome supplies anatomy. It supplies no retail knowledge whatsoever.
  * A fly's optic lobe evolved to detect looming predators and optic flow.
    Nothing in it evolved to read a sales chart. Any agreement between its
    output and a real revenue dip is coincidence until measured against a
    control.
  * Nothing here claims the network analyses, understands or predicts sales.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Upstream canvas and palette. Do not change: the optic projection is fit to it.
WIDTH, HEIGHT = 320, 180
BG = (235, 240, 249)
HEADER_BG = (19, 36, 71)
HEADER_FG = (219, 229, 249)
GRID = (200, 212, 233)
RISE = (0, 101, 183)      # blue: this point is above the previous one
FALL = (197, 37, 78)      # red: below
MARKER = (27, 39, 81)
FOOT_FG = (28, 46, 82)

# ⚠ 13, NOT 100. Upstream keeps 100 ticks because crypto ticks arrive every
# few seconds and the chart is a texture, not a readable series. Weekly retail
# data is different: 143 weeks squeezed into the 294px plotting area leaves
# 2.1px per week, so individual weeks stop being resolvable and the rise/fall
# colouring turns into noise. Measured on real data:
#     143 weeks -> 2.1 px/week   unreadable
#      52 weeks -> 5.8 px/week   unreadable
#      26 weeks -> 11.8 px/week  readable
#      13 weeks -> 24.5 px/week  clearly readable
# 13 also matches the baseline window the real detector calibrates against,
# so both look at the same stretch of history.
MAX_POINTS = 13


def sales_frame(label: str, history, footer: str = "") -> np.ndarray:
    """Weekly series -> RGB frame.

    label    short title drawn in the header, e.g. "RETAIL-WEEKLY".
    history  iterable of weekly values, oldest first. Only the last
             MAX_POINTS are drawn, matching upstream behaviour.
    footer   optional one-line caption, e.g. the week being analysed.
    """
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)

    d.rectangle((0, 0, WIDTH - 1, 27), fill=HEADER_BG)
    d.text((9, 8), label, fill=HEADER_FG)

    for x in range(12, 310, 30):
        d.line((x, 34, x, 160), fill=GRID)
    for y in range(38, 162, 24):
        d.line((10, y, 308, y), fill=GRID)

    values = np.asarray(list(history)[-MAX_POINTS:], dtype=float)
    values = values[np.isfinite(values)]

    if len(values):
        # Upstream span logic, kept verbatim: a flat series must still occupy
        # a sane fraction of the canvas instead of collapsing to one line,
        # otherwise the brightness inputs see almost nothing.
        span = max(float(np.ptp(values)), float(np.mean(values)) * 0.002)
        lo = float(values.min()) - span * 0.12
        span *= 1.24
        points = [
            (12 + i * 294 / max(1, len(values) - 1), 153 - (v - lo) / span * 109)
            for i, v in enumerate(values)
        ]
        if len(points) > 1:
            for a, b in zip(points, points[1:]):
                # y grows downward, so b[1] <= a[1] means the value went UP
                d.line((*a, *b), fill=RISE if b[1] <= a[1] else FALL, width=3)
        for x, y in points:
            d.rectangle((x - 1, y - 1, x + 1, y + 1), fill=MARKER)

    if footer:
        d.text((9, 165), footer[:50], fill=FOOT_FG)
    return np.asarray(im, dtype=np.uint8)


def weekly_series(df, metric: str = "revenue", stores=None) -> list[float]:
    """Aggregate a transaction frame into a weekly series, oldest first.

    Expects columns: date (datetime64), store, revenue, check_id.
    metric: revenue | checks | avg_check

    ⚠ ONLY FULL WEEKS. A six-day week is not comparable with a seven-day one
    and would show up as a dip the fly never saw in the data — the same rule
    the real detector follows.
    """
    import pandas as pd

    d = df if stores is None else df[df["store"].isin(stores)]
    d = d.copy()
    d["week"] = d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")
    agg = d.groupby("week").agg(
        revenue=("revenue", "sum"),
        checks=("check_id", "nunique"),
        days=("date", "nunique"),
    )
    agg = agg[agg["days"] >= 7]
    if metric == "avg_check":
        series = agg["revenue"] / agg["checks"]
    elif metric == "checks":
        series = agg["checks"]
    else:
        series = agg["revenue"]
    return [float(v) for v in series.to_numpy()]
