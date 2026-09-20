"""Read the fly's turn neurons as a verdict about a retail week.

WHAT THIS DOES. Upstream decodes the same neurons as BUY / SELL / HOLD. The
anatomy is identical here; only the label changes:

    DNp20 left  vs right  ->  "week looks worse" vs "week looks better"
    DNpe017 (gate)        ->  whether to say anything at all

Those are real, named descending neurons. In a living fly they steer turning;
DNp20 activity biases the animal left or right. Nothing about them evolved to
read a sales chart. The mapping below is a convention we impose, not a
discovery.

WHY THE GATE MATTERS MOST. The interesting output is not "up" or "down" —
a coin gets that right half the time. It is whether the gate fires at all,
i.e. whether the network reacts to this week differently from any other. That
is the only claim worth measuring against a control.

HONEST LIMITS (same discipline as upstream AGENTS.md):
  * No claim that the network analyses, understands or predicts sales.
  * Agreement with a real revenue dip is coincidence until measured against
    shuffled weeks — see `control_shuffled` below, which is the whole point.
  * The reinforcement signal is engineered, not a modeled pain receptor.
"""

from __future__ import annotations

import numpy as np


class RetailDecoder:
    """Same neurons as upstream Decoder, retail labels.

    Kept as a separate class rather than a patch so the upstream trading
    decoder stays intact and diffable against its origin."""

    VERDICTS = ("QUIET", "WORSE", "BETTER")

    def __init__(self, ids, annotation, threshold):
        types = annotation.type.fillna("")
        sides = annotation.somaSide.fillna("")
        self.left = np.flatnonzero(types.eq("DNp20") & sides.eq("L"))
        self.right = np.flatnonzero(types.eq("DNp20") & sides.eq("R"))
        self.gate = np.flatnonzero(types.eq("DNpe017"))
        if not len(self.left) or not len(self.right) or not len(self.gate):
            raise RuntimeError("Missing annotated BCI outputs")
        self.threshold = threshold
        self.identities = {
            k: [str(ids[i]) for i in getattr(self, k)]
            for k in ("left", "right", "gate")
        }

    def decode(self, counts, seconds):
        # Mean rates, not sums: the two sides hold different numbers of cells
        # and summing would bake in a constant bias toward the larger side.
        left = float(np.mean(counts[self.left]) / seconds)
        right = float(np.mean(counts[self.right]) / seconds)
        difference = right - left
        gate = int(counts[self.gate].sum())

        if not gate or abs(difference) < self.threshold:
            verdict = "QUIET"
        elif difference > 0:
            verdict = "BETTER"
        else:
            verdict = "WORSE"

        return {
            "verdict": verdict,
            "difference": difference,
            "gate_spikes": gate,
            "left_rate": left,
            "right_rate": right,
        }


def control_shuffled(series, seed: int = 0):
    """Same weeks, order destroyed. The control every claim is measured against.

    ⚠ THIS IS THE POINT OF THE PROJECT, not an afterthought. If the network
    reacts the same way to shuffled weeks as to real ones, then whatever it is
    doing carries no information about the business — and saying so plainly is
    a more useful result than a chart that looks clever.

    Mirrors what FlyDoom does with rewired graphs: keep the statistics, remove
    the structure, see whether the output notices.
    """
    rng = np.random.default_rng(seed)
    out = list(series)
    rng.shuffle(out)
    return out


def summarise(results) -> dict:
    """Aggregate a run into the only numbers worth reporting."""
    verdicts = [r["verdict"] for r in results]
    n = len(verdicts) or 1
    return {
        "weeks": len(verdicts),
        "quiet": verdicts.count("QUIET"),
        "worse": verdicts.count("WORSE"),
        "better": verdicts.count("BETTER"),
        "spoke_pct": round((n - verdicts.count("QUIET")) / n * 100, 1),
        "mean_gate": round(float(np.mean([r["gate_spikes"] for r in results])), 1)
        if results else 0.0,
    }
