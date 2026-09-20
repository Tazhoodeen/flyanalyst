# Flyanalyst

A fly-connectome simulation reading weekly retail sales charts. Real neural
output, real shop data, **and a control that says it means nothing**.

**Result up front:** the network signalled on 37% of real weeks and on
31% ± 11 of the same weeks shuffled into random order. `z = +0.53`. Its
reaction to a real retail business is statistically indistinguishable from its
reaction to noise. That is the finding, and it was the point of building this.

## How it works

Weekly revenue of a three-shop retail network becomes an RGB chart. It
stimulates 3,335 brightness inputs and 811 R8 colour inputs in the retained
**MaleCNS v1.0 graph: 166,700 neurons, 25.6 million connections**. A fixed
neural readout of named descending neurons produces a verdict:

| neurons | in a living fly | here |
|---|---|---|
| `DNp20` left vs right | steers turning | "week looks worse" vs "better" |
| `DNpe017` | gating | whether to say anything at all |

Those neurons are real and measured. The mapping onto retail is a convention
imposed by this repository, not a discovery. Nothing in a fly's optic lobe
evolved to read a sales chart.

## The control is the experiment

Every run scores the same weeks twice: in real order, and shuffled five times
with different seeds. Shuffling keeps every value and destroys only the
sequence. If the network reacts the same way to both, it is responding to the
texture of a line rather than to the business behind it.

It does react the same way. Five control runs ranged from 14.8% to 44.4%
signalling, and the real 37% sits unremarkably inside that spread.

⚠ **One control is not enough.** Seed 0 alone would have produced "the fly
reacts twice as often to real data!" — 37% against 14.8%. The spread across
seeds is what makes the comparison honest.

## Run it

Python 3.11, a C++17 compiler, macOS/Linux. Several GB for the dataset;
16 GB RAM recommended.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m stonkfly prepare          # downloads MaleCNS v1.0
python stonkfly/retail_run.py --db demo/demo_sales.duckdb --weeks 26
```

The bundled `demo/demo_sales.duckdb` is **synthetic**: shaped like real retail
weekly data, containing no real business figures. Point `--db` at your own
checks database to run it on yours.

Options: `--metric revenue|checks|avg_check`, `--window` (weeks drawn per
frame, default 13), `--seeds` (control repetitions, default 5).

## Why 13 weeks per frame

Upstream keeps 100 ticks because crypto ticks arrive seconds apart and the
chart is a texture. Weekly retail data is sparser. Measured on real data:

| weeks shown | pixels per week | readable |
|---|---|---|
| 143 | 2.1 | no |
| 52 | 5.8 | no |
| 26 | 11.8 | yes |
| 13 | 24.5 | clearly |

13 also matches the baseline window a conventional significance test would
use, so both look at the same stretch of history.

## What this is not

- Not an analyst. It does not analyse, understand or predict sales.
- Not evidence that connectome simulations can do business analytics.
- Not a product, and not validated for any business use.

The honest version of the headline is: an impressive mechanism is not the same
thing as a correct answer, and the only way to tell them apart is a control.

## Credits

Forked from [Stonkfly](https://github.com/nftechie/stonkfly), which is itself
built on DOOMFLY's connectome importer, inferred visual projection, spiking
kernel and plasticity. Connectome data: MaleCNS v1.0 (FlyEM / Janelia,
Cambridge, MRC LMB, Google Research), CC-BY 4.0. See `THIRD_PARTY.md`.

Code MIT, as upstream.
