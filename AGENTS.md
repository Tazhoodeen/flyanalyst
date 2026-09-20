# Flyanalyst

Inherits the Stonkfly discipline. Retail-specific rules replace the trading
ones; everything about the graph and the reinforcement signal stands as
written upstream.

- Preserve the full retained MaleCNS v1.0 graph. No pruning, no scripted
  verdicts presented as neural output, no LLM deciding what the network
  "found", no hidden selection of the runs that look best.
- Separate sales observations, sensory proxies, neural propagation,
  plasticity, and fixed decoding. The renderer may reshape a chart; it must
  never choose a verdict.
- **Do not claim the network analyses, understands or predicts sales.** The
  connectome supplies anatomy and no retail knowledge whatsoever. The mapping
  of `DNp20` turn neurons onto "worse / better" is a convention imposed here.
- **Every claim ships with its control.** A result on real weeks means nothing
  without the shuffled comparison beside it, and one shuffle is not a control —
  report the spread across seeds. Publishing the real number alone, even when
  it looks good, is the failure this repository exists to demonstrate.
- Report negative results as results. "Indistinguishable from noise" is the
  finding, not a bug to tune away. Do not add parameters until the difference
  becomes significant.
- Learning stays off while scoring. With plasticity on, each frame rewrites the
  weights and the run measures the order of the weeks as much as the weeks —
  exactly what the shuffled control is meant to isolate.
- Only full weeks enter a series. A six-day week is not comparable with a
  seven-day one and would appear as a dip that never happened.
- Reinforcement into identified dopamine cells is an engineered input. Do not
  claim modeled pain, pleasure, consciousness, or validated learning.
- Real business data stays out of the repository. The bundled database is
  synthetic; shop names, real revenue and client identifiers are never
  committed. Data, checkpoints and logs stay ignored.
- Keep README short and lead with the result, including when the result is
  that nothing was found.
