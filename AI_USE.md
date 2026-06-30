# AI Tool Usage Declaration

## Tools Used

**Claude (Anthropic)** — Used throughout the project for:

1. **Code scaffolding.** Initial structure of `utils.py`, `bc.py`, `naive_dqn.py`, `iql.py`, `irl_reward.py`, `cmdp.py`, `ope.py` was generated with Claude assistance and then reviewed, debugged, and modified to run correctly with the actual environment.

2. **Debugging.** Traced the `hidden=(256,256)` vs `hidden=(128,128)` size mismatch between BC and IQL loaders. Diagnosed why IQL Q-loss increases while Q-values stabilise (expected IQL behaviour, not divergence). Identified the warm-start CMDP failure (random Q_c + high λ corrupts the policy).

3. **Algorithm explanations.** Helped articulate the IQL expectile regression derivation and the DR estimator derivation for the OPE section.

4. **Report writing assistance.** Section structure and phrasing in REPORT.md was drafted with Claude, then edited for accuracy against actual experimental results.

## What Was Not AI-Generated

- All experimental numbers (Q-divergence curves, eval tables, OPE results) come from actual training runs logged in `logs/`.
- Architecture and hyperparameter choices (128×128 vs 256×256, IQL vs CQL, expectile τ=0.7, CMDP budget d=1.0) were made by the student based on the dataset statistics and course readings.
- The failure analysis (why IS/PDIS collapses, why CMDP needs more steps, the warm-start failure) was diagnosed from the actual output, not hallucinated.
- The CMDP hypothesis was written before running the experiments.

## Policy

Nothing submitted here cannot be explained, modified, or defended live. All code is understood line-by-line. The AI was used as an accelerator, not as an oracle.
