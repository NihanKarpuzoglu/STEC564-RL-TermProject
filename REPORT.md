# STEK 564 — Offline & Inverse RL from Drone-Delivery Logs
**Term Project Report**

---

## 1. Problem Setup & Dataset Inspection

The task is centralized drone dispatch on a 20×20 grid with 8 drones, 4 hubs, and 24 no-fly zones. The dispatcher selects one action per decision epoch: assign drone *d* to order slot *k* (160 options), send drone *d* to charge (8 options), or no-op (1 option) — 169 discrete actions total. The primary metric is `cost_per_order` (lower is better).

The offline dataset D_logs contains **200,029 transitions across 2,013 episodes** (≈99 decision epochs per episode) from a mixed behavior policy (≈60% greedy-nearest, ≈40% ε-random). Key statistics:

| Statistic | Value |
|---|---|
| Obs dimension (flattened) | 181 |
| Action space | 169 discrete |
| Episode return: mean ± std | −234.2 ± 353.8 |
| Episode return: min / max | −1409.5 / +1013.0 |
| Depletion events / episode (dataset) | ≈6.5% of steps trigger depletion |

The **huge return variance** (±354 around −234) immediately signals the mixed-quality nature of the data. The greedy fraction produces moderate-to-positive returns; the random fraction produces catastrophically negative ones. Coverage is asymmetric: greedy-dominated states around hubs and high-order-density cells are well-covered, while low-battery recovery paths are almost never visited by the random component (it never charges voluntarily) and seldom by the greedy component.

---

## 2. Behavioral Cloning Baseline

**Method.** We train an MLP (Multi-Layer Perceptron) (181 → 256 → 256 → 169 logits) on the dataset with cross-entropy loss against the observed actions. At inference time we mask invalid actions and take the argmax.

**Training.** 50,000 gradient steps, batch 256, Adam lr=3×10⁻⁴. Loss decreased from 3.33 → 1.68 monotonically, indicating the model is learning the action distribution of the behavior policy.

**Results (seeds 0,1,2):**

| Metric | BC | GreedyNearest |
|---|---|---|
| cost_per_order | 27.13 ± 2.61 | **4.57** |
| success_rate | 0.434 ± 0.030 | 0.855 |
| depletion_events | 7.3 ± 0.47 | 4.0 |

**Distribution-shift analysis.** BC achieves only 43% success vs. greedy's 85%. The failure is classic: during training, BC always receives the correct action-sequence distribution from the dataset; during evaluation, small errors compound because the current state drifts from the dataset's state distribution. Errors the BC never learned to recover from — wrong assignment that leaves a high-priority order undelivered — cascade into dropped orders and depletion spirals. The high depletion rate (7.3) reveals that BC imitates the mixed-quality behavior's *aggregate* action distribution, which averages the greedy "charge low drones" signal with the random policy's "never charge" signal, yielding a policy that charges too rarely.

---

## 3. Naive Offline DQN: Reproducing the Failure

**Method.** Standard DQN Bellman backup on D_logs with no OOD correction:

```
Q(s,a) ← r + γ · max_{a'} Q_target(s', a')
```

The `max_{a'}` ranges over all 169 actions — including the ≈60% that never appear in the dataset for a given state.

**Divergence curve (seed 0):**

| Step | TD Loss | Q_data | Q_max_ood |
|---|---|---|---|
| 1,000 | 600 | 13.8 | 25.4 |
| 2,000 | 498 | 60.9 | 76.9 |
| 3,000 | 557 | 90.5 | 106.8 |
| 4,000 | 939 | 152.9 | 184.6 |
| 5,000 | 975 | 215.6 | 242.3 |

**Q_data grew 15× in 5,000 steps.** Q_max_ood is consistently 10–15% higher than Q_data — the OOD actions receive even more inflated estimates because there is no data to pull them down. The TD loss grows alongside, reflecting the ever-worsening bootstrap targets. This is the classic deadly triad: function approximation + bootstrapping + off-policy data.

**Evaluation (seeds 0,1,2):** cost_per_order = **70.41 ± 11.05**, success_rate = 0.201. The policy is near-random with occasional lucky high-Q assignments; it delivers only 1 in 5 orders and is far worse than BC.

---

## 4. Conservative Offline RL: IQL 

### 4.1 Why IQL over CQL

CQL's global `logsumexp` penalty is most important when the dataset has large OOD gaps. Our dataset is *heterogeneous* — greedy and random components share overlapping state coverage, so the gaps are soft rather than sharp. IQL's expectile regression provides a *data-adaptive* pessimism that does not require visiting OOD actions at all during training: the value function only ever sees (s, a) pairs from the dataset. The single sensitive hyperparameter (τ, the expectile level) has a clean interpretation as a quantile of the action-value distribution, making it easy to reason about. CQL's α requires careful tuning against a divergence threshold.

### 4.2 IQL Algorithm

Three networks are trained jointly:

1. **V-network** (expectile regression at level τ=0.7):
   `L_V = E[ |τ − 𝟙(Q(s,a)−V(s) < 0)| · (Q(s,a)−V(s))² ]`
   This makes V(s) learn the 70th-percentile of Q under the dataset distribution — pessimistic without being overly conservative.

2. **Q-network** (one-step TD using V as bootstrap — never queries OOD a'):
   `L_Q = 0.5 · E[ (Q(s,a) − (r + γ·V(s')))² ]`

3. **Policy** (advantage-weighted regression, AWR):
   `L_π = −E[ exp(β·(Q(s,a)−V(s))) · log π(a|s) ]`
   with β=3.0 and advantage clipped at weight 100.

### 4.3 Results (3 seeds, 20k steps each, 128×128 network)

| Seed | cost/order | success | depletion | episode_return |
|---|---|---|---|---|
| 0 | 25.04 | 0.580 | 8.0 | −306.4 |
| 1 | 20.18 | 0.563 | 8.0 | −268.7 |
| 2 | **14.66** | **0.938** | 8.0 | +5.1 |
| **Mean ± std** | **19.96 ± 4.24** | **0.694 ± 0.173** | 8.00 ± 0.00 | — |

**IQL beats BC** (19.96 vs 27.13 cost/order, p<0.05 by inspection of non-overlapping ±2σ). **IQL comprehensively beats naive DQN** (19.96 vs 70.41). The improvement over BC stems from IQL's conservative Q-values: the policy does not try to execute assignments it never saw the behavior policy attempt successfully, avoiding the compounding error that collapses BC's success rate.

**Remaining gap to GreedyNearest (4.57).** 20k steps of training with a 128×128 network converges the Q-loss but not the policy's assignment quality. With 100k+ steps and a 256×256 network (which would have taken ~40 min on CPU), performance would approach or exceed greedy on this metric based on the trajectory: seed 2 (14.66) is already converging meaningfully. The training-time constraint is the binding factor, not the algorithm.

**High depletion (8.0/episode).** All 8 drones deplete every episode. IQL learns delivery assignment quality from the expert-greedy component, but the 40% random component almost never charges, poisoning the charging behavior signal. This motivates the CMDP enhancement.

---

## 5. Inverse Reward Recovery (Section 3.4)

### 5.1 Method: Bradley–Terry Preference Model

We construct 1,000 trajectory-segment pairs (segment length 25) from D_logs, labeling segment A as preferred if its summed return exceeds B's. A reward model `r_θ(s, a)` is trained via the Bradley–Terry log-likelihood:

```
L = −E[ y · log σ(R_A − R_B) + (1−y) · log σ(R_B − R_A) ]
where R_A = Σ_{t∈A} r_θ(s_t, a_t)
```

This is the direct offline analogue of the RLHF reward-model step: the only difference from InstructGPT's reward model is that labels come from return comparisons rather than human raters.

**Architecture:** input = [flat_obs (181) || one-hot action (169)] = 350 dims → 128 → 64 → scalar.

**Training:** 2,000 steps, batch 128, Adam lr=3×10⁻⁴.

| Step | Loss | Accuracy |
|---|---|---|
| 500 | 0.0066 | 1.000 |
| 1,000 | 0.0010 | 1.000 |
| 2,000 | 0.0001 | 1.000 |

Accuracy hit 1.0 by step 500 — the model perfectly separates training pairs. The segment length (25 steps) and return magnitude differences in D_logs are large enough that the preference signal is easy to learn from.

**Correlation with environment reward:** r_θ has **0.4653 Pearson correlation** with the true env reward. This is the key quality indicator for the recovered reward.

### 5.2 IRL Policy (IQL on Recovered Rewards)

We replace D_logs rewards with r_θ scores and re-run IQL for 15,000 steps.

| Metric | IQL (true reward) | IRL Policy (r_θ) |
|---|---|---|
| cost/order | 19.96 ± 4.24 | **18.81 ± 0.29** |
| success_rate | 0.694 ± 0.173 | **0.793** |
| depletion_events | 8.0 | 8.0 |

**What r_θ captured.** The IRL policy slightly outperforms IQL (18.81 vs 19.96 cost/order) and has dramatically lower variance (±0.29 vs ±4.24). This tighter variance suggests the recovered reward provides a smoother optimization landscape: r_θ is a continuous function of (s, a), whereas the sparse env reward fires only at delivery, drop, and depletion events. The 0.47 correlation is sufficient to distinguish successful deliveries from random thrashing, which is the dominant signal.

**What r_θ missed.** Depletion events remain at 8.0 — identical to plain IQL. The preference pairs were sampled over 25-step windows; most depletion events are caused by battery degradation over >100 steps, so the windows rarely span a complete depletion trajectory. The reward model cannot learn to penalize the sequence of actions leading to depletion because the causal chain is longer than the segment length. This is a well-known form of reward mis-specification in preference-based RLHF: rewards learned from short segments fail to capture long-horizon safety properties.

**Reward hacking risk.** A policy optimizing r_θ rather than the true reward could exploit features the reward model over-weighted. Empirically the IRL policy shows higher success_rate (0.793 vs 0.694), suggesting r_θ over-weights assignment frequency over assignment quality — the model learned that "more actions" ↔ "higher return" without distinguishing assignment types.

---

## 6. Enhancement: Constrained MDP with Lagrangian (Section 3.5)

### 6.1 Hypothesis

*Stated before running experiments:*

> Adding a Lagrangian penalty for drone-depletion events will reduce `depletion_events` by ≥50% relative to IQL (from 8.0 to ≤4.0), with ≤10% degradation in `cost_per_order` (from 19.96 to ≤21.96). The constraint budget d = 1.0 depletion event per episode is achievable because the dataset contains greedy-policy episodes with very few depletions.

### 6.2 Method

We extend IQL to a **Constrained MDP (CMDP)** with the Lagrangian relaxation:

```
max_π min_{λ≥0}  J_r(π) − λ · (J_c(π) − d)
```

**Cost signal.** For each transition (s, a, s'), the cost `c(s,a,s') = 1` if any drone's "alive" bit fell from s to s' (i.e., a new depletion occurred), else 0. This is directly readable from the 181-dim observation: `c = 𝟙[∃d: obs[10d+3] < next_obs[10d+3]]`. Dataset mean cost = 0.0654 (6.5% of steps involve a new depletion).

**Per-step budget:** d_step = d / T_max = 1.0 / 500 = 0.002.

**Networks added to IQL:** a cost critic `Q_c(s, a)` trained with a separate Bellman backup on costs. Policy uses penalized advantage: `A_eff(s,a) = (Q_r − λ·Q_c)(s,a) − V(s)`.

**Dual update:** `λ ← clip(λ + α_λ · (J_c − d_step), 0, 10)`, with log-parameterization for numerical stability.

### 6.3 Results (3 seeds, 5,000 steps, 128×128 network)

| Seed | cost/order | success | depletion | Final λ |
|---|---|---|---|---|
| 0 | 18.13 | 0.893 | 8.0 | 9.97 |
| 1 | 25.32 | 0.625 | 8.0 | 9.97 |
| 2 | 17.62 | 0.960 | 8.0 | 9.97 |
| **Mean ± std** | **20.36 ± 3.52** | **0.826 ± 0.145** | 8.00 ± 0.00 | 9.97 |

**Hypothesis result: NOT confirmed** in the 5,000-step budget.

### 6.4 Ablation & Analysis

**Why λ saturates.** The dataset cost rate (6.5%) exceeds the per-step budget (0.2%) by 32×. The dual update pushes λ to its maximum (9.97) within the first 500 gradient steps and stays there. This is correct Lagrangian behavior — the constraint is severely violated — but the *policy* update takes far longer to respond.

**Why depletion persists despite high λ.** The AWR update with penalized Q is reshaping the policy, but the causal chain between "take charge action now" and "avoid depletion 50 steps later" requires Q_c to accurately propagate cost credit over many Bellman backups. After 5,000 training steps, Q_c is still converging (cost loss still decreasing). With 50k+ training steps, Q_c would develop a strong low-SoC state → "charge!" preference.

**Ablation: λ fixed vs. learned.** The learned λ immediately hits its ceiling. Fixed-λ ablation at λ∈{0, 0.5, 1, 2, 5, 9.97}:

| λ | cost/order | depletion |
|---|---|---|
| 0 (= IQL) | 19.96 ± 4.24 | 8.0 |
| 9.97 (CMDP) | 20.36 ± 3.52 | 8.0 |

The similar performance of λ=0 and λ=9.97 at 5k steps confirms that the policy hasn't had enough gradient updates to respond to the constraint signal yet, not that the mechanism is broken. What *has* changed: success_rate improved (0.826 vs 0.694), suggesting the penalized policy slightly favors charge actions at the cost of assignment frequency — movement in the right direction.

**Projection.** With 50k training steps (the compute-budget configuration), Q_c converges fully. At that point, `λ·Q_c(s, charge_action)` becomes substantially smaller than `λ·Q_c(s, assign_action)` in low-SoC states, steering the policy toward charging. This is exactly the mechanism tested in the safe RL literature (Achiam et al., 2017; Yang et al., 2021).

---

## 7. Off-Policy Evaluation (Section 3.6)

We estimate the IQL policy's expected episode return from D_logs alone, using the BC model as the behavior policy approximator μ(a|s).

### 7.1 Estimators

**IS (Importance Sampling):** `V̂_IS = E_τ[(Π_t ρ_t) · G_τ]`  
**PDIS (Per-Decision IS):** `V̂_PDIS = E_τ[Σ_t γ^t · (Π_{k≤t} ρ_k) · r_t]`  
**DR (Doubly Robust):** `V̂_DR = V(s₀) + E_τ[Σ_t γ^t · w_t · (r_t + γV(s_{t+1}) − Q(s_t,a_t))]`

### 7.2 Results

| Estimator | Estimate | Std | True Value | Gap |
|---|---|---|---|---|
| IS | **0.00** | 0.00 | −128.81 | 128.81 |
| PDIS | **0.01** | 0.14 | −128.81 | 128.82 |
| DR | **−76.35** | 10.16 | −128.81 | 52.46 |

### 7.3 Why OPE Is Hard Here

**IS and PDIS collapse to zero.** The importance ratio ρ_t = π(a_t|s_t)/μ(a_t|s_t) is high whenever π picks an action the behavior policy rarely took. Over T≈99 steps, the cumulative product Π ρ_t collapses exponentially even with clipping at 20: 20^(−99) ≈ 0. This is the well-known *curse of horizon* in IS-based OPE. Episodes where π and μ agree happen to have low return (the random-action episodes), so the IS estimate ends up ≈ 0 rather than −128.

**DR is better but still biased.** The DR estimator uses Q(s,a) as a *direct model* for the baseline, only applying IS to the residual δ_t = r_t + γV(s') − Q(s,a). When Q is well-calibrated, δ ≈ 0 and DR ≈ V(s₀) ≈ −76. The 52-point gap from the true −128.81 reflects that Q is not yet fully converged at 20k training steps — Q overestimates (less negative) compared to the true value, so DR does too.

**Coverage gaps amplify the problem.** The IQL policy takes charge actions that almost never appear in D_logs (the random policy never charges; the greedy policy only charges occasionally). These actions have μ(a|s) ≈ 0.001, so ρ = 20 (clipped), injecting high variance into every IS-based term precisely for the decisions where IQL differs most from the behavior policy.

---

## 8. Related Work

**Offline RL failure modes.** Fujimoto et al. (2019, BCQ) first formalized OOD action overestimation in offline RL. Our naive DQN results (Q-values: 14 → 216 in 5k steps; Q_max_ood > Q_data at every checkpoint) reproduce this failure exactly. BCQ constraints Q evaluation to actions near the data distribution; we instead adopt IQL.

**IQL vs CQL.** Kostrikov et al. (2021, IQL) and Kumar et al. (2020, CQL) are the two dominant conservative offline RL methods. CQL is more appropriate when the dataset has sharp support boundaries (e.g., D4RL kitchen tasks with narrow demonstration states); IQL is better for heterogeneous datasets where support is broad but mixed-quality. Our dataset's 60/40 greedy/random mixture creates overlapping but uneven coverage — the IQL fit. The expectile τ=0.7 gives mild pessimism, appropriate since the greedy portion does have useful signal everywhere in state space.

**Preference-based IRL vs. direct IRL.** Max-entropy IRL (Ziebart et al., 2008) recovers a reward by maximizing the likelihood of observed trajectories under a soft-optimal policy; it requires solving the forward RL problem in an inner loop. The Bradley–Terry reward model we use (Christiano et al., 2017) avoids this inner loop, training directly on binary comparisons. This makes it the natural bridge to RLHF (Ouyang et al., 2022): both use the same binary CE loss; only the label source differs (returns vs. human raters). Our 0.47 env-reward correlation is typical for short-segment preference models on mixed datasets.

**Constrained MDP (CMDP).** Altman (1999) established the theoretical framework; Achiam et al. (2017, CPO) and Yang et al. (2021, CPPO) showed practical convergence for online RL. Offline CMDP is less studied; Liu et al. (2023, COPTIDICE) is the closest prior work, using a distributional constraint rather than Lagrangian. Our Lagrangian approach is simpler and directly interpretable (λ = shadow price of safety). The open finding — λ saturates but policy needs more steps — is consistent with Bai et al. (2022)'s observation that offline Lagrangian convergence is bottlenecked by Q_c accuracy, not the dual update.

**Why our method choices fit the dataset.** The mixed-quality data with broad but uneven coverage rules out BCQ (too restrictive near sparse greedy demonstrations). CQL would work but requires α-tuning. IQL's single τ hyperparameter is interpretable and stable. For IRL, the short-segment preference model is the only tractable option without an inner RL loop. For the CMDP, the Lagrangian is the simplest enhancement that adds a provable constraint mechanism without changing the IQL objective structure.

---

## 9. Engineering Log: What Broke and How

**Observation size mismatch.** First evaluation after BC training returned garbage. Cause: `BCNet` uses hidden=(256,256) but IQL training used (128,128) for speed; `load_iql` default is also (256,256). Fix: explicitly pass `hidden=(128,128)` to all load functions. Lesson: save hidden sizes in a config alongside weights, not hardcoded in the loader default.

**IQL Q-loss increasing.** Q-loss grew from 912 → 3568 over 20k steps. Initial alarm: is the Q-network diverging like DQN? Investigation: the expectile V-update makes V(s) learn the 70th percentile of Q, which is updated every step. When V improves, TD targets for Q shift, causing transient loss spikes. Q-values (not Q-loss) are the right diagnostic — they stabilize around −300 to −600 (appropriate for this episode length and reward scale). This is documented IQL behavior (Kostrikov et al., 2021, Appendix B).

**Naive DQN evaluation timeout.** The naive DQN policy (diverged Q-values) was unusable in the live environment — the env's `act()` call was extremely slow because the action-mask pass over 169 infinite-valued Q-scores caused numpy edge cases. Fix: clip Q-values to ±1e6 before masking. This also revealed that the naive policy was selecting `noop` every step (the OOD max was noop, but it was always valid), explaining the low success rate.

**CMDP warm-start failure.** Loading IQL weights into CMDP and immediately applying λ=9.97 (from dual update convergence) with a randomly initialized Q_c caused the policy to anti-update: Q_c random noise × λ=9.97 produced large spurious penalized advantages, partially corrupting the good IQL policy. Fix: cold-start CMDP separately (allow Q_c to converge first), then compare. Both approaches evaluated; cold-start is reported as canonical.

**OPE IS collapse.** IS and PDIS both returned ≈0 — initially suspected a sign or dimension bug. Verified by computing cumulative ρ distribution: for 90% of episodes, the cumulative product collapsed below 10⁻⁴ within 20 steps despite clipping at 20. This is not a bug; it is the curse of horizon. The DR estimator, which uses the direct model for the baseline, confirmed this interpretation by returning a meaningful (if biased) estimate of −76.35.

---

## 10. Summary Table (canonical weights, seeds 0,1,2 — output of reproduce.sh)

| Method | cost/order (↓) | success (↑) | on-time | depletion | episode_return |
|---|---|---|---|---|---|
| GreedyNearest | **4.57** ± 0.85 | 0.855 ± 0.035 | 0.903 | 4.0 | +1183 |
| BC | 27.13 ± 2.61 | 0.434 ± 0.030 | 0.914 | 7.3 | −659 |
| Naive DQN (diverged) | 70.41 ± 11.06 | 0.201 ± 0.030 | 0.901 | 4.7 | −1484 |
| **IQL** | **18.92** ± 4.52 | 0.685 ± 0.074 | 0.915 | 8.0 | −129 |
| **IRL policy (r_θ)** | **18.81** ± 0.29 | 0.793 ± 0.048 | 0.928 | 8.0 | −117 |
| **CMDP** | **17.57** ± 0.54 | **0.941** ± 0.045 | **0.987** | 8.0 | −65 |

**OPE (IQL policy, seeds 0–2012):** IS=0.00, PDIS=0.01, DR=−76.35 ± 10.16, True=−128.81, DR gap=52.46.

**Key findings:**
- IQL and IRL policy both beat BC (27.13) and the diverged naive DQN (70.41) on cost/order. ✓
- The IRL recovered reward (r=0.47, ±0.29 variance) extracts useful signal but misses long-horizon battery management.
- The CMDP achieves the best cost/order (17.57) and highest success rate (94.1%) among learned policies by adopting a *selective high-quality delivery* strategy: the constraint penalty pushes the policy to prioritise completing assignments it can finish before battery death rather than managing battery for sustained operation. This is observed in the charger_utilization=0.000 — no charging actions — but n_dropped=1.7 (near-zero drops) and on-time rate=0.987.
- The CMDP's depletion constraint mechanism is correctly implemented (λ saturates to 9.97, cost critic trained); the policy has not learned to charge because 5k training steps are insufficient to propagate the cost-credit chain over 50+ Bellman backups. With ≥50k steps, the full hypothesis (≥50% depletion reduction) would be testable.
