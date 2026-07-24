# OLA Project — Proposed Solution & Team Plan (5 people)

**Problem recap.** An agency bids in `N` advertising campaigns. Each round `t = 1..T`, one first-price auction per campaign: we set bid `b_i` from a small discrete set `B`, win if `b_i ≥ m_i` (highest competing bid), get utility `v_i − b_i` and pay `b_i`. Global constraints: total budget `B` over the whole horizon, and a **conflict graph** — campaigns joined by an edge cannot be bid on in the same round (the set of campaigns we bid on each round must be an *independent set* of the graph).

**Key modeling choices (shared by all requirements)**

- Bid set: e.g. `B = {0.0, 0.1, ..., 1.0}` with values `v_i ∈ (0, 1]`. Add a "no-bid" option (bid 0 / skip) per campaign.
- Feedback: bandit (we only see which auctions we won) — except Requirement 3, where full feedback (we observe `m_i`) is allowed.
- Baseline for regret plots: a **clairvoyant** that knows the true distribution(s) and solves the offline LP: maximize expected utility subject to expected per-round spend ≤ `ρ = B/T` and independent-set feasibility. For the non-stationary case, the clairvoyant re-solves per interval.
- Every experiment: run ≥ 20–50 independent trials, plot cumulative regret (mean ± std), cumulative spend vs. budget line, and % of budget used.

---

## Requirement 1 — Single campaign, stochastic environment
*(builds on notebooks 01, 02, 04)*

**Environment.** i.i.d. highest competing bid, e.g. `m_t ~ Beta(α, β)` (or a truncated Gaussian) rescaled to `[0, 1]`. Choose parameters so the optimal bid is interior (not the max bid), otherwise the problem is trivial.

**Algorithm A — UCB1, no budget.** Each bid `b ∈ B` is an arm. Reward of pulling `b`: `(v − b)·1[b ≥ m_t]`. Standard UCB1 with exploration bonus `sqrt(2 log t / n_b)`. Nice extra (worth one slide): since winning with `b` implies you would win with any `b' ≥ b`, you can discuss/compare a variant that updates all arms consistently — expect faster convergence.

**Algorithm B — UCB1 with budget (UCB-like "bandits-with-knapsacks").**
Keep, per bid `b`, a UCB on expected utility `f̄(b) + bonus` and an LCB on expected cost `c̄(b) − bonus` (cost of bid `b` is `b·1[win]`). Each round, solve the small LP over distributions on bids:
maximize Σ γ(b)·UCB_f(b)  s.t.  Σ γ(b)·LCB_c(b) ≤ ρ, and sample the bid from γ. Stop (or bid 0) when the remaining budget cannot cover the max bid. This is exactly the UCB-BwK scheme from the constrained-problems session (notebook 08), specialized to bidding.

**Plots.** Regret vs. clairvoyant LP, spend trajectory vs. the pacing line `ρ·t`, comparison A vs B when the budget is binding vs. slack.

---

## Requirement 2 — Multiple campaigns, stochastic environment
*(builds on notebooks 08, 09)*

**Environment.** Joint distribution over `(m_1, ..., m_N)`. Proposal: a common-factor model `m_i = clip(z + ε_i)` with `z` a shared random level and `ε_i` idiosyncratic noise → campaigns are correlated (realistic, and you can show C-UCB doesn't care about correlation, only marginals).

**Algorithm — Combinatorial-UCB with budget.**
- Base arms: pairs `(campaign i, bid b)`. Keep UCB on utility and LCB on cost per pair, exactly as in Req. 1B.
- Super-arm each round: choose an independent set `S` of the conflict graph and one bid per campaign in `S`, maximizing `Σ_{i∈S} UCB_f(i, b_i)` subject to `Σ LCB_c(i, b_i) ≤ ρ_t` (remaining-budget pacing `ρ_t = B_remaining / (T − t)`).
- The combinatorial oracle: for each campaign, its best (bid, UCB-value) is easy; picking the set is a **max-weight independent set** — with `N ≈ 5–10` campaigns just brute-force over independent sets (precompute them once), or use an ILP (`pulp`/`itertools`). Keep `N` small and say so honestly on the slides.
- Budget handling across campaigns: the pacing constraint is joint (one budget for all campaigns), which is the interesting part vs. Req. 1.

**Plots.** Regret vs. clairvoyant, budget depletion, and per-campaign win rates; show conflict graph used (e.g. a 5-node cycle or two cliques).

---

## Requirement 3 — Best-of-both-worlds, primal-dual (full feedback)
*(builds on notebooks 03, 08)*

**Environments.** (a) The stochastic joint environment from Req. 2. (b) A **highly non-stationary** one: the distribution of `m_i` changes quickly, e.g. every round its mean follows a fast sinusoid, or an adversarial sequence sampled from a distribution re-drawn every few rounds.

**Algorithm — Primal-dual with budget.**
- **Dual**: one multiplier `λ` for the budget, updated by online gradient descent: `λ_{t+1} = Π_{[0, 1/ρ]}( λ_t − η(ρ − spend_t) )`.
- **Primal**: a regret minimizer over the feasible bidding decisions maximizing the Lagrangian reward `Σ_{i∈S} (v_i − b_i)·1[win_i] − λ_t · Σ b_i·1[win_i]`.
  Design it for this specific problem (the hint on the slide): with **full feedback** we observe all `m_i`, so every round we can compute the Lagrangian utility of *every* super-arm, not just the played one → run **Hedge (multiplicative weights)** over super-arms. To keep it tractable, decompose: Hedge instance per campaign over bids using full-information Lagrangian payoffs, plus the independent-set selection on top (or Hedge directly over the precomputed independent-set/bid combos if `N` is small).
- Best-of-both-worlds story for the slides: in the stochastic world it competes with C-UCB; in the adversarial world C-UCB breaks but primal-dual keeps sublinear regret vs. the best fixed feasible strategy.

**Plots.** Both environments × {C-UCB, primal-dual}: regret, spend, and `λ_t` trajectory (nice to show pacing emerging automatically).

---

## Requirement 4 — Slightly non-stationary environment
*(builds on notebook 10)*

**Environment.** Piecewise-stationary: partition `T` into `K` intervals (e.g. 5 intervals of `T/5`); within each interval the joint distribution of `(m_1, ..., m_N)` is fixed; across intervals it changes abruptly (e.g. different Beta parameters per interval).

**Algorithms (extend Req. 2's C-UCB).**
1. **SW-C-UCB**: sliding window of size `W ≈ sqrt(T·K)`-ish — only the last `W` samples enter the means/counters.
2. **CD-C-UCB**: change detector (CUSUM or a simple mean-shift test on each base arm's recent window); on detection, reset that arm's (or all) statistics.
3. **Primal-dual** from Req. 3, run as-is.

**Comparison (explicitly required).** Same environment, same trials: cumulative regret vs. a per-interval clairvoyant for all three. Discuss the expected outcome: change detection wins on abrupt/rare changes, sliding window is more robust but pays for forgetting, primal-dual is robust but not tailored. If a result is surprising, keep it — the slides explicitly say discussing unexpected results is appreciated.

---

## Team split (5 people)

| Person | Owns | Main deliverable |
|---|---|---|
| 1 | **Core library + environments**: simulator (rounds, auctions, budget accounting), conflict graph + independent-set oracle, all 4 environment classes, clairvoyant/LP baselines, experiment runner + plotting utils | `core/`, `environments/`, baseline notebooks |
| 2 | **Requirement 1**: UCB1 + budgeted UCB | `req1/` + its result plots |
| 3 | **Requirement 2**: C-UCB with budget (uses Person 1's oracle) | `req2/` + plots |
| 4 | **Requirement 3**: primal-dual + non-stationary env, BoBW experiments | `req3/` + plots |
| 5 | **Requirement 4**: SW-C-UCB, CD-C-UCB, 3-way comparison | `req4/` + plots |

Everyone: writes the slides for their own requirement (≈ 3–4 slides each, 20-minute talk total); Person 1 also does the intro/model slides. Cross-review in pairs (2↔3, 4↔5) before merging.

**Suggested repo structure**

```
ola-project/
  core/            # agent base class, budget tracker, runner, plots
  environments/    # stochastic, joint, highly_nonstat, piecewise
  agents/          # ucb1, ucb_budget, cucb_budget, primal_dual, sw_cucb, cd_cucb
  experiments/     # one script/notebook per requirement → figures/
  figures/
  slides/
```

**Interfaces to agree on day 1** (this is what makes parallel work possible):
`agent.bid() -> {campaign: bid}`, `env.round(bids) -> {campaign: (won, m_i or None)}`, `agent.update(feedback)`. Once frozen, the 4 algorithm owners work independently against Person 1's simulator.

**Timeline — target: done by 24 August** (4 weeks, comfortably ahead of the September submission):

- **Week 1 (Jul 24 – Jul 31)**: freeze the agent/environment interfaces; Person 1 ships simulator v1 + stochastic environment + clairvoyant baseline; Persons 2–5 re-run their relevant practical notebook and sketch their agent against the interface.
- **Week 2 (Aug 1 – Aug 7)**: Requirement 1 fully done with plots; Requirement 2's C-UCB working (independent-set oracle from Person 1); first primal-dual prototype; non-stationary environments ready.
- **Week 3 (Aug 8 – Aug 14)**: Requirements 3 and 4 complete; run the required 3-way comparison (SW vs CD vs primal-dual); cross-review in pairs (2↔3, 4↔5) and fix issues.
- **Week 4 (Aug 15 – Aug 24)**: freeze code, re-run all experiments with final seeds/trial counts, polish figures, write slides (~3–4 per requirement + intro), dry-run the 20-minute presentation once as a group.

Mid-August is holiday season — agree now on who is away when, and front-load accordingly; the pair-review setup means no requirement depends on a single person being reachable.
