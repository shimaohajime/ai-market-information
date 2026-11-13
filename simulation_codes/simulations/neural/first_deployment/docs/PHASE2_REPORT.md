# Resource-Constrained Learning in a Microfounded Asset Market

**Step 03, Phase 2 | October 2026 | Research-in-progress technical report**

## Executive summary

We implemented and trained two forecasting architectures - a learnable stable diagonal linear state-space model (SSM) and a finite-context GRU - directly within the microfounded CARA/competitive-clearing market developed in Phase 1. Forecast heads support learned Gaussian conditional mean and, in the general model, state-dependent variance. The primary experiment holds the incumbent data-generating market law fixed and evaluates one-shot adoption; endogenous retraining and the post-adoption stationary market are explicitly excluded.

A 36-fit experiment across three eight-factor economies shows that greater state capacity in the aligned-payoff economy reduced heldout forecasting negative log likelihood (NLL) for both SSM and GRU, while **average conditional fresh-price likelihood precision declined**. The signs agree with the earlier information-crowd-out mechanism, but fresh precision **increased for one of three matched training seeds** in each architecture. The evidence is therefore *a suggestive computational pilot*, not a verification of the general theorem.

## 1. Economic and computational design

Fundamentals satisfy $f_{t+1}=Ff_t+G\eta_{t+1}$, with $f_t\in\mathbb R^8$. At date $t$, a one-period claim traded at $p_t$ pays $D_{t+1}=\ell'f_t+\varepsilon_{t+1}$ after the auction. Investor type $c$ receives $s_{ci,t}=L_cf_t+\xi_{ci,t}$ and publicly released $u_t=(p_{t-1},D_t,m_t)$. The public-history encoder $z_{c,t}$ is computed from the last 48 realizations of $u$; it never receives $p_t$ or $D_{t+1}$.

The general Gaussian *subjective* head is

$$\mu_{ci,t}(p)=a_c(z_{c,t})+\beta_c(z_{c,t})s_{ci,t}+\chi_c(z_{c,t})p,\qquad v_{ci,t}=v_c(z_{c,t})>0.$$

CARA utility implies $x_{ci,t}(p)=[\mu_{ci,t}(p)-R_fp]/[\gamma_cv_c(z_{c,t})]$. With a continuum of price-taking investors, competitive clearing yields

$$p_t=\frac{\sum_c w_{c,t}a_c(z_{c,t})+\left(\sum_c w_{c,t}\beta_c(z_{c,t})L_c\right)f_t-S_t}{\sum_c w_{c,t}(R_f-\chi_c(z_{c,t}))},\qquad w_{c,t}=\frac{\pi_c}{\gamma_cv_c(z_{c,t})}.$$

We constrain $|\chi_c(z)|<0.45<R_f$, which makes aggregate demand slope strictly positive in every experimental auction. Model A retains constant $\beta_c,\chi_c,v_c$ with nonlinear/learned public-state intercept $a_c(z)$; Model B permits all four head components to depend on $z$. A Gaussian working belief and payoff NLL training are **not** equivalent to objective Bayesian optimality or ex ante decision-aware learning.

The learning data are generated from Phase 1's **diagnostic**, untrained incumbent policies. Individual private signals are not averaged before fitting; the continuum law of large numbers is used **only to clear the market**. We fit Gaussian likelihood with Adam using a 65%/17%/18% chronological train/validation/test split. Architectures use 4 or 12 hidden dimensions and the same 48-period lookback. We compare across capacities *within* a family rather than treating parameter count as comparable computational cost across architectures.

## 2. The primary inherited-law experiment

We train candidate heads on one identical inherited market distribution $Q_0$. We then retain the realized historical $u_t$ and all incumbent investors' historical schedules, replace the target type's demand head, and clear each auction afresh at the inherited latent state and supply shocks. Crucially, the trained AI's current-price coefficient multiplies the newly clearing *candidate price*, but the historical encoder sees the inherited *past price*. Thus first deployment is separated from later stationary-market adjustment and subsequent retraining.

**E1: Aligned economy.** All investors' fresh private-signal loadings are proportional to the same target $\theta_t$. Conditioning on the inherited pre-auction public information, price has a Gaussian supply-shock channel with instantaneous likelihood precision $J_t=A_t^2/\sigma_\zeta^2$. We report the average $E[J_t]$, **not** mutual information. Across three training seeds:

| Architecture | Hidden states | Heldout NLL ↓ | Mean fresh precision J ↑ | OOS linear decoder gain |
|---|---:|---:|---:|---:|
| Linear SSM | 4 | 0.94043 | 0.27184 | 7.94% |
| Linear SSM | 12 | **0.93946** | 0.26559 | 7.81% |
| GRU | 4 | 0.95134 | 0.27605 | 7.75% |
| GRU | 12 | **0.94584** | 0.27392 | 8.17% |

Increasing hidden capacity reduces average NLL and fresh-channel precision under both architectures. However, each has 2/3 seeds with negative change and 1/3 with positive change in J. A second, independent inherited-market shock bank preserves the **sign for each trained checkpoint**, confirming that this sign heterogeneity is driven primarily by training outcomes rather than evaluation-path randomness. The training objective is nonconvex for GRU and nonlinear in SSM decay parameters, and its realized optimum need not preserve comparative statics for exact population minimizers.

**E2: Nonaligned economy.** A third investor receives a signal about a different factor, and dividend payoff contains factors outside the private target. Here $J=A^2/\sigma_\zeta^2$ is generally invalid for target information. We use an independent heldout *linear* decoder of $\theta_t$ from inherited $u_t$ with and without the counterfactual $p_t$:

| Architecture | States | Heldout NLL ↓ | Price-added decoder MSE improvement |
|---|---:|---:|---:|
| Linear SSM | 4 | 0.99254 | 3.74% |
| Linear SSM | 12 | 0.99292 | 3.63% |
| GRU | 4 | 1.00651 | 3.93% |
| GRU | 12 | 1.00113 | 3.96% |

The relationship is not uniformly monotone. In particular, the high-capacity SSM did not improve average NLL in this finite-data fit. These tests are **descriptive**, not exact conditional-information estimates.

## 3. Conditional-risk learning (E3)

E3 retains the more general factor loadings but the objective dividend innovation volatility varies with latent market state. Model B learns state-dependent $v_c(z)$ jointly with the public mean and private/current-price slopes. The heldout NLL comparison, averaging two seeds, is:

| Architecture (12 states) | Model A: constant variance | Model B: state-dependent variance and slopes |
|---|---:|---:|
| Linear SSM | 1.03551 | **1.03053** |
| GRU | 1.04616 | **1.03900** |

The state-dependent head improves probabilistic forecast fit, and a GRU-B actually deployed in the E3 fully recursive *fixed-policy* price market exhibits a realized subjective variance standard deviation of 0.1404. This is **not yet** a variance-only causal ablation: Model B also has more flexible conditional slopes.

## 4. Numerical microfoundation and independent validation

Seventeen unit/integration tests pass. They check timing and look-ahead, representation nestedness (an exact embedding of small SSM/GRU in larger instances), price-affine head reconstruction in physical units, positive demand slopes, clearing identities, checkpoint serialization, frozen auction replay, and functioning learned-head dynamic deployment.

The maximum aggregate demand-minus-supply error across all 36 one-shot scenarios is $1.33\times10^{-15}$; the minimum clearing slope is 0.795. A second inherited-market shock bank was used to independently re-evaluate all six matched E1 capacity pairs. Block-bootstrap uncertainty conditional on fixed fitted models is small, but **does not account for variation from the learning optimizer, training data, or economy calibration**. Results are not significance statements about architectural differences.

We further ran selected learned heads in the *fully price-endogenous but fixed-policy* market for 650 periods after 450 burn-in observations. The maximal clearing residual was $9.2\times10^{-16}$; numerically coupled price paths starting with different initial prices agreed on the evaluated tail. These checks do not establish a general stationary law, or retraining equilibrium existence/stability.

## 5. Computational accounting and interpretation

With an identical 48-period public input window, the SSM-4, SSM-12, GRU-4, and GRU-12 Model A designs use respectively **28, 76, 128, 664 trainable parameters**, with approximate per-date inference multiply-accumulate counts **1,156; 3,468; 5,188; 29,388**. These are algorithmic approximations, not profiled hardware FLOPs. Nestedness is within architectures. If a larger model fails to improve loss, the failure may reflect finite-data estimation or optimization rather than the fundamental absence of computational value.

For context, the inherited Phase 1 incumbent policy is **not a fitted low-capacity benchmark**, and its E1 fresh-channel precision is approximately 0.167. Comparisons between inherited diagnostic baseline and *any* trained policy must not be interpreted as capacity-driven crowd-out. The proper within-experiment capacity contrasts are the trained small-versus-large models on the same $Q_0$.

## 6. What this does and does not establish

**Established:** fully reproducible fitted public-history encoders, Gaussian mean/variance heads, price-taking CARA demand, exactly cleared counterfactual auctions, heldout predictive scoring, E1 conditional supply-channel precision, E2/E3 heldout linear decoder improvements, and descriptive stability checks.

**Not established:** a population-optimal/nested-class numerical proof, robust economy-wide crowd-out for all optimizer runs, welfare effects, causal separation of variance from mean learning, unconditional or conditional mutual information for general nonlinear economies, true computational-minimum complexity $C_\epsilon(Q)$, multi-epoch endogenous retraining or equilibrium reversal in 8d markets.

**Recommended next step:** Phase 3 should iteratively generate a new market law from the fitted policies, retrain *all* investor types, and compare initial impact, frozen-policy stationary change and retraining response. Before claiming generality, add an information-restricted oracle and an optimizer/budget convergence study, then repeat across richer market calibrations. Transformer and 16d scalability experiments come afterward.

## Reproduce

From the package root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run_phase2.py
python validate_phase2.py
```

The full artifact contains 36 learned checkpoints, the Phase 1 data panels, `phase2_results.json`, `phase2_models.csv`, `phase2_validation.json`, figures, and exact source code. See `README.md`.
