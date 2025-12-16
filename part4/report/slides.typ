#import "@preview/touying:0.6.1": *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.3.2": *
#import cosmos.clouds: *

#show: show-theorion

#import themes.metropolis: *
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-common(frozen-counters: (theorem-counter,)),
  config-info(
    title: [LINMA2222 Project Presentation],
    subtitle: [Optimal Portfolio Strategy],
    author: [Lucas Ahou • Aymeric Couplet],
    date: datetime.today(),
    institution: [UCLouvain],
    logo: image("typst-uclouvain-frontpage/UCL_blanc.png", height: 0.8cm),
  ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

////////////////////////////////////////////////////////////
// Styling helpers (bold + colors)
////////////////////////////////////////////////////////////
#let c-blue = rgb(0, 92, 164)
#let c-red = rgb(180, 40, 40)
#let c-green = rgb(0, 120, 70)
#let c-gray = rgb(120, 120, 120)

#let emph(body) = text(weight: "bold", body)
#let blue(body) = text(fill: c-blue, weight: "bold", body)
#let red(body) = text(fill: c-red, weight: "bold", body)
#let green(body) = text(fill: c-green, weight: "bold", body)
#let muted(body) = text(fill: c-gray, body)

#let clipop = $op("clip")$

/// Helpers to avoid underscore issues in math labels
#let picl = $pi_"cl"$
#let pipcl = $pi_"pcl"$
#let pilqr = $pi_"lqr"$
#let pilspi = $pi_"lspi"$
#let pilspepi = $pi_"lspepi"$
#let Klqr = $K_"lqr"$

#title-slide(logo: none)

== Outline <touying:hidden>
#components.adaptive-columns(outline(title: none, indent: 1em))

////////////////////////////////////////////////////////////
// Main deck (clear + structured, ~12 min)
////////////////////////////////////////////////////////////

= Problem setup

== What are we solving?
#slide[
  #title[Goal]

  #blue[Objective] Maximize infinite-horizon average utility from trading one asset.

  #v(0.3em)
  #emph[State] $x_t = (q_t, z_t^a, z_t^u)^top$ \
  #muted[
    $q_t$: inventory (fraction of max position) \
    $z_t^a$: latent market factor (random, persistent) \
    $z_t^u$: temporary impact state (driven by our trades)
  ]

  #v(0.3em)
  #emph[Control] $u_t$ (trade amount) \
  with constraint (feasible trading): #red[$q_t in [0,1]$] enforced via clipping/MPC.

  #v(0.3em)
  #blue[Methods we compare]
  - Baseline linear feedback (#picl) + clipping
  - Direct policy search (CMA-ES)
  - Model-based control (LQR, MPC)
  - Data-driven RL/ADP (LSTD→LSPI, Poisson/LSPE(+PI), Q-$lambda$)
  #speaker-note[
    - 1 slide to frame: state, action, constraint, and what we tried.
    - emphasize: average-reward setting, not discounted.
  ]
]

== True dynamics and reward (intuitive)
#slide(composer: (1fr, 1fr))[
  #block[
    #blue[Dynamics (true system)]
    $
    q_(t+1) &= q_t + u_t \
    z_(t+1)^a &= (1 - omega^a) z_t^a + omega^a sigma^a xi_t^a \
    z_(t+1)^u &= (1 - omega^u) z_t^u + omega^u beta^u u_t
    $
    with $xi_t^a ~ cal(N)(0,1)$.
  ]

  #v(0.4em)
  #muted[
    Interpretation:
    - $z^a$: exogenous, mean-reverting “alpha” signal
    - $z^u$: our own transient price impact
  ]
][
  #block[
    #blue[Price and P&L]
    $p_(t+1) = p_t + z_t^a + z_t^u + gamma^u u_t + sigma^p xi_t^p$
    with $xi_t^p ~ cal(N)(0,1)$.
  ]

  #v(0.3em)
  #blue[Gross P&L]
  $
  g_t := 1000 dot ((q_(t+1)p_(t+1) - q_t p_t) - (q_(t+1) - q_t)\, overline(p)_(t,t+1))
  $

  #v(0.2em)
  #blue[Utility] $c(g) = 1 - exp(-g)$ (risk-averse).

  #v(0.3em)
  #muted[
    Key idea: trade-off between exploiting $z^a$ and controlling impact ($z^u$, $gamma^u$).
  ]
  #speaker-note[
    - explain P&L decomposition in one sentence: mark-to-market minus execution cost.
    - mention risk aversion: utility reduces incentive for high-variance strategies.
  ]
]

= Baselines and direct search

== Baseline linear controller (and why clipping matters)
#slide[
  #title[Baseline: linear feedback]

  #blue[Unconstrained policy]
  $
  pi_"cl"(x_t) = K_"cl"x_t = -0.5 q_t + 0.5 z_t^a + 0.5 z_t^u
  $

  #v(0.2em)
  #green[Works] Long-run average reward is positive (profitable on average).

  #v(0.2em)
  #red[Issue] Allows negative inventory (shorting) → unrealistic for our setting.

  #v(0.3em)
  #blue[Feasible version via clipping]
  $
    pi_"pcl"(x_t) = op("clip")(pi_"cl"(x_t), [-q_t, 1 - q_t])
  $
  #muted[
    Clipping creates “saturation episodes”: actions hit bounds, limiting exploitation.
  ]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/unclipped_policy_ex_trajectory_states_actions.svg", width: 100%),
      caption: [Example trajectory (#picl)]
    ),
    figure(
      image("figures/clipped_policy_ex_trajectory_average_reward.svg", width: 100%),
      caption: [Reward drops when clipping saturates]
    ),
  )
  #speaker-note[
    - point at saturation: plateaux in q_t/u_t and corresponding reward drops.
    - message: feasibility costs performance if policy not designed for constraints.
  ]
]

== Direct policy search (CMA-ES): when model is hard
#slide[
  #title[Direct policy search (CMA-ES)]

  #blue[Motivation] Reward is non-linear (utility) + stochastic → gradient-free search works well.

  #v(0.2em)
  #blue[Parameterized feasible policies]
  - Linear clipped: $pi_l(x)=op("clip")(p_3 [q, z^a, z^u]^top, [-q,1-q])$
  - Quadratic clipped: $pi_q(x)=op("clip")(p_10 phi(x), [-q,1-q])$

  #v(0.3em)
  #green[Result]
  - Improves reward distributions vs baselines.
  - Quadratic is only marginally better → diminishing returns vs complexity.

  #figure(
    image("figures/policy_comparison_histogram_first_three.svg", width: 85%),
    caption: [Reward comparison (baseline vs optimized)]
  )
  #speaker-note[
    - emphasize design choice: CMA-ES because objective is noisy.
    - highlight: improved mean but increased variance can happen (risk vs return).
  ]
]

= Model-based control

== LQR approximation: from non-linear utility to tractable control
#slide[
  #title[LQR approximation (model-based benchmark)]

  #blue[Approximate dynamics]
  $x_(t+1) = F x_t + G u_t + D xi_t$

  $F = mat(1,0,0; 0,1-omega^a,0; 0,0,1-omega^u)$,
  $quad G = mat(1; 0; omega^u beta^u)$.

  #v(0.3em)
  #blue[Approximate reward]
  $ EE[c_"quad"(g_t) | x_t, u_t] approx 1/2 x^top S x + x^top P u + 1/2 u^top R u $

  #v(0.2em)
  #muted[
    Design choice: keep only up to 2nd order terms ⇒ standard infinite-horizon LQR.
  ]

  #v(0.3em)
  #green[Why this is useful]
  - Gives an interpretable benchmark policy.
  - Enables analytical comparisons (Riccati, theoretical average reward).
]

== LQR policy and performance
#slide[
  #title[LQR policy #pilqr]

  $pi_"lqr"(x_t) = -K_"lqr"x_t$
  with $K_"lqr" approx mat(1.112, -2.649, -2.528)$.

  #v(0.2em)
  #green[Observations]
  - More aggressive trades (larger $u_t$ and inventory swings).
  - Achieves much higher average reward than the baseline.
  - Empirical reward close to theoretical prediction (from Riccati solution).

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/lqr_optimal_policy_ex_trajectory_states_actions.svg", width: 100%),
      caption: [States & action]
    ),
    figure(
      image("figures/lqr_optimal_policy_ex_trajectory_average_reward.svg", width: 100%),
      caption: [Average reward]
    ),
  )
  #speaker-note[
    - message: LQR captures “right direction” even though true utility is non-quadratic.
    - if asked: theory value computed via trace formula (appendix).
  ]
]

== Enforcing feasibility: clipped LQR vs MPC
#slide[
  #title[Feasible control: clipped LQR vs MPC]

  #blue[Clipped LQR]
  $pi_"plqr"(x)=op("clip")(pi_"lqr"(x),[-q,1-q])$ \
  #red[Issue:] saturation hurts reward when “best” action is infeasible.

  #v(0.3em)
  #blue[MPC]
  - Solve constrained optimization each step (horizon $cal(N)$).
  - We used #emph[$cal(N)=10$] as compute/reward compromise.
  - In experiments: similar reward to clipped LQR, higher compute cost.

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/lqr_clip_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
      caption: [Clipped LQR distribution]
    ),
    figure(
      image("figures/mpc_optimal_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
      caption: [MPC distribution]
    ),
  )
  #speaker-note[
    - interpret: feasibility is the main performance bottleneck.
    - MPC didn’t outperform enough to justify per-step optimization cost here.
  ]
]

= RL / ADP (data-driven)

== LSTD → LSPI: learning from trajectories
#slide[
  #title[LSTD and LSPI (value-based policy improvement)]

  #blue[Idea]
  Approximate $Q$ with linear features: $Q^theta(x,u) approx theta^top psi(x,u)$.

  #v(0.2em)
  #blue[Design choice: basis]
  - Quadratic basis matches LQR structure (exact in LQR case).
  - Higher degree basis needed for original non-quadratic utility.

  #v(0.3em)
  #green[Empirical result]
  LSPI improves over #picl and can approach model-based performance.

  #figure(
    image("figures/q63_policy_rewards.png", width: 75%),
    caption: [Reward comparison: #picl vs #pilqr vs #pilspi]
  )
  #speaker-note[
    - mention exploration policy: Gaussian around K_cl x.
    - point: good features are more important than fancy optimizer.
  ]
]

== Poisson error & LSPE(+PI): average-reward-consistent learning
#slide[
  #title[Poisson error and LSPE(+PI)]

  #blue[Average-reward setting]
  $cal(P)^theta(x,u) = EE[r + Q^theta(x^+, pi(x^+)) | x,u] - eta^theta - Q^theta(x,u)$

  #v(0.2em)
  #blue[LSPE]
  Minimize mean-squared Poisson error → least-squares in $(theta, eta)$.

  #v(0.2em)
  #blue[Kernel approximation]
  Expected next-feature uses finite-support (Monte-Carlo) approximation of the transition kernel.

  #v(0.3em)
  #green[With policy improvement (LSPE+PI)]
  - start from #picl,
  - alternate evaluation + improvement,
  - on LQR approximate model: quickly recovers LQR-like gain.

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("../../fig_in_git/convergence_during_LSPE+PI_on_Approx_Model.svg", width: 100%),
      caption: [Convergence to #Klqr]
    ),
    figure(
      image("../../fig_in_git/policy_comparison_histogram_Q8_7.svg", width: 100%),
      caption: [Rewards: #picl vs #pilqr vs #pilspepi]
    ),
  )
]

= Takeaways

== Final takeaways (what we learned)
#slide[
  #title[Takeaways]

  #blue[Benchmarks]
  - #pilqr is the best unconstrained controller (high reward, interpretable).
  - Feasibility (#red[$q in [0,1]$]) is the main source of performance loss.

  #v(0.3em)
  #blue[Feasible controllers]
  - clipped LQR ≈ MPC (here), but MPC costs more compute without clear gains.

  #v(0.3em)
  #blue[Learning-based methods]
  - With the right feature structure, LSPI and LSPE+PI can recover near-LQR behavior.
  - Main difficulty: constraint handling + saturation (needs constraint-aware improvement).

  #v(0.5em)
  #muted[Appendix contains: derivations, algorithms, hyperparameters, extra plots, and “bonus” parts.]
]

== Q&A
#slide[
  #title[Questions]

  #muted[
    Backup slides follow:
    - derivations (quadratic form, LQR reward, Riccati)
    - CMA-ES parameterization
    - E-PIA, Q-$lambda$ learning details
    - Poisson vs Bellman, LSPE normal equations
    - constrained PI discussion
  ]
]

////////////////////////////////////////////////////////////
// Appendix (bonus information)
////////////////////////////////////////////////////////////
#show: appendix
= Appendix

== Appendix: quadratic form of $g_t$
#slide[
  #title[Quadratic form of $g_t$]

  #muted[
    Use if asked: “How do you write $g_t$ as a quadratic form?”
  ]

  $g_t = 1/2 y_t^top H y_t$ with $y_t = (q_t, z_t^a, z_t^u, u_t, xi_t^a, xi_t^p)^top$.

  #v(0.3em)
  #muted[
    (Full $H$ matrix is in the report, Q2.3.)
  ]
]

== Appendix: LQR solution details
#slide[
  #title[LQR: Riccati + theoretical reward]

  #blue[Optimal gain]
  $K = (hat(R) + G^top M^* G)^(-1) (G^top M^* F + N^top)$

  #v(0.2em)
  #blue[Riccati equation]
  $
  M^* = F^top M^* F - (F^top M^* G + N)(hat(R) + G^top M^* G)^(-1)(G^top M^* F + N^top) + Q
  $

  #v(0.3em)
  #blue[Theoretical average reward]
  $J^*_("reward") = -1/2 tr(M^* D D^top)$

  #v(0.3em)
  #muted[
    We solved by iterating the finite-horizon Riccati recursion until convergence.
  ]
]

== Appendix: MPC design choice
#slide[
  #title[MPC: horizon choice $cal(N)$]

  #blue[Trade-off]
  - larger $cal(N)$: better look-ahead, more compute per step
  - smaller $cal(N)$: cheaper, more myopic

  #v(0.3em)
  #muted[
    We tested $cal(N)$ from 1 to 15 and selected $cal(N)=10$ as a good compute/reward compromise.
  ]

  #v(0.3em)
  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/mpc_optimal_policy_ex_trajectory_states_actions.svg", width: 100%),
      caption: [Example MPC trajectory]
    ),
    figure(
      image("figures/mpc_optimal_policy_ex_trajectory_average_reward.svg", width: 100%),
      caption: [Example MPC average reward]
    ),
  )
]

== Appendix: E-PIA (policy iteration) convergence
#slide[
  #title[E-PIA (policy iteration) → LQR]

  #muted[
    E-PIA updates $K_k$ using Lyapunov/Riccati-like evaluation and a greedy improvement step.
  ]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/EPIA_convergence.svg", width: 100%),
      caption: [Convergence to $K_"lqr"$]
    ),
    figure(
      image("figures/EPIA_K_values.svg", width: 100%),
      caption: [$K_k$ components vs iteration]
    ),
  )
]

== Appendix: LSTD basis choices (why quadratic often wins)
#slide[
  #title[LSTD basis choices]

  #blue[Quadratic (LQR-consistent)]
  $
  psi(x,u) = (&q^2, 2q z^a, 2q z^u, 2q u,
             (z^a)^2, 2z^a z^u, 2z^a u,
             (z^u)^2, 2z^u u, u^2)
  $

  #v(0.3em)
  #blue[Higher-degree (true utility)]
  #muted[
    For $c(g)=1-exp(-g)$, a degree-2 basis may underfit; we used higher degree in Part III.
  ]
]

== Appendix: Q-$lambda$ learning (stability)
#slide[
  #title[Q-$lambda$ learning: speed vs stability]

  #muted[
    Larger $lambda$ can speed convergence (credit assignment over multiple steps),
    but may cause instability if the learning rate $alpha$ is not reduced.
  ]

  #figure(
    image("figures/convergence_during_Q-λ_Learning_for_different_λ.svg", width: 75%),
    caption: [Effect of $lambda$ on convergence / instability]
  )
]

== Appendix: Poisson vs Bellman error (when to use which)
#slide[
  #title[Poisson error vs Bellman error]

  #blue[Bellman error] (discounted)
  $
    cal(B)^theta(x,u) = EE[r + gamma Q^theta(x^+, pi(x^+)) | x,u] - Q^theta(x,u)
  $
  with $gamma in [0,1)$.

  #v(0.3em)
  #blue[Poisson error] (average reward, $gamma=1$)
  $
    cal(P)^theta(x,u) = EE[r + Q^theta(x^+, pi(x^+)) | x,u] - eta^theta - Q^theta(x,u)
  $

  #v(0.3em)
  #muted[
    In our project we target average reward ⇒ Poisson/LSPE is the natural framework.
  ]
]

== Appendix: LSPE normal equations (bonus)
#slide[
  #title[LSPE: least-squares system]

  #muted[
    If $Q^theta(x,u)=theta_1^top psi_Q(x,u)$ and $eta^theta=theta_2$,
    define $Psi(x,u) = vec(overline(psi)(x,u)-psi_Q(x,u), -1)$ and
    $cal(P)^theta(x,u)=overline(r)(x,u) + theta^top Psi(x,u)$.
  ]

  #v(0.2em)
  #blue[Normal equations]
  $
  A_"lspe" theta = b_"lspe"
  $
  where
  $
  A_"lspe" = EE[Psi Psi^top], quad
  b_"lspe" = -EE[Psi\, overline(r)].
  $

  #v(0.2em)
  #muted[
    We approximate expectations using steady-state samples under an exploration policy.
  ]
]

== Appendix: finite-support kernel approximation (bonus)
#slide[
  #title[Finite-support transition kernel]

  #blue[Monte-Carlo kernel]
  $
    kappa_M(x^+ | x,u) = 1/M sum_(m=1)^M delta_(x^((m)+)(x,u))(x^+)
  $

  #v(0.3em)
  #muted[
    Sample $xi^(a,(m)) ~ cal(N)(0,1)$, build next states $x^((m)+)$, then use empirical expectation
    to compute $overline(psi)(x,u)$.
  ]
]

== Appendix: constrained policy improvement (why it can hurt)
#slide[
  #title[Constrained PI: why performance drops]

  #blue[Constraint]
  $0 <= q + u <= 1$

  #v(0.3em)
  #muted[
    Constraint-aware improvement often:
    - changes argmax of $Q^theta(x,u)$ (clipped optimizer hits boundaries),
    - increases saturation frequency,
    - reduces ability to exploit “strong signal” states.
  ]

  #v(0.3em)
  #figure(
    image("../../fig_in_git/policy_comparison_histogram_Q8_9_constrained.svg", width: 70%),
    caption: [Reward distributions under constraints]
  )
]

== Appendix: baseline distributions (for quick reference)
#slide[
  #title[Baseline reward distributions]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/unclipped_policy_final_reward_distribution_1000_trajectories.svg", width: 100%),
      caption: [Unclipped baseline (#picl)]
    ),
    figure(
      image("figures/clipped_policy_final_reward_distribution_1000_trajectories.svg", width: 100%),
      caption: [Clipped baseline (#pipcl)]
    ),
  )
]
