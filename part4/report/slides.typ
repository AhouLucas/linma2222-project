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
// Main deck (covers the whole project)
////////////////////////////////////////////////////////////

= Problem & model

== Goal and setting (average reward)
#slide[
  #title[Goal]

  #blue[We solve an average-reward trading control problem:]
  $
    max_pi lim_(T -> oo) 1/T * EE[ sum_(t=0)^(T-1) r_t ]
  $

  #v(0.3em)
  #block[
    #emph[State:] $x_t = (q_t, z_t^a, z_t^u)^top$ \
    #emph[Control:] $u_t$ (buy/sell) \
    #emph[Noise:] $xi_t^a, xi_t^p ~ cal(N)(0,1)$ \
    #emph[Constraint (feasible trading):] #red[$q_t in [0,1]$]
  ]

  #v(0.4em)
  #muted[
    We compare: baseline feedback + clipping, CMA-ES search, LQR/MPC, and RL/ADP (LSTD/LSPI, Q-$lambda$, LSPE(+PI)).
  ]
]

== Dynamics & reward (financial meaning)
#slide(composer: (1fr, 1fr))[
  #block[
    #blue[True dynamics]
    $
    q_(t+1) &= q_t + u_t \
    z_(t+1)^a &= (1 - omega^a) z_t^a + omega^a sigma^a xi_t^a \
    z_(t+1)^u &= (1 - omega^u) z_t^u + omega^u beta^u u_t
    $
  ]

  #v(0.3em)
  #muted[
    - $z^a$: exogenous mean-reverting “alpha” signal \
    - $z^u$: transient price impact due to our own trades
  ]
][
  #blue[Price / execution and P&L]
  $
    p_(t+1) = p_t + z_t^a + z_t^u + gamma^u u_t + sigma^p xi_t^p
  $
  $
    overline(p)_(t,t+1) = theta p_t + (1-theta)p_(t+1) + theta gamma^u u_t
  $

  #v(0.3em)
  #blue[Gross stage reward]
  $
  g_t := 1000 dot ((q_(t+1)p_(t+1)-q_t p_t) - (q_(t+1)-q_t) overline(p)_(t,t+1))
  $

  #v(0.2em)
  #muted[
    - Mark-to-market term: change in value of holdings \
    - Execution term: cash flow at average execution price
  ]
]

== Utility function: why variance hurts (Q2.4)
#slide[
  #title[Utility penalizes variance]

  #blue[Net reward:] $r_t = c(g_t)$ with
  $
    c(g) = max(g - 1/2 g^2, 1 - exp(-g))
  $

  #v(0.3em)
  #green[Key message]
  Even if $EE[g_t]=0$, increasing $"Var"(g_t)$ decreases $EE[c(g_t)]$.
  #figure(
    image("figures/plot_q2.4.svg", width: 70%),
    caption: [$EE[c(g)]$ decreases as variance increases (empirical)]
  )
  #speaker-note[
    - risk aversion: big swings are penalized
    - motivates more “stable” strategies in later parts
  ]
]

= Closed-loop analysis (baselines + constraints)

== Baseline linear policy #picl
#slide[
  #title[Baseline feedback policy]

  $
    pi_"cl"(x_t) = K_"cl"x_t = -0.5 q_t + 0.5 z_t^a + 0.5 z_t^u
  $

  #v(0.2em)
  #green[Observation]
  - states/actions centered near 0 → frequent buy/sell
  - long-run average reward > 0 → “profitable” on average

  #v(0.2em)
  #red[Issue]
  - can make $q_t < 0$ (shorting) → infeasible in our setting

  #figure(
    image("figures/unclipped_policy_ex_trajectory_states_actions.svg", width: 95%),
    caption: [Example trajectory under #picl]
  )
]

== Enforcing feasibility: clipped baseline #pipcl
#slide[
  #title[Clipping: feasible but saturating]

  #blue[Projected policy]
  $
    pi_"pcl"(x_t) = max(-q_t, min(1-q_t, pi_"cl"(x_t)))
  $

  #v(0.2em)
  #muted[
    Clipping creates saturation episodes: action hits bounds ⇒ “missed opportunities”.
  ]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/clipped_policy_ex_trajectory_states_actions.svg", width: 100%),
      caption: [States & action (plateaux = saturation)]
    ),
    figure(
      image("figures/clipped_policy_ex_trajectory_average_reward.svg", width: 100%),
      caption: [Reward drops correlate with saturation]
    ),
  )
]

== Better feasible policy: direct search (CMA-ES)
#slide[
  #title[CMA-ES feasible policy search]

  #blue[Why CMA-ES?]
  - objective is noisy (Monte Carlo average reward)
  - non-smooth because of clipping

  #v(0.2em)
  #blue[Policies tested]
  - linear clipped: $pi_l(x)=op("clip")(p_3 [q, z^a, z^u]^top, [-q,1-q])$
  - quadratic clipped: $pi_q(x)=op("clip")(p_10 phi(x), [-q,1-q])$

  #v(0.3em)
  #green[Result]
  CMA-ES improves reward distribution vs baselines; quadratic only slightly better.

  #figure(
    image("figures/policy_comparison_histogram_first_three.svg", width: 85%),
    caption: [Reward comparison: baseline vs CMA-ES policies]
  )
]

= Model-based control (approximation + constraints)

== LQR approximation (why and how)
#slide[
  #title[LQR approximation]

  #blue[Linear dynamics]
  $x_(t+1) = F x_t + G u_t + D xi_t$

  #v(0.2em)
  #blue[Quadratic reward approximation]
  $
    EE[c_"quad"(g_t) | x_t, u_t]
    approx 1/2 x^top S x + x^top P u + 1/2 u^top R u
  $

  #v(0.3em)
  #muted[
    Design choice: keep only 2nd-order terms → tractable Riccati solution (benchmark controller).
  ]
]

== LQR optimal policy (unconstrained)
#slide[
  #title[LQR policy #pilqr]

  - $pi_"lqr"(x_t) = -K_"lqr" x_t$ with $K_"lqr" approx mat(1.112, -2.649, -2.528)$

  #v(0.3em)
  #green[Observation]
  - more aggressive but timed actions
  - much higher average reward; empirical close to theoretical

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
]

== Feasibility: clipped LQR vs MPC
#slide[
  #title[Feasible control: clipped LQR vs MPC]

  #blue[Clipped LQR]
  - feasible but saturates ⇒ average reward decreases vs unconstrained LQR

  #v(0.2em)
  #blue[MPC]
  - solve constrained finite-horizon optimization each step
  - chosen horizon: #emph[$cal(N)=10$] (good compute/reward compromise)

  #v(0.2em)
  #green[Empirical takeaway]
  MPC ≈ clipped LQR performance here, but MPC is more expensive.

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/lqr_clip_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
      caption: [Clipped LQR reward distribution]
    ),
    figure(
      image("figures/mpc_optimal_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
      caption: [MPC reward distribution]
    ),
  )
]

= RL / ADP (Parts III)

== LSTD → LSPI (value-based improvement)
#slide[
  #title[LSTD and LSPI]

  #blue[Policy evaluation]
  Fit $Q^theta(x,u) approx theta^top psi(x,u)$ using LSTD from exploration data.

  #v(0.2em)
  #blue[Policy improvement]
  Greedy step: $pi(x)=arg max_u Q^theta(x,u)$ ⇒ iterate (LSPI).

  #v(0.2em)
  #green[Result on the true reward model]
  LSPI improves over #picl and can be competitive.

  #figure(
    image("figures/q63_policy_rewards.png", width: 75%),
    caption: [Rewards: #picl vs #pilqr vs #pilspi]
  )
]

== Q-$lambda$ learning (Parts III)
#slide[
  #title[Q-$lambda$ learning]

  #blue[On LQR approximate model]
  - learned gains move toward #Klqr
  - $lambda$ increases speed but may cause instability unless step-size is reduced

  #v(0.2em)
  #figure(
    image("figures/convergence_during_Q-λ_Learning_for_different_λ.svg", width: 75%),
    caption: [Effect of $lambda$ on convergence / instability]
  )
]

== Q-$lambda$ on the true (deterministic) system
#slide[
  #title[Q-$lambda$ on true system: cautious policy]

  #muted[
    Deterministic true system stabilizes to $(0,0,0)$, so gains occur mainly early in the episode.
  ]

  #figure(
    image("figures/policy_comparison_histogram_Q-7-2.svg", width: 70%),
    caption: [Average reward comparison on deterministic true system]
  )
]

= Mean Poisson error (Part IV)

== Poisson error vs Bellman error (why LSPE)
#slide[
  #title[Poisson error & LSPE motivation]

  #blue[Average-reward setting → Poisson error]
  $cal(P)^theta(x,u) = EE[r + Q^theta(x^+,pi(x^+)) | x,u] - eta^theta - Q^theta(x,u)$

  #v(0.2em)
  #blue[Discounted setting → Bellman error]
  $cal(B)^theta(x,u) = EE[r + gamma Q^theta(x^+,pi(x^+)) | x,u] - Q^theta(x,u)$

  #v(0.3em)
  #green[Takeaway]
  Since our project is average-reward, #emph[Poisson error is the natural consistency condition].
]

== LSPE evaluation (Part IV: Q8.5–Q8.6)
#slide[
  #title[LSPE policy evaluation]

  #blue[Kernel approximation]
  Use a finite-support (Monte Carlo) approximation of the transition kernel.

  #v(0.3em)
  #grid(
    columns: (1fr, 1fr),
    figure(
      image("../../fig_in_git/comparison_Q8.5:_Q_mathrmLSPE_vs_hat_Q_mathrmMC_(Poisson).svg", width: 100%),
      caption: [True system: $Q_"LSPE"$ vs Monte-Carlo estimate]
    ),
    figure(
      image("../../fig_in_git/comparison_Q8.6:_Q_mathrmLSPE_(Approx_Model)_vs_Q_mathrmExact_(LQR_Model).svg", width: 100%),
      caption: [LQR model: $Q_"LSPE"$ vs exact $Q$]
    ),
  )
]

== LSPE+PI (Part IV: Q8.7–Q8.8) + constrained PI (Q8.9)
#slide[
  #title[LSPE+PI and constraint effects]

  #blue[LSPE+PI]
  - start from #picl
  - alternate evaluation (LSPE) + improvement
  - on LQR model: gain converges quickly to #Klqr

  #v(0.3em)
  #grid(
    columns: (1fr, 1fr),
    figure(
      image("../../fig_in_git/convergence_during_LSPE+PI_on_Approx_Model.svg", width: 100%),
      caption: [Gain convergence to #Klqr]
    ),
    figure(
      image("../../fig_in_git/policy_comparison_histogram_Q8_7.svg", width: 100%),
      caption: [Rewards: #picl vs #pilqr vs #pilspepi]
    ),
  )

  #v(0.2em)
  #muted[
    With constrained PI, saturation increases; unconstrained PI can look worse unless all policies are constrained for fair comparison.
  ]
]

= Conclusion

== Summary of methods and results
#slide[
  #title[Summary]

  #blue[Baselines]
  - #picl: positive reward but infeasible (can short)
  - #pipcl: feasible but reward drops during saturation

  #v(0.2em)
  #blue[Feasible improvement]
  - CMA-ES: boosts performance; quadratic gives small extra gain

  #v(0.2em)
  #blue[Model-based]
  - #pilqr: best unconstrained benchmark (high reward, interpretable)
  - clipped LQR ≈ MPC (here), MPC costs more compute

  #v(0.2em)
  #blue[Learning]
  - LSPI and LSPE+PI can recover near-LQR behavior with good features
  - main difficulty is constraint handling (clipping / constrained PI)
]

== Design choices & difficulties (explicit)
#slide[
  #title[Design choices & difficulties]

  #blue[Design choices]
  - enforce feasibility via clipping; compare with MPC
  - CMA-ES for noisy/non-smooth policy optimization
  - quadratic features for LQR-consistent learning; higher-degree for true utility
  - MPC horizon #emph[$cal(N)=10$] as compute/reward trade-off
  - exploration: Gaussian around $K_"cl"x$ with tuned variance

  #v(0.4em)
  #blue[Difficulties + how we addressed them]
  - saturation episodes reduce reward ⇒ compare “fairly” (all constrained vs all unconstrained)
  - Q-$lambda$ instability at large $lambda$ ⇒ reduce learning rate $alpha$
  - average-reward evaluation needs $eta$ ⇒ Poisson/LSPE directly estimates it
]

== Q&A
#slide[
  #title[Questions]

  #muted[
    Backup slides follow: derivations, algorithms, extra plots, and bonus details.
  ]
]

////////////////////////////////////////////////////////////
// Appendix (bonus + all extra details)
////////////////////////////////////////////////////////////
#show: appendix
= Appendix

== Appendix: $g_t$ as quadratic form (Q2.3)
#slide[
  #title[Quadratic form of $g_t$]

  $g_t = 1/2 y_t^top H y_t$ with
  $y_t = (q_t, z_t^a, z_t^u, u_t, xi_t^a, xi_t^p)^top$.

  #muted[
    Full matrix $H$ is given in the report (Q2.3). Use this slide if asked “how did you get the quadratic structure?”.
  ]
]

== Appendix: LQR Riccati and theoretical reward (Q4.4)
#slide[
  #title[LQR details]

  #blue[Optimal policy]
  $u_t = -K x_t$

  #v(0.2em)
  #blue[Riccati fixed point]
  $
  M^* = F^top M^* F - (F^top M^* G + N)(hat(R)+G^top M^* G)^(-1)(G^top M^* F + N^top) + Q
  $

  #v(0.2em)
  #blue[Theoretical average reward]
  $J^*_("reward") = -1/2 tr(M^* D D^top)$

  #muted[
    We compute $M^*$ by iterating finite-horizon Riccati until convergence.
  ]
]

== Appendix: Baseline distributions (Q3.2 / Q3.4)
#slide[
  #title[Baseline distributions]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/unclipped_policy_final_reward_distribution_1000_trajectories.svg", width: 100%),
      caption: [Unclipped baseline]
    ),
    figure(
      image("figures/clipped_policy_final_reward_distribution_1000_trajectories.svg", width: 100%),
      caption: [Clipped baseline]
    ),
  )
]

== Appendix: LQR distributions (Q4.6 / Q4.8)
#slide[
  #title[LQR distributions]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/lqr_optimal_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
      caption: [Unclipped LQR]
    ),
    figure(
      image("figures/lqr_clip_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
      caption: [Clipped LQR]
    ),
  )
]

== Appendix: MPC trajectories (Q4.10–Q4.11)
#slide[
  #title[MPC extra plots]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/mpc_optimal_policy_ex_trajectory_states_actions.svg", width: 100%),
      caption: [Example MPC states/actions]
    ),
    // figure(
    //   image("figures/mpc_optimal_policy_cumulative_reward_100_trajectories.svg", width: 100%),
    //   caption: [MPC average reward over 100 sims]
    // ),
  )
]

== Appendix: Exact Policy Iteration (E-PIA) (Q5.1)
#slide[
  #title[E-PIA convergence]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/EPIA_convergence.svg", width: 100%),
      caption: [Error norm to #Klqr]
    ),
    figure(
      image("figures/EPIA_K_values.svg", width: 100%),
      caption: [$K_k$ components]
    ),
  )
]

== Appendix: LSTD evaluation plots (Q6.3–Q6.4)
#slide[
  #title[LSTD evaluation]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/q63_lspd_vs_qhat.png", width: 100%),
      caption: [True system: $Q^theta$ vs $\hat Q$]
    ),
    // figure(
    //   image("figures/q64_lspd_vs_qhat.png", width: 100%),
    //   caption: [LQR model: $Q^theta$ vs exact/empirical $Q$]
    // ),
  )
]

== Appendix: LSPI convergence and constrained PI (Q6.6–Q6.7)
#slide[
  #title[LSPI: convergence + constraints]

  #grid(
    columns: (1fr, 1fr),
    // figure(
    //   image("figures/q64_policy_convergence.png", width: 100%),
    //   caption: [LSPI gain converges to #Klqr]
    // ),
    figure(
      image("figures/q67_policy_rewards.png", width: 100%),
      caption: [Constrained PI can reduce reward]
    ),
  )
]

== Appendix: LSPE constrained comparison (Q8.9)
#slide[
  #title[LSPE+PI with constrained improvement]

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("../../fig_in_git/policy_comparison_histogram_Q8_9.svg", width: 100%),
      caption: [Unconstrained comparison]
    ),
    figure(
      image("../../fig_in_git/policy_comparison_histogram_Q8_9_constrained.svg", width: 100%),
      caption: [Fair comparison: all constrained]
    ),
  )
]
