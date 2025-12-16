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

#title-slide(
  logo: none,
)

== Outline <touying:hidden>
#components.adaptive-columns(outline(title: none, indent: 1em))

////////////////////////////////////////////////////////////
// 12-min deck (12–14 slides)
////////////////////////////////////////////////////////////

= Problem & model


== Dynamics (true system)
#slide(composer: (1fr, 1fr))[

  #block[
    #emph[State:] $x_t = (q_t, z_t^a, z_t^u)^top$  \
     #emph[Control:] $u_t$ \
    #emph[Noise:] $xi_t^a, xi_t^p ~ cal(N)(0, 1)$
  ]

  #v(0.3em)
  #blue[Inventory & latent factors]
  $
  q_(t+1) &= q_t + u_t \
  z_(t+1)^a &= (1 - omega^a) z_t^a + omega^a sigma^a xi_t^a \
  z_(t+1)^u &= (1 - omega^u) z_t^u + omega^u beta^u u_t
  $


][

  #blue[Reward]

  $
  g_t := 1000 dot ((q_(t+1) p_(t+1) - q_t p_t) - (q_(t+1) - q_t) overline(p)_(t, t+1)) \
  c(g) = 1 - exp(-g)
  $
  with 
  $p_(t+1) = p_t + z_t^a + z_t^u + gamma^u u_t + sigma^p xi_t^p
  $

  - #blue[Mark-to-market term:] change in value of holdings
  - #blue[Execution term:] cash flow at $overline(p)_(t, t+1)$


]

== LQR approximation
#slide[
  #title[LQR approximation]

  #blue[Linear dynamics]
  $x_(t+1) = F x_t + G u_t + D xi_t$

  $F = mat(1,0,0; 0,1-omega^a,0; 0,0,1-omega^u) quad quad G = mat(1; 0; omega^u beta^u)$.

  #v(0.4em)
  #blue[Quadratic reward approximation]
  $ EE[c_"quad"(g_t) | x_t, u_t] approx 1/2 x^top S x + x^top P u + 1/2 u^top R u $

  #v(0.2em)
  #muted[→ Convert to cost minimization ⇒ standard infinite-horizon LQR.]
]

= Models

== Baseline linear policy
#slide[

  - $pi_"cl"(x_t) = K_"cl" x_t = -0.5 q_t + 0.5 z_t^a + 0.5 z_t^u$

    - long-run average reward > 0 → #emph[profitable but unrealistic]
  - $pi_"pcl"(x_t) = op("clip")(pi_"cl"(x_t), [-q_t, 1 - q_t])$ (Enforce #red[$q_t in [0,1]$])
    - #red[Reward drops] during saturation episodes.

  // #v(0.4em)


  #figure(
    image("figures/unclipped_policy_ex_trajectory_states_actions.svg", width: 95%),
    caption: [Example trajectory under #picl]
  )


  #figure(
    image("figures/clipped_policy_ex_trajectory_average_reward.svg", width: 85%),
    caption: [Average reward drops when clipping is active]
  )
  #speaker-note[
    - states/actions centered around 0
    - inventory goes negative (shorting)
  ]
]





== Direct policy search (CMA-ES)


#slide[
  #title[CMA-ES policy search]

  - #blue[Linear clipped form] $pi_l(x) = op("clip")(p_3 [q_t, z_t^a, z_t^u]^top, [-q_t, 1 - q_t])$

  #v(0.2em)
  - #blue[Quadratic clipped form] $pi_q(x) = op("clip")(p_10 phi(x_t), [-q_t, 1 - q_t])$

  #v(0.5em)
  #green[Empirical takeaway]
  - CMA-ES improves reward distributions vs baselines.
  - Quadratic slightly better than linear → #emph[diminishing returns].

  #figure(
    image("figures/policy_comparison_histogram_first_three.svg", width: 85%),
    caption: [Reward comparison: baseline vs optimized]
  )
]

// = Model-based control (LQR / MPC)


== LQR optimal policy
#slide[
  #title[LQR policy #pilqr]
  - $pi_"lqr"(x_t) = -K_"lqr" x_t quad quad quad$ with $K_"lqr" approx mat(1.112, -2.649, -2.528)$


  #v(0.4em)
  #green[Observations]
  - More aggressive actions and larger inventory swings.
  - #emph[Much higher average reward], close to theoretical prediction.

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

== Constraints: clipped LQR vs MPC
#slide[
  #title[Feasible control: clipped LQR vs MPC]

  - #blue[Clipped LQR] (feasible but saturates) → #red[reward decreases].
  - #blue[MPC] solves a constrained optimization each step:
    - horizon $cal(N)$ (we used #emph[$cal(N)=10$]),
    - similar performance to clipped LQR in our experiments,
    - #red[higher computation cost].

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



== LSTD → LSPI
#slide[
  #title[LSTD and LSPI]

  - Approximate $Q$ by $Q^theta(x,u) approx theta^top psi(x,u)$.
  - Basis choice:
    - #green[quadratic basis] matches LQR structure exactly,
    - higher-degree basis needed for non-quadratic utilities.

  #v(0.4em)
  #green[Result]
  - LSPI improves over #picl and can approach model-based performance.

  #figure(
    image("figures/q63_policy_rewards.png", width: 75%),
    caption: [Reward comparison: #picl vs #pilqr vs #pilspi]
  )
]

== Q-$lambda$ learning
#slide[
  #title[Q-$lambda$ learning]

  - Learned gains move toward #Klqr (on approximate system).
  - Larger $lambda$ speeds learning but may become #red[unstable] unless step size is reduced.

  #v(0.4em)
  #muted[Plots intentionally removed here; uncomment if you want them back.]
]

== Poisson error & LSPE(+PI)
#slide[
  #title[Poisson error and LSPE(+PI)]

  #blue[Average-reward setting uses Poisson error:]
  $cal(P)^theta(x,u) = EE[r + Q^theta(x^+, pi(x^+)) | x,u] - eta^theta - Q^theta(x,u)$

  - Minimizing MSPE → least squares in $(theta, eta)$.
  - Finite-support kernel (Monte Carlo) approximates expectations.

  #v(0.3em)
  #green[With policy improvement (LSPE+PI)]
  - start from #picl,
  - alternate evaluation and improvement,
  - can recover LQR-like behavior.

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

= Conclusion

== Summary
#slide[
  #title[Summary]

  - Best #emph[unconstrained] controller: #pilqr \
    #muted[(highest reward, but allows $q_t < 0$).]

  - Best #emph[feasible] controllers:
    - clipped LQR ≈ MPC (in our experiments),
    - MPC costs more compute without clear gains.

  - RL/ADP:
    - with the right basis, LSPI and LSPE+PI recover near-LQR performance quickly,
    - #red[constraint handling] is the main difficulty and source of performance loss.
]

== Design choices & difficulties
#slide[
  #title[Design choices & difficulties]

  #blue[Design choices]
  - action clipping to enforce #red[$q_t in [0,1]$],
  - CMA-ES for noisy objective optimization,
  - quadratic basis for LQR-consistent learning,
  - MPC horizon #emph[$cal(N)=10$] as compute/reward compromise.

  #v(0.4em)
  #blue[Difficulties]
  - clipping saturation reduces reward during “good opportunities”,
  - Q-$lambda$ instability at high $lambda$ without tuning $alpha$,
  - average-reward evaluation requires estimating $eta$ → Poisson/LSPE is natural.
]

== Conclusion
#slide[
  #title[Conclusion]

  - #emph[LQR] gives a strong, interpretable benchmark and near-theoretical performance.
  - With constraints, clipped LQR and MPC provide feasible control (similar performance here).
  - RL methods can match model-based control when approximation architecture matches structure.

  #v(0.8em)
  #muted[Backup slides in appendix.]
]

////////////////////////////////////////////////////////////
// Appendix
////////////////////////////////////////////////////////////
#show: appendix
= Appendix

== Reward quadratic form
#slide[
  #title[Appendix: quadratic form]

  $g_t = 1/2 y_t^top H y_t$
  with $y_t = (q_t, z_t^a, z_t^u, u_t, xi_t^a, xi_t^p)^top$.

  #muted[Use this if asked: “Why does a quadratic approximation make sense?”]
]

== Baseline distributions
#slide[
  #title[Appendix: baseline distributions]

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
