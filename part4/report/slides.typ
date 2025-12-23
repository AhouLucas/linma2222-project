#import "@preview/touying:0.6.1": *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/theorion:0.3.2": *
#import cosmos.clouds: *

#show: show-theorion

#import themes.metropolis: *
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-common(frozen-counters: (theorem-counter,), show-notes-on-second-screen: right),
  config-info(
    title: [LINMA2222 - Stochastic Optimal Control & RL],
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

== Dynamics & reward
#slide(composer: (1fr, 1fr))[


  #blue[Average-reward trading control:]
  $
    max_pi lim_(T -> oo) 1/T * EE[ sum_(t=0)^(T-1) r_t ]
  $

  #block[
    #emph[State:] $x_t = (q_t, z_t^a, z_t^u)^top$ \
    #emph[Control:] $u_t$\
    #emph[Noise:] $xi_t^a, xi_t^p ~ cal(N)(0,1)$ \
    #emph[Constraint:] #red[$q_t in [0,1]$]
  ]
  #block[
    #blue[True dynamics]
    $
    q_(t+1) &= q_t + u_t \
    z_(t+1)^a &= (1 - omega^a) z_t^a + omega^a sigma^a xi_t^a \
    z_(t+1)^u &= (1 - omega^u) z_t^u + omega^u beta^u u_t
    $
  ]

  #v(0.3em)
  // #muted[
  //   - $z^a$: exogenous mean-reverting “alpha” signal \
  //   - $z^u$: transient price impact due to our own trades
  // ]
][
  #blue[Price / execution and P&L]
  $
    p_(t+1) = p_t + z_t^a + z_t^u + gamma^u u_t + sigma^p xi_t^p
  $
  $
    overline(p)_(t,t+1) = theta p_t + (1-theta)p_(t+1) + theta gamma^u u_t
  $

  #v(0.3em)
  #blue[Reward]
  $
  g_t &:= 1000 dot (underbrace((q_(t+1)p_(t+1)-q_t p_t)) - underbrace((q_(t+1)-q_t) overline(p)_(t,t+1)))\

  r_t &:= c(g_t) = max(g_t - 1/2 g_t^2, 1 - exp(-g_t))
  $

  #v(0.2em)
  
  #speaker-note[
    - model : buy/sell shares to maximize profit
  ]
]

// == Utility penalizes variance
// #slide(composer: (1fr, 1fr))[

//   #blue[Net reward:] $r_t = c(g_t)$ with
//   $
//     c(g) = max(g - 1/2 g^2, 1 - exp(-g))
//   $

//   #v(0.3em)

//   Increasing $"Var"(g_t)$ decreases $EE[c(g_t)]$.
  
// ][
// #figure(
//     image("figures/plot_q2.4.svg", width: 100%),
//     caption: [$EE[c(g)]$ decreases as variance increases (empirical)]
//   )
// ]



== Baseline linear policy #picl and #pipcl
#slide[
  #blue[baseline policy]
  $
    pi_"cl"(x_t) &= K_"cl"x_t = -0.5 q_t + 0.5 z_t^a + 0.5 z_t^u\
    pi_"pcl"(x_t) &= max(-q_t, min(1-q_t, pi_"cl"(x_t)))
  $

  #v(1em)

  average rewards :
  - $picl$ : 0.011451 
  - $pipcl$ : 0.005796

  #speaker-note[
  - can make $q_t < 0$ 
    
    => infeasible
  ]
][
  #figure(
    image("figures/unclipped_policy_ex_trajectory_states_actions.svg", width: 95%),
    caption: [Example trajectory under #picl]
  )
]

#slide[

  #v(0.2em)

  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/unclipped_policy_ex_trajectory_average_reward.svg", width: 100%),
      caption: [Rewards of #picl]
    ),
    figure(
      image("figures/clipped_policy_ex_trajectory_average_reward.svg", width: 100%),
      caption: [Rewards of #pipcl]
    ),
  )
  #speaker-note[
    no good use of this
  ]
]

== Better policy via CMA-ES
#slide(composer: (1fr, 1fr))[

  #v(0.2em)
  method made for noisy optimization.

  #blue[CMA-ES linear policy]
  $ pi_l(x)=op("clip")(p_3 [q, z^a, z^u]^top, [-q,1-q]) $
  #blue[CMA-ES quadratic policy]
  $ pi_q(x)=op("clip")(p_10 phi(x), [-q,1-q]) $

  #v(0.3em)

  #speaker-note[
  - quadratic only slightly better.
  ]

][

  #figure(
    image("figures/policy_comparison_histogram_first_three.svg", width: 100%),
    caption: [Reward comparison: baseline vs CMA-ES policies]
  )
]

= Model-based control

== LQR approximation
#slide[
  #title[LQR approximation]

  #blue[Linear dynamics]
  $x_(t+1) = F x_t + G u_t + D xi_t$

  #v(0.2em)
  #blue[Reward : ]
  minimise the expected cost:
  $
    hat(r) (x,u):= 1/2 x^top S x + x^top P u + 1/2 u^top R u
    \ max quad lim_ oo 1/T EE[ sum_(t=0)^(T-1) hat(r)(x_t,u_t)]
    

  $

  $
    EE[r(x,u)] approx EE[c_"quad" (g(x,u))] approx EE[hat(r)(x,u)]
  $
  with $c_"quad"(g) = g - 1/2 g^2$

  #v(0.3em)
  #speaker-note[
    Design choice: keep only 2nd-order terms → tractable Riccati solution (benchmark controller).
  ]
][
  #blue[Resulting matrices]
  $
    F = mat(1, 0, 0;
          0, 1-omega^a, 0;
          0, 0, 1-omega^u)
    quad
    \
    G = mat(1;
          0;
          omega^u beta^u)
  $

  #speaker-note[
    - linearize dynamics & quadratic reward
    - obtain LQR model (F,G,Q,R,S)
  ]
  
]

== LQR: Riccati equation & optimal gain
#slide[
  #title[LQR policy derivation]

  #blue[Discrete Algebraic Riccati Equation (DARE)]
  $
    M^* = Q + F^top M^* F - (F^top M^* G + S)(R + G^top M^* G)^(-1)(G^top M^* F + S^top)
  $

  #v(0.3em)
  #blue[Optimal gain]
  $
    K_"lqr" = -(R + G^top M^* G)^(-1)(G^top M^* F + S^top)approx
    //  mat(1.112, -2.649, -2.528) 
    // in bold
    bold(mat(1.112, -2.649, -2.528))
  $

  #v(0.3em)
  #blue[Theoretical average reward]
  $
    J^* = -1/2 tr(M^* D D^top) = #emph[0.01936]
  $
  #speaker-note[
    - Solve DARE iteratively or via scipy.linalg.solve_discrete_are
    - Closed-loop stability: eigenvalues of $(F + G K)$ inside unit circle
  ]
]

== LQR optimal policy
#slide[
  #blue[LQR policy]

  $ pi_"lqr"(x_t) = -K_"lqr" x_t quad $


  #v(0.3em)

  - average reward: #emph[0.019467]
  - average reward: #emph[0.009693] (constrained)
  - theoretical  :  #emph[0.01936]

  #speaker-note[

  #blue[Clipped LQR policy]
  - average reward decreases
  ]
][
  #grid(
    rows: (1fr, 1fr),
    figure(
      image("figures/lqr_optimal_policy_ex_trajectory_states_actions.svg", width: 100%),
      // caption: [States & action]
    ),
    figure(
      image("figures/lqr_optimal_policy_ex_trajectory_average_reward.svg", width: 100%),
      // caption: [Average reward]
    ),
  )


]

== MPC: Formulation
#slide[
  #title[Model Predictive Control]

  #blue[Variables:] $z_k in RR^(n_x)$ (predicted states), $v_k in RR^(n_u)$ (predicted controls)

  #v(0.2em)
  #blue[Optimization problem at time $t$:]
  $
    min_(z, v) quad sum_(k=0)^(cal(N)-1) underbrace(1/2 z_k^top Q z_k + z_k^top S v_k + 1/2 v_k^top R v_k, hat(r)(z_k, v_k))
  $
  #blue[Subject to:]
  $
    z_0 &= x_t \
    z_(k+1) &= F z_k + G v_k, quad &&k = 0, dots, cal(N)-1 \
    y_min &<= z_(k+1) <= y_max, quad &&k = 0, dots, cal(N) \
    u_min &<= v_k <= u_max, quad &&k = 0, dots, cal(N)-1
  $


  #speaker-note[
    - z, q: decision variables (predicted trajectory)
    - H, E: output matrix (here H selects position q_t)
    - y_min, y_max: output bounds (position in [0,1])
    - R_0: terminal cost (here R_0 = Q)
  ]
]

== MPC: QP Reformulation
#slide(composer: (1fr, 1fr))[
  #title[Stacked QP formulation]

  #blue[Stack variables:] 
  \ $w = [z_0, dots, z_cal(N), v_0, dots, v_(cal(N)-1)]^top$

  #v(0.2em)
  #blue[Solve :]
  $
    min_w quad 1/2 w^top H_"obj" w
  $
  $
    H_"obj" = mat(
      Q, , , S, , ;
      , dots.down, , , dots.down, ;
      , , Q, , , S;
      S^top, , , R, , ;
      , dots.down, , , dots.down, ;
      , , S^top, , , R
    )
  $

][
  #blue[Equality constraints] :
  $
    A_"eq" w = [x_t, 0, dots, 0]^top
  $

  #blue[bounds] :
  $
    y_min <= z_k <= y_max, quad u_min <= v_k <= u_max
  $

  #v(0.2em)

  #speaker-note[
    - Standard QP: efficient solvers available
    - Sparse structure exploited
    - Receding horizon: only apply $v_0^*$
  ]
]

== MPC: Condensed Formulation
#slide[
  #title[Condensed QP (eliminate states)]

  #blue[Lifted dynamics:] substitute $z_k = F^k x_0 + sum_(j=0)^(k-1) F^(k-1-j) G v_j$

  $
    Z = cal(F) x_0 + cal(G) V
  $
  where $Z = [z_0, dots, z_cal(N)]^top$, $V = [v_0, dots, v_(cal(N)-1)]^top$

  #v(0.2em)
  #blue[Reduced QP:] only optimize over $V in RR^(n_u dot cal(N))$
  $
    min_V quad 1/2 V^top hat(H) V + hat(F)^top V
  $
  with:
  $
    hat(H) &= 2(cal(G)^top tilde(Q) cal(G) + tilde(R) + cal(G)^top tilde(S) + tilde(S)^top cal(G)) \
    hat(F) &= 2(cal(G)^top tilde(Q) cal(F) + tilde(S)^top cal(F)) x_0
  $

  #blue[Constraints:] $quad l <= A_"lin" V <= u$ #h(0.5em) (input + output bounds)

  #speaker-note[
    - Eliminates state variables → smaller QP
    - $tilde(Q), tilde(R), tilde(S)$: block-diagonal cost matrices
    - Output constraints: $y_min <= H z_k + E v_k <= y_max$
  ]
]

== MPC: Results
#slide[


  #v(0.2em)
  #blue[MPC]
  - solve constrained finite-horizon optimization each step
  - chosen horizon: #emph[$cal(N)=10$] (good compute/reward compromise)

  #speaker-note[
    N=10 : good horizon
    - greater N suffer from diverging prediction (no noise model)
  ]
  


][
  #grid(
    rows: (1fr, 1fr),
    figure(
      image("figures/mpc_optimal_policy_ex_trajectory_average_reward.svg", width: 100%),
      caption: [Clipped LQR reward distribution]
    ),
    figure(
      image("figures/mpc_optimal_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
      caption: [MPC reward distribution]
    ),
  )
]

= Policy Improvement (Part III)

== Exact Policy Iteration Algorithm (E-PIA) (Q5)
#slide[
  #title[E-PIA on LQR model]

  #blue[Policy iteration steps]
  1. *Evaluation:* Solve Lyapunov equation for $P_k$:
  $ P_k = Q_k + A_k^top P_k A_k $
  where $A_k = F + G K_k$, $Q_k = Q + S K_k + (S K_k)^top + K_k^top R K_k$

  2. *Improvement:* Update gain:
  $ K_(k+1) = -(R + G^top P_k G)^(-1)(S^top + G^top P_k F) $

]

== E-PIA Results (Q5)
#slide(composer: (1fr, 1fr))[
  #title[E-PIA convergence]

  #green[Results]
  - From #picl, converges to #Klqr in $approx 5$ iterations
  - Confirms that Riccati-based LQR is optimal for the quadratic model

  #v(0.3em)
  #blue[Learned gain]
  $ Klqr approx mat(1.112, -2.649, -2.528) $
][
  // #grid(
  //   rows: (1fr, 1fr),
  //   figure(
  //     image("../../part3/figures/q64_policy_convergence.png", width: 80%),
  //     caption: [Gain components $K_k$ over iterations]
  //   ),
  //   figure(
  //     image("../../part3/figures/q63_policy_rewards.png", width: 80%),
  //     caption: [Cumulative reward comparison]
  //   ),
  // )
  #figure(
    image("figures/EPIA_convergence_during_E-PIA_Iteration.svg", width: 100%),
  )
]

== LSTD: Least-Squares Temporal Difference (Q6.3–Q6.4)
#slide[
  #title[LSTD for policy evaluation]

  #blue[Objective]
  Approximate the Q-function under a fixed policy $pi$:
  $ Q^pi (x,u) approx theta^top psi(x,u) $

  #v(0.2em)
  #blue[LSTD update]
  Solve for $theta$ using instrumental variables:
  $ theta = [1/N W + R_N]^(-1) phi.alt_N $
  where:
  $ R_N = 1/N sum_(k=1)^N Upsilon_k Upsilon_k^top quad quad phi.alt_N = 1/N sum_(k=1)^N Upsilon_(k+1) gamma_k $


  #speaker-note[
    + $Upsilon_k = psi(x(k), u(k)) - psi(x(k+1), pi(x(k+1)))$
    + $gamma_k = c(x(k), u(k))$
  ]
] 

== LSTD: Design Choices & Results
#slide(composer: (1fr, 1fr))[
  #title[LSTD for true model]

  #red[Key design choices]
  - *Degree-4 polynomial basis*: cost contains exponential term
  - *No bias term*: undiscounted setting, $Q(0,0)=0$
  - *Regularization*: $(R_N + epsilon I)$ for numerical stability
  - *Large $N$*: $N = 3000$ samples for well-conditioned matrix

  #v(0.3em)
  #green[Results]
  - Good match with Monte-Carlo estimates on true system
][
  #v(0.2em)

  #figure(
    image("../figures/q63_lspd_vs_qhat.png", width: 100%),
    caption: [True system: $Q^theta$ vs $hat(Q)_"MC"$]
  )
]


== LSTD: Design Choices & Results
#slide(composer: (1fr, 1fr))[
  #title[LSTD for LQR model]

  #red[Key design choices]
  - *Degree-2 polynomial basis*: $Q_"exact"$ is quadratic

  #v(0.3em)
  #green[Results]
  - Perfect match with the exact $Q$-function
][
  #v(0.2em)

  #figure(
  image("../figures/q64_lspd_vs_qhat.png", width: 100%),
  caption: [LQR model: $Q^theta$ vs $Q_"exact"$]
)
]



== LSPI: Least-Squares Policy Iteration (Q6.5–Q6.7)
#slide(composer: (1fr, 1fr))[
  #title[LSPI algorithm]

  #blue[Iterate:]
  1. *Evaluation:* LSTD → $theta_k$
  2. *Improvement:* Find greedy policy w.r.t. $Q^(theta_k)$

  #v(0.2em)
  #blue[Policy improvement step]
  For linear policy $u = -K x$:
  - Sample states $x_s in [-1,1]^3$
  - Optimize: $u^*_s = arg max_u theta^top psi(x_s, u)$
  - Fit: $K_(k+1) = (X^top X)^(-1) X^top (-U^*)$

][
  #v(0.2em)
  #figure(
    grid(
      rows: (auto, auto),
      image("../figures/q64_policy_convergence.png", width: 80%),
      image("../figures/q63_policy_rewards.png", width: 80%),
    )
  )
]

== Q-$lambda$ Learning (Q7)
#slide(composer: (1fr, 1fr))[
  #title[Q-$lambda$: online temporal-difference]

  #blue[Algorithm]
  - Quadratic features: $psi(x,u) = "vec"([x;u][x;u]^top)$
  - $zeta_(t+1) = lambda zeta_t + psi_t$
  - TD error: $cal(D)_t = c_t + Q_(t+1) - Q_t$
  - Update: $theta <- theta + alpha cal(D)_t zeta_t$

  #v(0.2em)
  #blue[Effect of $lambda$]
  - $lambda = 0$: one-step TD (slow but stable)
  - $lambda arrow 1$: Monte-Carlo-like (faster but can diverge)

][
  #figure(
    image("figures/convergence_during_Q-λ_Learning_for_different_λ.svg", width: 100%),
    caption: [Convergence to #Klqr for different $lambda$]
  )
]

== Q-$lambda$ Results
#slide(composer: (1fr, 1fr))[
  #title[Q-$lambda$ performance]

  #blue[Convergence]
  - Converges to #Klqr for LQR settings
  - Higher $lambda$ → faster convergence (with tuned $alpha$)

  #v(0.3em)
  #blue[Reward for deterministic system]
  - $pi_(Q(lambda))$ less risky\ $==>$ less profit in some cases
][
  #figure(
    grid(
      rows: (auto, auto),
      image("figures/q_lambda_true_system_x0_1_average_reward.svg"),
      image("figures/q_lambda_true_system_x0_2_average_reward.svg")
    )
  )
]

== Part III Summary
#slide[

  #table(
    columns: (1.2fr, 1.2fr, 0.8fr, 2.5fr),
    align: (left, center, center, left),
    [*Method*], [*Model needed*], [*Online*], [*Key insight*],
    [E-PIA], [Yes (full)], [No], [Exact convergence to #Klqr via Lyapunov + improvement],
    [LSTD], [No], [No], [Basis: degree-4 for true, degree-2 for LQR],
    [LSPI], [No], [No], [Iterate: LSTD evaluation + greedy improvement],
    [Q-$lambda$], [No], [Yes], [$lambda$ controls bias-variance: higher $lambda$ → faster but needs smaller $alpha$],
  )

  #green[Takeaways]
  - E-PIA: model-based benchmark, converges in ~6 iterations
  - LSTD/LSPI: model-free, basis choice critical (degree-4 for true reward)
  - Q-$lambda$: online learning, $pi_(Q_lambda)$ is more conservative (less risky, less profit)
]

= LSPE for Average-Reward (Part IV)

== Motivation: Poisson vs Bellman Error
#slide[
  #title[Why LSPE for average-reward?]

  #blue[Discounted setting → Bellman error]
  $ cal(B)^(theta)(x,u) = EE[r + gamma Q^(theta)(x',pi(x')) | x,u] - Q^(theta)(x,u) $

  #v(0.3em)
  #blue[Average-reward setting → Poisson error]
  $ cal(P)^(theta)(x,u) = EE[r + Q^(theta)(x',pi(x')) | x,u] - eta^theta - Q^(theta)(x,u) $

  #v(0.3em)
  #green[Key difference]
  - No discount factor $gamma$ → need to subtract average reward $eta$
  - LSPE minimizes mean squared Poisson error
  - Jointly estimates $theta$ (Q-parameters) and $eta$ (average reward)
]

== LSPE Algorithm (Q8.5–Q8.6)
#slide[
  #title[Least-Squares Policy Evaluation]

  #blue[Objective]
  Find $theta = [theta_Q; eta]$ that minimizes:
  $ min_theta EE[(cal(P)^(theta)(x,u))^2] $

  #v(0.2em)
  #blue[LSPE solution]
  Define $phi_k = [EE[psi'] - psi_k; -1]$, then solve:
  $ A theta = b quad "where" quad A = EE[phi phi^top], quad b = -EE[phi dot r] $

  #v(0.2em)
  #red[Implementation]
  - Monte-Carlo approximation of $EE[psi(x', pi(x'))]$
  - Regularization: $(A + lambda I)$ for stability
  - Quadratic basis $psi$: 14 features (no constant term)
]

== LSPE Results: Policy Evaluation
#slide(composer: (1fr, 1fr))[
  #title[Q8.5–Q8.6: Evaluating #picl]

  #blue[Q8.5: True system]
  - LSPE matches Monte-Carlo $hat(Q)$
  - Validates the Poisson formulation

  #v(0.3em)
  #blue[Q8.6: LQR approximate model]
  - LSPE matches exact $Q_"LQR"$
  - Confirms correctness of implementation
][
  #figure(
    grid(
      rows: (auto, auto),
      image("../../fig_in_git/comparison_Q8.5:_Q_mathrmLSPE_vs_hat_Q_mathrmMC_(Poisson).svg", width: 70%),
      image("../../fig_in_git/comparison_Q8.6:_Q_mathrmLSPE_(Approx_Model)_vs_Q_mathrmExact_(LQR_Model).svg", width: 70%),

    )
    
  )
]

== LSPE + Policy Iteration (Q8.7–Q8.8)
#slide[
  #title[LSPE+PI algorithm]

  #blue[Iterate:]
  1. *Data collection:* Explore with current policy + noise ($sigma_"exp" = 0.02$)
  2. *Evaluation:* LSPE → $theta_Q$, $eta$
  3. *Improvement:* Greedy policy from $Q^theta$

  #v(0.2em)
  #blue[Greedy policy extraction]
  For quadratic $Q(x,u) = dots + b(x) u + a u^2$:
  $ u^* = -b(x) / (2a) $

  #v(0.2em)
  #red[Two settings tested]
  - Q8.7: Train & evaluate on *true system*
  - Q8.8: Train on *LQR model*, evaluate on true system
]

== LSPE+PI Results (Q8.7–Q8.8)
#slide(composer: (1fr, 1fr))[
  #title[Convergence and performance]

  #green[Q8.8: LQR model]
  - Gain $K$ converges to #Klqr in _few_ iterations
  - Confirms LSPE+PI recovers optimal LQR policy

  #v(0.3em)
  #green[Q8.7: True system]
  - Similar performance as $K_"lqr"$
][
  #figure(
    grid(
      rows: (auto, auto),
      
      image("../../fig_in_git/convergence_during_LSPE+PI_on_Approx_Model.svg", width: 90%),      
      image("../../fig_in_git/policy_comparison_histogram_Q8_8.svg", width: 90%),

      ),
    )
]

== Constrained Policy Improvement (Q8.9)
#slide(composer: (1fr, 1fr))[
  #title[Enforcing $q in [0,1]$]

  #blue[Constrained improvement]
  $ u^* = arg max_(u in [-q, 1-q]) Q^(theta)(x, u) $

  #v(0.2em)
  #red[Effect of constraints]
  - Saturation episodes reduce reward
  - Unconstrained policies look better... but are infeasible!

  #v(0.2em)
  #green[Fair comparison]
  - Compare all policies under same constraint
][
  #figure(
    grid(
      rows: (auto, auto),
      image("../../fig_in_git/policy_comparison_histogram_Q8_9.svg", width: 90%),
      image("../../fig_in_git/policy_comparison_histogram_Q8_9_constrained.svg", width: 90%),
    ),
    caption: [Unconstrained (above) & Constrained (below)]
  )
]

== Part IV Summary
#slide[
  #table(
    columns: (1.5fr, 1.5fr, 2fr),
    align: (left, center, left),
    [*Question*], [*Setting*], [*Key result*],
    [Q8.5], [True system], [$Q_"LSPE" approx hat(Q)_"MC"$ — validates Poisson formulation],
    [Q8.6], [LQR model], [$Q_"LSPE" approx Q_"exact"$ — confirms implementation],
    [Q8.7], [LSPE+PI, true], [Converges to near-optimal policy],
    [Q8.8], [LSPE+PI, LQR], [Recovers #Klqr exactly],
    [Q8.9], [Constrained PI], [Saturation reduces reward; fair comparison needed],
  )

  #v(0.3em)
  #green[Takeaways]
  - LSPE naturally handles average-reward setting via Poisson error
  - Joint estimation of $theta_Q$ and $eta$ (average reward)
  - Constraint handling requires care in both improvement and evaluation
]

= Model Comparison
== Summary of methods and results
#slide[
  // #title[Summary]

  // #blue[Baselines]
  // - #picl: positive reward but infeasible (can short)
  // - #pipcl: feasible but reward drops during saturation

  // #v(0.2em)
  // #blue[Feasible improvement]
  // - CMA-ES: boosts performance; quadratic gives small extra gain

  // #v(0.2em)
  // #blue[Model-based]
  // - #pilqr: best unconstrained benchmark (high reward, interpretable)
  // - clipped LQR ≈ MPC (here), MPC costs more compute

  // #v(0.2em)
  // #blue[Learning]
  // - LSPI and LSPE+PI can recover near-LQR behavior with good features
  // - main difficulty is constraint handling (clipping / constrained PI)
   
  // Table with policy, avg reward, comments
  #let policy-table = table(
    columns: (1.5fr, 0.5fr, 0.5fr, 1fr, 1fr, 2fr),
    align: (center, center, center, center, center, left),
    [*Policy*], [*Model*], [*const*], [*Avg Reward*],[*Avg Reward (unc)*], [*Comments*],
    [#picl], [True], [#red[✗]], [0.005796], [0.011451], [baseline],
    // [#pipcl], [True], [#green[✓]], [], [], [saturation reduces reward],
    [CMA-ES], [True], [#green[✓]], [0.009654], [---], [better],
    // [CMA-ES quadratic], [True], [#green[✓]], [], [], [Marginal gain over linear],
    [#pilqr], [#red[LQR]], [#red[✗]], [0.009693], [0.019467], [],
    // [Clipped #pilqr], [LQR], [#green[✓]], [0.009616], [0.009914], [Feasibility cost but feasible],
    [MPC ($cal(N)=10$)], [#red[LQR]], [#green[✓]], [], [], [higher compute],
    [E-PIA], [#red[LQR]], [#red[✗]],[0.009651], [0.019584], [], 
    [#pilspi], [True], [#red[✗]], [], [], [near-LQR on true model],
    [#pilspepi approx], [#red[LQR]], [#red[✗]], [0.009833], [0.019534], [converges to #Klqr],
    [#pilspepi true], [True], [#red[✗]], [0.009763], [0.019452], [converges to #Klqr],
    [#pilspepi true constr], [True], [#green[✓]], [0.009725], [---], [],
    [$Q_lambda$], [], [],[0.000011], [0.000019],  [],
  )

  #policy-table

]

#slide()[

  
  #grid(
    columns: (1fr, 1fr),
    figure(
      image("figures/policy_comparison_bar_all_models.svg", width: 100%),
      caption: [Average reward comparison (unconstrained)]
    ),
    figure(
      image("figures/policy_comparison_bar_all_models_constrained.svg", width: 100%),
      caption: [Average reward comparison (constrained)]
    ),
  )

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
// = Conclusion


== The end
#slide[
  #title[Thank you for your attention !]
  
  #muted[
    Aymeric Couplet and Lucas Ahou
  ]
]

////////////////////////////////////////////////////////////
// Appendix (bonus + all extra details)
////////////////////////////////////////////////////////////
#show: appendix
= Appendix





== Appendix: $g_t$ as quadratic form (Q2.3)<touying:hidden>
#slide[
  #title[Quadratic form of $g_t$]

  $g_t = 1/2 y_t^top H y_t$ with
  $y_t = (q_t, z_t^a, z_t^u, u_t, xi_t^a, xi_t^p)^top$.

]

== Appendix: LQR Riccati and theoretical reward (Q4.4)<touying:hidden>
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

  #speaker-note()[
    We compute $M^*$ by iterating finite-horizon Riccati until convergence.
  ]
]

== Appendix: Baseline distributions (Q3.2 / Q3.4)<touying:hidden>
#slide[
  #title[Baseline distributions]

  #grid(
    columns: (1fr, 1fr),
    // figure(
    //   image("figures/unclipped_policy_final_reward_distribution_1000_trajectories.svg", width: 100%),
    //   caption: [Unclipped baseline]
    // ),
    // figure(
    //   image("figures/clipped_policy_final_reward_distribution_1000_trajectories.svg", width: 100%),
    //   caption: [Clipped baseline]
    // ),
  )
]

== Appendix: LQR distributions (Q4.6 / Q4.8)<touying:hidden>
#slide[
  #title[LQR distributions]

  #grid(
    columns: (1fr, 1fr),
    // figure(
    //   image("figures/lqr_optimal_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
    //   caption: [Unclipped LQR]
    // ),
    // figure(
    //   image("figures/lqr_clip_policy_final_reward_distribution_100_trajectories.svg", width: 100%),
    //   caption: [Clipped LQR]
    // ),
  )
]

== Appendix: MPC trajectories (Q4.10–Q4.11)<touying:hidden>
#slide[
  #title[MPC extra plots]

  #grid(
    columns: (1fr, 1fr),
    // figure(
    //   image("figures/mpc_optimal_policy_ex_trajectory_states_actions.svg", width: 100%),
    //   caption: [Example MPC states/actions]
    // ),
    // figure(
    //   image("figures/mpc_optimal_policy_cumulative_reward_100_trajectories.svg", width: 100%),
    //   caption: [MPC average reward over 100 sims]
    // ),
  )
]

== Appendix: Exact Policy Iteration (E-PIA) (Q5.1)<touying:hidden>
#slide[
  #title[E-PIA convergence]

  #grid(
    columns: (1fr, 1fr),
    // figure(
    //   image("figures/EPIA_convergence.svg", width: 100%),
    //   caption: [Error norm to #Klqr]
    // ),
    // figure(
    //   image("figures/EPIA_K_values.svg", width: 100%),
    //   caption: [$K_k$ components]
    // ),
  )
]

== Appendix: LSTD evaluation plots (Q6.3–Q6.4) <touying:hidden>
#slide[
  #title[LSTD evaluation]

  #grid(
    columns: (1fr, 1fr),
    // figure(
    //   image("figures/q63_lspd_vs_qhat.png", width: 100%),
    //   caption: [True system: $Q^theta$ vs $\hat Q$]
    // ),
    // figure(
    //   image("figures/q64_lspd_vs_qhat.png", width: 100%),
    //   caption: [LQR model: $Q^theta$ vs exact/empirical $Q$]
    // ),
  )
]

== Appendix: LSPI convergence and constrained PI (Q6.6–Q6.7) <touying:hidden>
#slide[
  #title[LSPI: convergence + constraints]

  #grid(
    columns: (1fr, 1fr),
    // figure(
    //   image("figures/q64_policy_convergence.png", width: 100%),
    //   caption: [LSPI gain converges to #Klqr]
    // ),
    // figure(
    //   image("figures/q67_policy_rewards.png", width: 100%),
    //   caption: [Constrained PI can reduce reward]
    // ),
  )
]

== Appendix: LSPE constrained comparison (Q8.9)<touying:hidden>
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
