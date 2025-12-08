#import "typst-uclouvain-frontpage/frontpage.typ": conf
#import "typst-uclouvain-frontpage/components.typ": *

#show: conf.with(
  lang: "en",
  cours: "LINMA2222 - Stochastic Optimal Control & RL",
  subject: "Project - Part 4",
  title: "Optimal Portfolio Strategy",
  students: (
    (name: "Lucas Ahou", noma: 35942200),
    (name: "Aymeric Couplet", noma: 59482200),
  ),
  teachers: (
    (name: "R. Jungers"),
    (name: "G. Bianchin"),
    (name: "G. Berger")
  ),
  heading_numbering: none
)


#let bg-color = rgb(150, 150, 150, 100)


= 8 Mean-square Poisson error


== Question 8.1
Compare the Poisson error with the Bellman error seen in the course for deterministic systems and for stochastic systems with discounted cost.

#answer(title: "Answer")[
  Given a policy $pi$ and approximations $Q^theta$ and $eta^theta$, the Poisson error is given by:

  $
    cal(P)^(theta)(x_t, u_t) := EE[r_t + Q^(theta)(x_(t+1), pi(x_(t+1))) | x_t, u_t] - eta^theta - Q^(theta)(x_t, u_t)
  $

  where $r_t$ is the stage reward, $eta^theta$ is an approximation of the average reward per stage and $Q^(theta)(dot, dot)$ is an approximation of the *relative* $Q$-function.

  Alternatively, the Bellman error is given by:

  $
    cal(B)^(theta)(x_t, u_t) := EE[r_t + gamma Q^(theta)(x_(t+1), pi(x_(t+1))) | x_t, u_t] - Q^(theta)(x_t, u_t)
  $

  with $gamma in [0, 1)$ the discount factor.

  - *Deterministic Systems*\
    Given a deterministic system of the following form:

    $
      x_(t+1) = f(x_t, u_t)
    $

    The Poisson error (PE) is then given by:

    $
    cal(P)^(theta)(x_t, u_t) = r_t + Q^(theta)(f(x_t, u_t), pi(f(x_t, u_t))) - eta^theta - Q^(theta)(x_t, u_t)
    $

    and the Bellman error (BE) by:

    $
      cal(B)^(theta)(x_t, u_t) = r_t + gamma Q^(theta)(f(x_t, u_t), pi(f(x_t, u_t))) - Q^(theta)(x_t, u_t)
    $

    The main differences are that PE has the $-eta^(theta)$ term and uses $gamma = 1$.

  - *Stochastic Systems*\
    For stochastic with discount $gamma$, PE is *not* valid. In fact, the Poisson equation assumes an ergodic average-reward setting as well as $gamma = 1$. In this case, BE must be used.

  In the context of our project, we want to maximize the infinite-horizon average reward. Because PE measures how well our approximations $Q^(theta)$ and $eta^theta$ are, this is the kind of problem where  we'd want to use it instead of BE.
]


== Question 8.2
Show that if $Q^(theta_1)$ and $eta^(theta_2)$ are linearly parametrized (i.e., $Q^(theta_1)(x, u) = theta^top_1 psi_(Q)(x, u)$ for some $psi_(Q) : cal(X) times cal(U) -> RR^d$ and $eta^(theta_2) = theta_2$), then minimizing the MSPE can be formulated as a linear system of equations to be solved in the least-square sense. Compute the matrix $A_"lspe"$ and the vector $b_"lspe"$ of the corresponding normal equations. Discuss the differences with the LSTD framework used in Part III.

#answer(title: "Answer")[
  In this setting, given a policy $pi$, PE is given by:

  $
    cal(P)^(theta)(x, u) := EE[r + Q^(theta_1)(x^+, pi(x^+)) | x, u] - eta^theta_2 - Q^(theta_1)(x, u)
  $

  where $Q^(theta_1)(x, u) = theta_1^top psi_(Q)(x, u)$ and $eta^theta_2 = theta_2$.
]

#answer(title: "Answer (cont.)")[
  If we define the _expected next feature vector_ as:

  $
    overline(psi)(x, u) := EE[psi_(Q)(x^+, pi(x^+)) | x, u]
  $

  and the expected reward as:
  
  $
    overline(r)(x, u) := EE[r | x, u]
  $

  Then:

  $
    EE[r + Q^(theta_1)(x^+, pi(x^+)) | x, u] = overline(r)(x, u) + theta_1^top overline(psi)(x, u)
  $

  and PE becomes:

  $
    cal(P)^(theta)(x,u) = overline(r)(x, u) + theta_1^top overline(psi)(x, u) - theta_2 - theta_1^top psi_(Q)(x, u)
  $

  As suggested in the hint, we will define the following augmented vector:

  $
    theta := mat(theta_1, theta_2, delim: "[")^top
  $

  and define also:

  $
    Psi(x, u) := vec(overline(psi)(x, u) - psi_(Q)(x, u), -1, delim: "[")\

    ==> cal(P)^(theta)(x, u) = overline(r)(x, u) + theta^top Psi(x, u)
  $

  We thus want to solve:

  $
    min_theta &EE_((x, u) tilde mu)[(overline(r)(x, u) + theta^top Psi(x, u))^2] := upright(L)(theta)\

    --> &gradient_(theta) upright(L)(theta) = 2 EE_((x, u) tilde mu)[Psi(x, u))(overline(r)(x, u) + theta^top Psi(x, u))] = 0\

    <==> &underbrace(EE_((x, u) tilde mu)[Psi(x, u)Psi^(top)(x, u)], A_"lspe") theta  = underbrace(- EE_((x, u) tilde mu)[Psi(x, u)overline(r)(x, u)], b_"lspe")
  $

  The main difference with the previous LSTD framework is that we obtain an explicit estimation of the average reward $eta$ via $theta_2$ obtained by solving the the regression problem above.
]


== Question 8.3

Propose an approximation of $A$ and $b$ in Question 8.2 from data obtained from the system controlled by some exploration policy $pi_"exp"$, assuming that $mu$ is the steady-state measure of the system controlled by $pi_"exp"$. Then, explain how such a subroutine can be implemented if you have the transition kernel $kappa(x^prime | x, u)$ of the system.

#answer(title: "Answer")[
  To approximate each quantity, we will first need to make ourselves a dataset. We will create the dataset as follows:

  $
    {(x_i, u_i, r_i, x^prime_i)}_(i=1)^(N)
  $

  where:
  - $(x_i, u_i) tilde mu$
  - $x^prime_i tilde kappa(dot | x_i, u_i)$
  - $r_i$ is the received reward

]

#answer(title: "Answer (cont.)")[
  To obtain the steady-state samples in practice, we run a long trajectory under $pi_"exp"$, discards an initial period and take samples spaced apart to reduce correlation.\
  Then, we have to be able to evaluate:

  $
    overline(psi)(x, u) = EE[psi_(Q)(x^+, pi(x^+)) | x, u]
  $

  We can do that under the stated assumption that we have access to a subroutine to compute:

  $
    EE[f(x^+, pi(x^+)) | x, u] = integral f(x^+, pi(x^+)) kappa(x^prime | x, u) dif x^prime
  $

  where $f$ would be $overline(psi)$ in our case. To implement this subroutine, we can either compute it analytically if we are in the special case where $kappa$ has a closed-form expression for its primitive, or if it is not possible, we can use a Monte-Carlo approximation:

  1. Given $(x, u)$, sample $M$ next states $x^(prime (1)), dots, x^(prime (M)) tilde kappa(dot | x, u)$

  2. Compute:
     $
       overline(psi)_("approx")(x, u) = 1/M sum_(m=1)^(M) psi_(Q)(x^(prime (m)), pi(x^(prime (m))))
     $

  To compute an approximation of $A$ and $b$, we need to do that for each sample $(x_i, u_i)$:

  $
    --> quad overline(psi)^((i))_("approx") = 1/M sum_(m=1)^(M) psi_(Q)(x_(i)^(prime (m)), pi(x_(i)^(prime (m))))
  $

  We then construct $hat(Psi)^((i))$:

  $
    --> quad hat(Psi)^((i)) = vec(overline(psi)^((i))_("approx") - psi_(Q)(x_i, u_i), -1) in RR^(d+1)
  $

  Finally, we can construct $hat(A)$ and $hat(b)$:

  $
    hat(A) &= 1/N sum_(i=1)^(N) hat(Psi)^((i))(hat(Psi)^((i)))^top\
    hat(b) &= -1/N sum_(i=1)^(N) r_i hat(Psi)^((i))
  $

  where we assumed that we have access to $r_i$ to act as an estimate for $overline(r)(x_i, u_i)$.
]