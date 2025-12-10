

import numpy as np
from scipy.linalg import solve_discrete_are

from model import W_A, W_U, BETA_U, GAMMA_U, THETA, SIGMA_P, SIGMA_A


#### SYSTEM MATRICES
# Define system matrices based on the problem description
F = np.array([[1, 0, 0], 
              [0, 1 - W_A, 0], 
              [0, 0, 1 - W_U]])

G = np.array([[1], 
              [0], 
              [W_U * BETA_U]])

D = np.array([[0, 0], [W_A * SIGMA_A, 0], [0, 0]])  # Disturbance matrix
  
# maximise r(x,u) = 0.5 x.T S x + x.T P u + 0.5 u.T R u 
# minimise - r(x,u)
# given    x_t+1 = F x_t + G u_t + D xi_t
S = np.array([[-(1000 * SIGMA_P)**2, 1000, 1000], 
                     [1000, 0, 0], 
                     [1000, 0, 0]]) 

P = np.array([[1000*GAMMA_U - THETA * ((1000 * SIGMA_P)**2)], 
                     [1000 * THETA], 
                     [1000 * THETA]]) 

R = np.array([[- (1000 * THETA * SIGMA_P)**2]])

# min c(x,u) = x.T Q x + u.T R u + 2 x.T S u = - 0.5 r(x,u)
Q = -0.5 * S
S = -0.5 * P
R = -0.5 * R

nu = 1  # number of inputs  = 1
nx = 3  # number of states  = 3
ny = 3  # number of outputs = 3

H = np.eye(ny)         # Output matrix (assuming full state observation)  # ny x nx
E = np.zeros((ny, 1))  # Direct feedthrough (assuming none)  # ny x nu

# we use F, G, Q, R, S from before
# min x.T Q x + u.T R u + 2 x.T S u
R_0 = Q # last stage cost  # nx x nx

model = (F, G, H, E, D, Q, R, S)


def solve_ricatti_infinite_horizon(F, G, Q, R, S, max_iterations=10_000, tolerance=1e-10):
    """Solve the discrete-time algebraic Riccati equation for infinite horizon LQR.
    """
    # Initialize with Q (not zero) for better convergence
    P = Q.copy()
    for i in range(max_iterations):
        # CORRECT Riccati iteration for MINIMIZATION problem
        P_next = Q + F.T @ P @ F - (F.T @ P @ G + S) @ np.linalg.inv(R + G.T @ P @ G) @ (G.T @ P @ F + S.T)

        if np.abs(np.max(P - P_next)) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
        P = P_next
    else:
        print("Warning: Riccati iteration did not converge")
    
    return P

def optimal_gain(A, B, Q, R, S, M):
    K = np.linalg.inv(R + B.T @ M @ B) @ (B.T @ M @ A + S.T)
    return K


def compute_lqr_gain(model):
    F, G, H, E, D, Q, R, S = model
    # M = solve_ricatti_infinite_horizon(F, G, Q, R, S)
    # K_opt = optimal_gain(F, G, Q, R, S, M)

    M = solve_discrete_are(F, G, Q, R, e=np.eye(3), s=S)
    K_lqr = -np.linalg.solve(R + G.T @ M @ G, (G.T @ M @ F + (1*S).T))

    residual = F.T @ M @ F - (F.T @ M @ G + S) @ np.linalg.inv(R + G.T @ M @ G) @ (G.T @ M @ F + S.T) + Q - M
    print(f"K_lqr : {K_lqr}  residual norm: {np.linalg.norm(residual, ord='fro')}")

    # Check closed-loop stability
    F_cl = F + G @ K_lqr
    eigenvalues = np.linalg.eigvals(F_cl)
    print("Stable:", all(np.abs(eigenvalues) < 1))

    # K_opt = - np.array([[-0.8864, 2.1253, 1.2096]])


    # Compute average approximate reward
    J = - np.linalg.trace(M @ D @ D.T)
    # J = - np.trace(M @ D @ D.T)
    print("Average approximate reward J:", J)

    return K_lqr



def model_step_approx(x, u, xi_a=None):
    """Compute the next state using the approximate model (without noise)."""
    F, G, H, E, D, Q, R, S = model
    x = np.asarray(x, dtype=float).reshape(-1)   # (nx,)
    u = np.asarray(u, dtype=float).reshape(-1)   # (nu,)

    x_next = F @ x + G @ u
    return x_next

def stage_reward_approx(x_t, u_t, xi_p_t=None):
    """Compute the approximate reward using the quadratic cost function."""
    F, G, H, E, D, Q, R, S = model
    x = np.asarray(x_t, dtype=float).reshape(-1)     # (nx,)
    u = np.asarray(u_t, dtype=float).reshape(-1)     # (nu,)

    r = - (x.T @ Q @ x + 2 * x.T @ S @ u + u.T @ R @ u)  # maximize reward
    return r



def get_lqr_policy():
    K_lqr = compute_lqr_gain(model)
    policy_lqr = lambda x: float(K_lqr @ x)
    return policy_lqr




########## DEBUGGING LQR cost ######################


# def true_cost_analytic(x, u):
#     q, za, zu = x  # assuming x = (q, za, zu)
#     a = 1000 * (q * (za + zu + GAMMA_U * u) + THETA * u * (za + zu))
#     b = 1000 * SIGMA_P * (q + THETA * u)

#     r_exact = a - 0.5 * (a**2 + b**2)  # E[c_quad(g)]
#     return -r_exact                   # cost = -reward


# saved = []
# for i in range(100):
#     x = np.multiply(rng.normal(0, 1, size=(3)), np.array([0.5, SIGMA_A, BETA_U])) * 1e-3
#     u = np.random.normal(0, 1, size=(1,)) * 1e-3
#     c_true = true_cost_analytic(x, u[0])
#     c = float(x.T @ Q @ x + 2 * x.T @ S @ u + u.T @ R @ u) # minimize this cost 

#     x_next_true = model_step(x, u[0], xi_a=0)
#     x_next = F @ x + G @ u # skipped : + D @ (0.1 * np.random.randn(D.shape[1]))

#     saved.append((x, u, c_true, x_next_true, c, x_next))


# ##### graphs
# x_array = np.array([item[0] for item in saved])
# u_array = np.array([item[1] for item in saved])
# c_true_array = np.array([item[2] for item in saved])
# x_next_true_array = np.array([item[3] for item in saved])
# c_array = np.array([item[4] for item in saved])
# x_next_array = np.array([item[5] for item in saved])
# # plot cost comparison
# plt.figure(figsize=(10, 6))
# color_intensity = np.sum(np.pow(x_array, 2), axis=1) + np.sum(np.pow(u_array, 2), axis=1)
# plt.scatter(c_array, c_true_array, alpha=0.5, c=color_intensity, cmap='viridis')
# plt.plot([np.min(c_array), np.max(c_array)], [np.min(c_array), np.max(c_array)], 'r--', label="y=x")
# plt.xlabel("Approximate Model Cost")
# plt.ylabel("True Model Cost")
# plt.title("Cost Comparison between Approximate and True Model")
# plt.legend()
# plt.grid()
# plt.savefig("figures/cost_comparison_approx_true_model.svg", format='svg')

# # plot next state comparison for each state variable
# state_labels = ['x[0]', 'x[1]', 'x[2]']
# for i in range(3):
#     plt.figure(figsize=(10, 6))
#     plt.scatter(x_next_array[:, i], x_next_true_array[:, i], alpha=0.5)
#     plt.plot([np.min(x_next_array[:, i]), np.max(x_next_array[:, i])], [np.min(x_next_array[:, i]), np.max(x_next_array[:, i])], 'r--', label="y=x")
#     plt.xlabel(f"Approximate Model Next State {state_labels[i]}")
#     plt.ylabel(f"True Model Next State {state_labels[i]}")
#     plt.title(f"Next State Comparison for {state_labels[i]} between Approximate and True Model")
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"figures/next_state_comparison_{state_labels[i]}_approx_true_model.svg", format='svg')
# policy_list = []


# a, b = np.polyfit(c_array, c_true_array, 1)
# print("slope:", a, "intercept:", b)


