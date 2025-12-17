"""
Compare all models from Part 4 of the LINMA2222 project.

This script evaluates and compares all the control policies developed:
1. Initial/Baseline policies (closed-loop, clipped)
2. CMA-ES optimized linear policy
3. LQR optimal policy (quadratic approximation)
4. MPC (Model Predictive Control)
5. E-PIA (Exact Policy Iteration Algorithm)
6. Q-λ learning (on approximate and true system)
7. LSPE+PI (Least-Squares Policy Evaluation + Policy Improvement)
"""

import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
rng = np.random.default_rng(42)
np.random.seed(42)

# ============================================================================
# IMPORTS FROM LOCAL MODULES
# ============================================================================
import sys
import os

# Add part4 directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import K_cl, SIGMA_A, BETA_U
from plotting import plot_reward_distribution, graph_K_evolution
from lqr import compute_lqr_gain, model as model_matrices


# ============================================================================
# POLICY DEFINITIONS
# ============================================================================

def create_all_policies(): # Create and return a dictionary of all policies to compare.
    policies = {}
    
    # ----- 1. Baseline Policies -----
    GREEN = "\033[92m"
    print(GREEN + "LOADING POLICIES:" + "\033[0m")
    policy_cl = lambda x: float(K_cl @ x)
    policy_cl_clipped = lambda x: np.clip(float(K_cl @ x), -x[0], 1 - x[0])
    policies["Initial CL"] = policy_cl
    policies["Clipped CL"] = policy_cl_clipped


    # print(GREEN + " - Random Policy" + "\033[0m")
    # random_policy = lambda x: rng.uniform(-x[0] * 0.1, (1 - x[0]) * 0.1)
    # policies["Random"] = random_policy

    
    # ----- 2. CMA-ES Optimized Policy -----
    from best_policy import get_cma_policy, get_cma_quadratic_policy
    policies["CMA-ES Linear"] = get_cma_policy()
    # policies["CMA-ES Quadratic"] = get_cma_quadratic_policy()

    
    # ----- 3. LQR Optimal Policy -----
    from lqr import get_lqr_policy
    policies["LQR"] = get_lqr_policy(model=model_matrices)
    policies["LQR Clipped"] = get_lqr_policy(model=model_matrices, clipped=True)
    
    # ----- 4. MPC Policy -----
    from mpc_condensed import get_mpc_policy
    policies["MPC (N=10)"] = get_mpc_policy(N=10, model=model_matrices)
    # policies["MPC (N=20)"] = get_mpc_policy(N=20, model=model_matrices)
    
    # ----- 5. E-PIA Policy -----

    from e_pia import get_epia_policy
    policies["E-PIA"] = get_epia_policy(model=model_matrices)

    # ----- 6. Q-λ Learning Policy -----
    from Q_lambda import get_q_lambda_policy
    policies["Q-λ (True Sys)"] = get_q_lambda_policy(use_precomputed=True)
    

    return policies


def compare_all_policies(policy_list, N=1000, T=1000, x0=(0, 0, 0), save_prefix="", constrained=False, n_traj_mpc=1):
    """Compare all policies and generate comparison plots."""

    print(f"COMPARING {len(policy_list)} POLICIES, N={N} trajectories, T={T} time steps, x0={x0}, constrained={constrained}")
    
    # Use the existing plot_reward_distribution function
    all_rewards = plot_reward_distribution(policy_list, name=save_prefix.rstrip("_"), n_traj=N, n_traj_mpc=n_traj_mpc, T=T, x0=x0, constrained=constrained)
    
    # print results
    return all_rewards



def main():
    """Main function to run all comparisons."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare all Part 4 models")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer trajectories", default=False)
    parser.add_argument("--no-mpc", action="store_true", help="Skip MPC (slow)", default=False)
    # parser.add_argument("--no-evolution", action="store_true", help="Skip K evolution plots", default=False)
    args = parser.parse_args()
    
    
    # Create all policies
    policies = create_all_policies()


    
    # Remove MPC if requested (it's slow)
    if args.no_mpc and "MPC (N=10)" in policies:
        del policies["MPC (N=10)"]
        print("Skipping MPC policy (--no-mpc flag)")
    
    
    # Set N and T based on quick mode
    N = 100 if args.quick else 5000
    T = 200 if args.quick else 1000
    n_traj_mpc = 2
    
    # Compare without constraints (standard evaluation)
    policy_list = [(policy, name) for name, policy in policies.items()]

    print("=" * 60 + "STANDARD EVALUATION (no constraints enforced)" + "="*60)
    results_standard = compare_all_policies(policy_list, N=N, T=T, save_prefix="all_models_", n_traj_mpc=n_traj_mpc)
    
    print("=" * 60 + "CONSTRAINED EVALUATION (q ∈ [0, 1] enforced)" + "="*60)
    results_constrained = compare_all_policies(policy_list, N=N, T=T, save_prefix="all_models_constrained_", constrained=True, n_traj_mpc=n_traj_mpc)
    
    # list all
    # mean_rewards = [(name, np.nanmean(all_rewards[i]), np.nanstd(all_rewards[i]), len(all_rewards[i])) for i, (policy, name) in enumerate(policy_list)]
    # mean_rewards.sort(key=lambda x: x[1], reverse=True)
    # for name, mean_reward, std_reward, count in mean_rewards:
    #     print(f"  Policy: {name:20s} | Mean Reward: {mean_reward:10.6f} | Std Dev: {std_reward:10.6f} | Samples: {count}")
    
    all_results = [(name, np.nanmean(results_standard[i]), np.nanmean(results_constrained[i])) for i, (policy, name) in enumerate(policy_list)]
    all_results.sort(key=lambda x: x[2], reverse=True)
    print("\nFINAL POLICY COMPARISON SUMMARY:")
    for name, mean_std, mean_constr in all_results:
        print(f"  Policy: {name:20s} | Mean Reward: {mean_constr:10.6f} | Mean Reward (unconstr): {mean_std:10.6f} | [{mean_constr:.6f}], [{mean_std:.6f}]")

    print("=" * 60+ "COMPARISON COMPLETE" + "="*60)
    print("\nPlots saved to figures/ directory")
    
    plt.show()


if __name__ == "__main__":
    main()
