# !pip install cma

import cma
import numpy as np
from model import generate_trajectories, reward


def test_policy(policy, N=1000):
    reward_value = 0
    all_x, all_u, all_xi_p = generate_trajectories(policy, x0=(0, 0, 0), T=1000, N=N)
    all_rewards = reward(all_x[:, :, :], all_u[:, :], all_xi_p[:, :], 1000).T
    reward_value = np.nanmean(all_rewards)
    return reward_value

def get_linear_policy(params):
    return lambda x: np.clip(params @ np.array([x[0], x[1], x[2]]) , -x[0], 1 - x[0])

def linear_function(params, N=100): # linear function
    policy = get_linear_policy(params)
    reward_value = test_policy(policy, N=N)
    return reward_value



def get_cma(recompute=False):
    best_params_cma = np.array([-0.8864, 2.1253, 1.2096])
    if recompute:
        es = cma.CMAEvolutionStrategy([0]*3, 1, {'seed': 42,
            'maxfevals': 20000,     
            'popsize':100,           # increase population size for noisy problems
        })
        ### 0.5 is the initial standard deviation, increase it for more exploration
        es.optimize(lambda params: -linear_function(params, N=1000))#, verb_disp=1)
        # verb_disp=1 to see the progress in real time
        print(f"Best reward : {-es.result.fbest} with params : {' '.join([f'{p:.4f}' for p in es.result.xbest])}")
        best_params_cma = es.result.xbest
    best_policy_cma = get_linear_policy(best_params_cma)
    return best_params_cma, best_policy_cma





# 17 minutes


###################### QUADRATIC POLICY

def get_quadratic_policy(params):
    return lambda x: np.clip(params @ np.array([x[0], x[1], x[2], x[0]**2, x[1]**2, x[2]**2, x[0]*x[1], x[0]*x[2], x[1]*x[2], np.ones_like(x[0])])
                               , -x[0], 1 - x[0])

def quadratic_function(params, N=100): # 10 params
    # quadratic function
    policy = get_quadratic_policy(params)
    reward_value = test_policy(policy, N=N)
    return reward_value

def get_cma_quadratic(recompute=False):
    if not recompute: raise NotImplementedError("No precomputed params for quadratic policy.")
    es = cma.CMAEvolutionStrategy([0]*10, 1, {'seed': 42, 
        'maxfevals': 20000,     
        'popsize':100,           # increase population size for noisy problems
    })
    ### 0.5 is the initial standard deviation, increase it for more exploration
    es.optimize(lambda params: -quadratic_function(params, N=1000))
    # verb_disp=1 to see the progress in real time
    best_params_cma_quadratic = es.result.xbest
    best_policy_cma_quadratic = get_quadratic_policy(best_params_cma_quadratic)
    print(f"Best reward : {-es.result.fbest} with params : {' '.join([f'{p:.4f}' for p in es.result.xbest])}")
    return best_params_cma_quadratic, best_policy_cma_quadratic
# 21 min



def get_cma_policy(recompute=False):
    _, policy = get_cma(recompute=recompute)
    return policy

def get_cma_quadratic_policy(recompute=False):
    _, policy = get_cma_quadratic(recompute=recompute)
    return policy


#### SCIPY NOT WORKING WELL


# # Advanced optimization for policy search (example: Differential Evolution and Bayesian Optimization)
# from scipy.optimize import differential_evolution

# if target_function == linear_function:
#     param_bounds = [(-100.0, 100.0)] * 3
# else:
#     param_bounds = [(-100.0, 100.0)] * 10


# # result = differential_evolution(lambda params: -target_function(params), param_bounds, seed=42, maxiter=100, polish=True)
# # print in real time the progress of the optimization
# result = differential_evolution(lambda params: -target_function(params, N=1000),  param_bounds, seed=42, maxiter=100, 
#                                 polish=True, callback=lambda x, f: print(f"Current best reward: {-f} with params: {" ".join([f'{p:.4f}' for p in x])}"))
# best_params_de = result.x

# print('Best reward (DE):', -result.fun)
# print('Best params (DE):', result.x)
# print(f"ALL outputs : {result}")


# # Best reward (DE): 0.00014893476071625646
# # [-0.03010771  0.98899323  0.30023768  0.02542109 -0.11388999  0.04947695 0.75390886 -0.4405498   0.30153338  0.00570656]
# # [-0.03502989  0.93203422  0.85333061  0.02800826 -0.38609742 -0.41453566  0.58533717  0.38185557 -0.93132613  0.0058954 ]
# #Average reward of best policy (DE) over 1000 trajectories: 0.00012731321932935195 and std: 0.004505140123694079

# # [-0.0326857   0.99948075 -0.83836307  0.01297369 -0.32935196 -0.16898026 0.93881118  0.74922339  0.13095927  0.01325202]
# # Average reward of best policy (DE) over 1000 trajectories: 0.00013211187504596597 and std: 0.010698030071014887


# # Best params (DE): [ 4.12655385e-04  9.14383436e+00  5.07172512e-01 -1.21986547e-01
# #   3.11259952e+00 -2.23009240e+00  8.06625065e+00 -6.74134828e+00
# #   7.11042371e+00  2.20034138e-02]
# # Average reward of best policy (DE) over 1000 trajectories: 0.0006561312720513467 and std: 0.010407829783256821
 

# best_policy_de = get_quadratic_policy(best_params_de)
# run_trajectories(best_policy_de, N=1000, name="best policy (DE)", show_all=True)