# %% [markdown]
# # LINMA2222 - Stochastic Optimal Control & Reinforcement Learning
# ## Portfolio optimal strategy
# ---

# %% [markdown]
# ### Imports

import numpy as np
import matplotlib.pyplot as plt

import os
os.makedirs("figures", exist_ok=True)

## desactive les plots pour tout le code ( sauvegarde juste le fichier svg )
# plt.show = lambda: plt.close('all')



rng = np.random.default_rng(42)  # For reproducibility

# %% [markdown]
# ### Constants

# %%
THETA = 0.5
W_A = 0.1
W_U = 0.2
SIGMA_A = 0.018
SIGMA_P = 0.02
BETA_U = -0.048
GAMMA_U = 0.06

# %% [markdown]
# ### Plot functions

# %%
def generate_trajectories(policy, x0=(0, 0, 0), T=1000, N=1, xi_a=None, xi_p=None):
    """Generate the state variables x_t and actions u_t for t=0,...,T
       using the given policy.

    Args:
        policy: a function that takes in the current state x_t and returns an action u_t
        x0 (ndarray): initial state. Defaults to 0.
        T (int): number of time steps. Defaults to 1000.
    """
    x = np.zeros((T+1, 3, N))  # State variables: q_t, za_t, zu_t
    u = np.zeros((T, N))       # Actions
    x[0] = np.array(x0).reshape(3, 1)
    if xi_a is None:  xi_a = rng.normal(0, 1, size=(T, N))
    if xi_p is None:  xi_p = rng.normal(0, 1, size=(T, N))

    for t in range(T):
        u[t] = policy(x[t])
        x[t+1, 0] = x[t, 0] + u[t] # q_t
        x[t+1, 1] = (1 - W_A) * x[t, 1] + W_A * SIGMA_A * xi_a[t]  # za_t
        x[t+1, 2] = (1 - W_U) * x[t, 2] + W_U * BETA_U * u[t]  # zu_t

    return x, u, xi_p


def g(q, za, zu , u, xi_p=None):
    """Compute the gross stage reward at each time given state x and action u."""
    # q, za, zu = x[:, 0], x[:, 1], x[:, 2]
    if xi_p is None:
        xi_p = rng.normal(0, 1, size=q.shape[0])

    # Formula found by replacing p_t+1 and p_t by their expressions in the given model (see Question 2.3)
    return 1000 * q * (za + zu + (GAMMA_U * u) + (SIGMA_P * xi_p)) + THETA * u * (za + zu + (SIGMA_P * xi_p))


def c(g):
    """Compute the net stage reward given gross stage reward g_t for each time t"""
    return np.maximum(g - (np.pow(g, 2) / 2), 1 - np.exp(-g))

def reward(x, u, xi_p, t):
    """Compute all the rewards over the trajectory given states x and actions u from time 0 to t."""
    """returns an array of shape (t,)"""
    g_t = g(x[:t, 0], x[:t, 1], x[:t, 2], u[:t], xi_p=xi_p[:t])
    c_t = c(g_t)
    return c_t

def average_reward(x, u, xi_p, t):
    return np.mean(reward(x, u, xi_p, t))


def generate_trajectories_and_avg_reward(policy, x0=(0, 0, 0), T=1000, N=1):
    """compute efficienctly the average reward over T time steps

    Args:
        policy: a function that takes in the current state x_t and returns an action u_t
        x0 (ndarray): initial state. Defaults to 0.
        T (int): number of time steps. Defaults to 1000.
    """
    u = 0
    x = np.zeros(3)

    total_reward = 0
    for t in range(T):
        u = policy(x)
        x[0] = x[0] + u # q_t
        x[1] = (1 - W_A) * x[1] + W_A * SIGMA_A * rng.normal(0, 1)  # za_t
        x[2] = (1 - W_U) * x[2] + W_U * BETA_U * u  # zu_t
        g_t = g(x[0], x[1], x[2], u, xi_p=rng.normal(0, 1))
        c_t = c(g_t)
        total_reward += c_t
    return total_reward / T


def plot_trajectories(x, u, xi_p, filename=None, mean=False, variance=False, policy_name=""):
    """Plot the trajectories of the state variables and actions over time in a 2x2 grid
       and add an additional plot for the average reward as a function of time."""
    
    print(f"Plotting {filename} : reward = {average_reward(x, u, xi_p, x.shape[0]-1)}")
    
    T = x.shape[0] - 1
    time = np.arange(T+1)
    # plt.title(f"Trajectories of states and actions over time\nPolicy: {policy_name}")
    # plt.title(policy_name)
    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # add the title above the 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Trajectories of states and actions over time\nPolicy: {policy_name}", fontsize=16)


    axs[0, 0].plot(time, x[:, 0], label=r'$q_t$', color='blue')
    if mean:
        axs[0, 0].hlines(np.mean(x[:, 0]), 0, T, colors='darkblue', linestyles='dashed', label='Mean')
    if variance:
        axs[0, 0].fill_between(time, np.mean(x[:, 0]) - np.std(x[:, 0]), np.mean(x[:, 0]) + np.std(x[:, 0]), color='cyan', alpha=0.3, label='Variance')
    axs[0, 0].set_title(r'$q_t$')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel(r'$q_t$')
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(time, x[:, 1], label=r'$z^{a}_{t}$', color='orange')
    if mean:
        axs[0, 1].hlines(np.mean(x[:, 1]), 0, T, colors='darkorange', linestyles='dashed', label='Mean')
    if variance:
        axs[0, 1].fill_between(time, np.mean(x[:, 1]) - np.std(x[:, 1]), np.mean(x[:, 1]) + np.std(x[:, 1]), color='moccasin', alpha=1, label='Variance')
    axs[0, 1].set_title(r'$z^{a}_{t}$')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel(r'$z^{a}_{t}$')
    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[1, 0].plot(time, x[:, 2], label=r'$z^{u}_{t}$', color='green')
    if mean:
        axs[1, 0].hlines(np.mean(x[:, 2]), 0, T, colors='darkgreen', linestyles='dashed', label='Mean')
    if variance:
        axs[1, 0].fill_between(time, np.mean(x[:, 2]) - np.std(x[:, 2]), np.mean(x[:, 2]) + np.std(x[:, 2]), color='lightgreen', alpha=1, label='Variance')
    axs[1, 0].set_title(r'$z^{u}_{t}$')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel(r'$z^{u}_{t}$')
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].plot(time[:-1], u, label=r'$u_{t}$', color='red')
    if mean:
        axs[1, 1].hlines(np.mean(u), 0, T-1, colors='darkred', linestyles='dashed', label='Mean')
    if variance:
        axs[1, 1].fill_between(time[:-1], np.mean(u) - np.std(u), np.mean(u) + np.std(u), color='lightcoral', alpha=1, label='Variance')
    axs[1, 1].set_title(r'$u_{t}$')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel(r'$u_{t}$')
    axs[1, 1].grid()
    axs[1, 1].legend()

    plt.tight_layout()
    if filename:
        plt.savefig("figures/" + filename + "_states_actions.svg", format='svg')
    # plt.show()

    # plot_average_reward(x, u, xi_p, T=T, filename=filename)







def plot_average_reward(x, u, xi_p, T=1000, filename=None, policy_name=""):
    # Plot average reward as a function of time
    avg_rewards = [np.mean(reward(x, u, xi_p, t)) for t in range(1, T+1)]
    
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, T+1), avg_rewards, label='Average Reward', color='purple')
    plt.title('Average Reward Over Time\nPolicy: ' + policy_name)
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.grid()

    plt.legend()
    if filename:
        plt.savefig("figures/" + filename + "_average_reward.svg", format='svg')
    # plt.show()



# %% [markdown]
# ### Question 3.1
# 
# $\pi_{cl}(x_t) = K_{cl} x_t$
# 

# %%
rng = np.random.default_rng(42)  # For reproducibility

# Question 3.1.1
policy = lambda x: -0.5 * x[0] + 0.5 * x[1] + 0.5 * x[2]
cliped_policy = lambda x: np.clip(policy(x), -x[0], 1-x[0])
random_policy = lambda x: rng.uniform(-x[0] * 0.1, (1 - x[0]) * 0.1)

x, u, xi_p = generate_trajectories(policy, x0=(0, 0, 0), T=1000)
plot_trajectories(x, u, xi_p, filename="question_3_1_1_unclipped", policy_name="Unclipped Initial Policy")
plot_average_reward(x, u, xi_p, filename="question_3_1_1_unclipped", policy_name="Unclipped Initial Policy")

x, u, xi_p = generate_trajectories(cliped_policy, x0=(0, 0, 0), T=1000)
plot_trajectories(x, u, xi_p, filename="question_3_1_1_clipped", policy_name="Clipped Initial Policy")
plot_average_reward(x, u, xi_p, filename="question_3_1_1_clipped", policy_name="Clipped Initial Policy")

# x, u, xi_p = generate_trajectories(random_policy, x0=(0, 0, 0), T=1000)
# plot_trajectories(x, u, xi_p, filename="question_3_1_1_random")
# plot_average_reward(x, u, xi_p, filename="question_3_1_1_random")


policy_list = [(policy, "Initial Policy"), (cliped_policy, "Clipped Initial Policy")]



# %%
### simulate trajectories and plot the average reward over time

rng = np.random.default_rng(42)  # For reproducibility

# Question 3.1.2
N = 1000
def run_trajectories(policy, N=1000, name="policy", show_all=False):
    """Run N trajectories using the given policy and plot the average reward over time."""
    if show_all:
        all_x = np.zeros(( 1001, 3, N))
        all_u = np.zeros((1000, N))
        all_xi_p = np.zeros((1000, N))

    all_rewards = np.zeros((1000, N))

    # for i in range(N):
    #     x, u, xi_p = generate_trajectories(policy, x0=(0, 0, 0), T=1000)
    #     if show_all:
    #         all_x[i] = x
    #         all_u[i] = u
    #         all_xi_p[i] = xi_p
    #     all_rewards[i] = reward(x, u, xi_p, 1000)

    all_x, all_u, all_xi_p = generate_trajectories(policy, x0=(0, 0, 0), T=1000, N=N)
    all_rewards = reward(all_x[:, :, :], all_u[:, :], all_xi_p[:, :], 1000).T

    reward_mean = np.mean(all_rewards, axis=0) # average reward for each time step

    print(f"Average reward of {name} over {N} trajectories: {np.mean(all_rewards)} and std: {np.std(all_rewards)}")


    ### Cumulative reward
    plt.figure(figsize=(9, 5))
    for i in range(N):
        plt.plot(np.arange(1, 1001), np.cumsum(all_rewards[i]) / (np.arange(1, 1001)), color='gray', alpha=0.05)
    plt.plot(np.arange(1, 1001), np.cumsum(reward_mean) / (np.arange(1, 1001)), label='Cumulative Average Reward', color='purple')
    plt.title(f'Cumulative Average Reward of {name} Over Time ({N} trajectories)')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Average Reward')
    plt.grid()
    # Save cumulative reward plot
    filename_safe = name.replace(" ", "_").lower()
    plt.savefig(f"figures/{filename_safe}_cumulative_reward_{N}_trajectories.svg", format='svg')
    # plt.show()


    ### show the average reward distribution
    plt.figure(figsize=(9, 5))
    # avg reward distribution as a density
    plt.hist(np.mean(all_rewards, axis=1), bins=50, color='purple', alpha=0.7, density=True)
    reward_final_mean = np.mean(all_rewards)
    reward_final_var = np.var(all_rewards)
    plt.text(0.02, 0.98, f'Mean: {reward_final_mean:.2e}\nVar: {reward_final_var:.2e}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # show the mean as a dashed line
    plt.axvline(reward_final_mean, color='darkviolet', linestyle='dashed', label='Mean')
    plt.legend()

    plt.title(f'Average Reward Distribution of {name} Over {N} trajectories')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid()
    # Save final reward distribution plot
    plt.savefig(f"figures/{filename_safe}_final_reward_distribution_{N}_trajectories.svg", format='svg')
    # plt.show()


    if show_all:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        for i in range(N):
            plt.plot(np.arange(1001), all_x[:, 0, i], color='blue', alpha=0.05)
        plt.title(r'$q_t$ over all trajectories')
        plt.xlabel('Time')
        plt.ylabel(r'$q_t$')
        plt.grid()
        plt.subplot(3, 1, 2)
        for i in range(N):
            plt.plot(np.arange(1001), all_x[:, 1, i], color='orange', alpha=0.05)
        plt.title(r'$z^{a}_{t}$ over all trajectories')
        plt.xlabel('Time')
        plt.ylabel(r'$z^{a}_{t}$')
        plt.grid()
        plt.subplot(3, 1, 3)
        for i in range(N):
            plt.plot(np.arange(1001), all_x[:, 2, i], color='green', alpha=0.05)
        plt.title(r'$z^{u}_{t}$ over all trajectories')
        plt.xlabel('Time')
        plt.ylabel(r'$z^{u}_{t}$')
        plt.grid()
        plt.tight_layout()
        # Save state trajectories plot
        plt.savefig(f"figures/{filename_safe}_state_trajectories_{N}_trajectories.svg", format='svg')
        # plt.show()

        ### show the distribution of all the state at all the times
        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(3, 1, 1)
        plt.hist(all_x[:, 0, :].flatten(), bins=50, color='blue', alpha=0.7, weights=np.ones_like(all_x[:, 0, :].flatten()) / all_x[:, 0, :].flatten().size)
        plt.title(r'Distribution of $q_t$ over all trajectories and times')
        plt.xlabel(r'$q_t$')
        plt.ylabel('Probability Density')
        plt.grid()
        # Add variance as text box
        q_var = np.var(all_x[:, 0, :].flatten())
        q_mean = np.mean(all_x[:, 0, :].flatten())
        plt.text(0.02, 0.98, f'Mean: {q_mean:.2e}\nVar: {q_var:.2e}', transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        ax2 = plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
        plt.hist(all_x[:, 1, :].flatten(), bins=50, color='orange', alpha=0.7, weights=np.ones_like(all_x[:, 1, :].flatten()) / all_x[:, 1, :].flatten().size)
        plt.title(r'Distribution of $z^{a}_{t}$ over all trajectories and times')
        plt.xlabel(r'$z^{a}_{t}$')
        plt.ylabel('Probability Density')
        plt.grid()
        # Add variance as text box
        za_var = np.var(all_x[:, 1, :].flatten())
        za_mean = np.mean(all_x[:, 1, :].flatten())
        plt.text(0.02, 0.98, f'Mean: {za_mean:.2e}\nVar: {za_var:.2e}', transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        

        ax3 = plt.subplot(3, 1, 3, sharey=ax1)
        plt.hist(all_x[:, 2, :].flatten(), bins=50, color='green', alpha=0.7, weights=np.ones_like(all_x[:, 2, :].flatten()) / all_x[:, 2, :].flatten().size)
        plt.title(r'Distribution of $z^{u}_{t}$ over all trajectories and times')
        plt.xlabel(r'$z^{u}_{t}$')
        plt.ylabel('Probability Density')
        plt.grid()
        # Add variance as text box
        zu_var = np.var(all_x[:, 2, :].flatten())
        zu_mean = np.mean(all_x[:, 2, :].flatten())
        plt.text(0.02, 0.98, f'Mean: {zu_mean:.2e}\nVar: {zu_var:.2e}', transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        # Save state and reward distributions plot
        plt.savefig(f"figures/{filename_safe}_state_distributions_{N}_trajectories.svg", format='svg')
        # plt.show()



run_trajectories(policy, N=N, name="unclipped policy", show_all=True)
run_trajectories(cliped_policy, N=N, name="clipped policy", show_all=True)

# x = np.mean(all_x, axis=0)
# u = np.mean(all_u, axis=0)
# xi_p = np.mean(all_xi_p, axis=0)
# r = np.mean(reward(x, u, xi_p, 1000))

# plot_trajectories(x, u, xi_p,  filename="question_3_1_2", mean=True, variance=True)


# %%
# reset best policy search
best_policy = None
best_reward = -np.inf
best_params = None

# %%
# Finding the best policy by random search



range_params = 1.0

def test_policy(policy, N=1000):
    reward_value = 0
    all_x, all_u, all_xi_p = generate_trajectories(policy, x0=(0, 0, 0), T=1000, N=N)
    all_rewards = reward(all_x[:, :, :], all_u[:, :], all_xi_p[:, :], 1000).T
    reward_value = np.mean(all_rewards)
    return reward_value

def get_linear_policy(params):
    return lambda x: np.clip(params @ np.array([x[0], x[1], x[2]]) , -x[0], 1 - x[0])

def linear_function(params, N=100): # linear function
    policy = get_linear_policy(params)
    reward_value = test_policy(policy, N=N)
    return reward_value

def get_quadratic_policy(params):
    return lambda x: np.clip(params @ np.array([x[0], x[1], x[2], x[0]**2, x[1]**2, x[2]**2, x[0]*x[1], x[0]*x[2], x[1]*x[2], np.ones_like(x[0])])
                               , -x[0], 1 - x[0])

def quadratic_function(params, N=100): # 10 params
    # quadratic function
    policy = get_quadratic_policy(params)
    reward_value = test_policy(policy, N=N)
    return reward_value





# %%
# !pip install cma

# import cma
# # CMA-ES optimization
# es = cma.CMAEvolutionStrategy([0]*3, 1, {'seed': 42,
#     'maxfevals': 20000,     
#     'popsize':100,           # increase population size for noisy problems
# })
# ### 0.5 is the initial standard deviation, increase it for more exploration
# es.optimize(lambda params: -linear_function(params, N=1000))#, verb_disp=1)
# # verb_disp=1 to see the progress in real time
# best_params_cma = es.result.xbest
# print(f"Best reward : {-es.result.fbest} with params : {' '.join([f'{p:.8f}' for p in es.result.xbest])}")

# best_params_cma = np.array([-0.0529, 37.2788, 29.9648])
best_params_cma = np.array([-0.8864, 2.1253, 1.2096])
best_policy_cma = get_linear_policy(best_params_cma)
policy_list.append((best_policy_cma, "Best linear Policy (CMA-ES)"))

# Best reward (CMA-ES): 0.0007081507843159959
# Best params (CMA-ES): [-0.03271245 18.91242714 -4.30407586]


# %%
# CMA-ES optimization
# es = cma.CMAEvolutionStrategy([0]*10, 1, {'seed': 42, 
#     'maxfevals': 20000,     
#     'popsize':100,           # increase population size for noisy problems
# })
# ### 0.5 is the initial standard deviation, increase it for more exploration
# es.optimize(lambda params: -quadratic_function(params, N=1000))
# best_params_cma_quadratic = es.result.xbest
# print(f"Best reward : {-es.result.fbest} with params : {' '.join([f'{p:.4f}' for p in es.result.xbest])}")
# verb_disp=1 to see the progress in real time
# result = -es.result.fbest

# result = 0.0
# best_params_cma_quadratic = np.array([-0.1674, 25.6970, 7.1492, 0.0350, 17.5269, -4.0057, 9.3930, 24.3607, -0.5704, 0.0603])
# best_policy_cma_quadratic = get_quadratic_policy(best_params_cma_quadratic)
# if result > -0.000005:
#     policy_list.append((best_policy_cma_quadratic, "Best quadratic Policy (CMA-ES)"))

# %%
# print(f"best params (CMA-ES): {best_params_cma}")
# best_policy_cma = get_linear_policy(best_params_cma)
# run_trajectories(best_policy_cma, N=1000, name="best policy (CMA-ES)", show_all=True)

run_trajectories(best_policy_cma, N=1000, name="best linear policy (CMA-ES)", show_all=True)
# run_trajectories(best_policy_cma_quadratic, N=1000, name="best policy (CMA-ES)", show_all=True)

# %%
# run for all 3 policies and store the rewards
N = 10000

print(f"Evaluation of all policies over {N} trajectories ...")

all_rewards = np.zeros((len(policy_list), N))
for i in range(len(policy_list)):
    x, u, xi_p = generate_trajectories(policy_list[i][0], x0=(0, 0, 0), T=1000, N=N)
    all_rewards[i] = np.mean(reward(x, u, xi_p, 1000), axis=0).T
    print(f"{policy_list[i][1]:30s}  Average reward = {np.mean(all_rewards[i]):.8f} , Std = {np.std(all_rewards[i]):.8f}")


# plot all the rewards
plt.figure(figsize=(10, 6))
labels = [policy_list[i][1] for i in range(len(policy_list))]
plt.hist(all_rewards.T, bins=30, label=labels, alpha=1)
# add the mean of each distribution
for i in range(all_rewards.shape[0]):
    plt.axvline(np.mean(all_rewards[i]), color=f'C{i}', linestyle='dashed', linewidth=1)
plt.xlabel('Average Reward')

plt.title('Distribution of Average Rewards for Different Policies')
plt.ylabel('Average Reward')
plt.grid()
plt.legend()
plt.savefig("figures/policy_comparison_histogram.svg", format='svg')
# plt.show()

# the integral plot
plt.figure(figsize=(10, 6))
for i in range(all_rewards.shape[0]):
    sorted_rewards = np.sort(all_rewards[i])
    cumulative = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
    plt.plot(sorted_rewards, cumulative, label=labels[i])
    plt.axvline(np.mean(all_rewards[i]), color=f'C{i}', linestyle='dashed', linewidth=1)

plt.xlabel('Average Reward')
plt.title('Cumulative Distribution of Average Rewards for Different Policies')
plt.ylabel('Cumulative Density')
plt.grid()
plt.legend()
plt.savefig("figures/policy_comparison_cumulative_distribution.svg", format='svg')

# %%


# x, u, xi_p = generate_trajectories(best_policy_cma_quadratic, x0=(0, 0, 0), T=1000)
# plot_trajectories(x, u, xi_p, filename="question_3_6_best_quadratic",  policy_name="Best quadratic Policy (CMA-ES)")
# plot_average_reward(x, u, xi_p, filename="question_3_6_best_quadratic")
plt.show()
