

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

THETA = 0.5
W_A = 0.1
W_U = 0.2
SIGMA_A = 0.018
SIGMA_P = 0.02
BETA_U = -0.048
GAMMA_U = 0.06

rng = np.random.default_rng(42)  # For reproducibility

def generate_trajectories(policy, x0=(0, 0, 0), T=1000, N=1, xi_a=None, xi_p=None, show_progress=False):
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

    # tqdm that goes away when done : 
    for t in range(T) if not show_progress else tqdm(range(T), desc="Generating trajectories"):
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
    """returns an array of shape (t,N)"""
    g_t = g(x[:t, 0], x[:t, 1], x[:t, 2], u[:t], xi_p=xi_p[:t])
    c_t = c(g_t)
    return c_t

def average_reward(x, u, xi_p, t):
    return np.nanmean(reward(x, u, xi_p, t))


def plot_trajectories(x, u, xi_p, filename=None, mean=False, variance=False, T=None, policy_name=""):
    """Plot the trajectories of the state variables and actions over time in a 2x2 grid
       and add an additional plot for the average reward as a function of time."""
    # print(f"Plotting {filename} : reward = {average_reward(x, u, xi_p, x.shape[0]-1)}")

    if T is None:
        T = x.shape[0] - 1
    # print(f"Plotting {filename} : reward = {average_reward(x, u, xi_p, T)}")
    
    time = np.arange(T+1)
    # plt.title(f"Trajectories of states and actions over time\nPolicy: {policy_name}")
    # plt.title(policy_name)
    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # add the title above the 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(9, 5))
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

    # plot_average_reward(x, u, xi_p, T=T, filename=filename)







def plot_average_reward(x, u, xi_p, T=1000, filename=None, policy_name=""):
    # Plot average reward as a function of time
    avg_rewards = [np.nanmean(reward(x, u, xi_p, t)) for t in range(1, T+1)]
    
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, T+1), avg_rewards, label='Average Reward', color='purple')
    plt.title('Average Reward Over Time of ' + policy_name)
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.grid()

    plt.legend()
    if filename:
        plt.savefig("figures/" + filename + "_average_reward.svg", format='svg')
    # plt.show()



# Question 3.1.2
def run_trajectories(policy, N=100, name="policy", show_all=False, T=1000, show_progress=False):
    """Run N trajectories using the given policy and plot the average reward over time."""
    if show_all:
        all_x = np.zeros(( T+1, 3, N))
        all_u = np.zeros((T, N))
        all_xi_p = np.zeros((T, N))

    all_rewards = np.zeros((T, N))

    # for i in range(N):
    #     x, u, xi_p = generate_trajectories(policy, x0=(0, 0, 0), T=T)
    #     if show_all:
    #         all_x[i] = x
    #         all_u[i] = u
    #         all_xi_p[i] = xi_p
    #     all_rewards[i] = reward(x, u, xi_p, T)

    all_x, all_u, all_xi_p = generate_trajectories(policy, x0=(0, 0, 0), T=T, N=N, show_progress=show_progress)
    all_rewards = reward(all_x[:, :, :], all_u[:, :], all_xi_p[:, :], T).T

    reward_mean = np.nanmean(all_rewards, axis=0) # average reward for each time step
    # skip nan values
    print(f"Average reward of {name} over {N} trajectories: {np.nanmean(all_rewards)} and std: {np.std(all_rewards)}")


    ### Cumulative reward
    plt.figure(figsize=(9, 5))
    # for i in range(N):
    #     plt.plot(np.arange(1, T+1), np.cumsum(all_rewards[i]) / (np.arange(1, T+1)), color='gray', alpha=0.05)
    # plt.plot(np.arange(1, T+1), np.cumsum(reward_mean) / (np.arange(1, T+1)), label='Average Reward', color='purple')

    # just the reward
    for i in range(N):
        plt.plot(np.arange(1, T+1), all_rewards[i], color='gray', alpha=0.05)
    plt.plot(np.arange(1, T+1), reward_mean, label='Average Reward', color='purple')
    plt.title(f'Average Reward by time of {name} Over {N} trajectories')
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.grid()
    # Save cumulative reward plot
    filename_safe = name.replace(" ", "_").lower()
    plt.savefig(f"figures/{filename_safe}_cumulative_reward_{N}_trajectories.svg", format='svg')
    # plt.show()


    ### show the reward distribution
    plt.figure(figsize=(9, 5))
    plt.hist(np.nanmean(all_rewards, axis=1), bins=50, color='purple', alpha=0.7, density=True)
    # max_reward = np.max(all_rewards, axis=1)
    # all_rewards[:, -1]
    reward_final_mean = np.nanmean(all_rewards)
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
        plt.figure(figsize=(9, 5))
        plt.subplot(3, 1, 1)
        for i in range(N):
            plt.plot(np.arange(T+1), all_x[:, 0, i], color='blue', alpha=0.05)
        plt.title(r'$q_t$ over all trajectories')
        plt.xlabel('Time')
        plt.ylabel(r'$q_t$')
        plt.grid()
        plt.subplot(3, 1, 2)
        for i in range(N):
            plt.plot(np.arange(T+1), all_x[:, 1, i], color='orange', alpha=0.05)
        plt.title(r'$z^{a}_{t}$ over all trajectories')
        plt.xlabel('Time')
        plt.ylabel(r'$z^{a}_{t}$')
        plt.grid()
        plt.subplot(3, 1, 3)
        for i in range(N):
            plt.plot(np.arange(T+1), all_x[:, 2, i], color='green', alpha=0.05)
        plt.title(r'$z^{u}_{t}$ over all trajectories')
        plt.xlabel('Time')
        plt.ylabel(r'$z^{u}_{t}$')
        plt.grid()
        plt.tight_layout()
        # Save state trajectories plot
        plt.savefig(f"figures/{filename_safe}_state_trajectories_{N}_trajectories.svg", format='svg')
        # plt.show()

        ### show the distribution of all the state at all the times
        plt.figure(figsize=(9,5))
        ax1 = plt.subplot(3, 1, 1)
        plt.hist(all_x[:, 0, :].flatten(), bins=50, color='blue', alpha=0.7, weights=np.ones_like(all_x[:, 0, :].flatten()) / all_x[:, 0, :].flatten().size)
        plt.title(r'Distribution of $q_t$ over all trajectories and times')
        plt.xlabel(r'$q_t$')
        plt.ylabel('Probability Density')
        plt.grid()
        # Add variance as text box
        q_var = np.var(all_x[:, 0, :].flatten())
        q_mean = np.nanmean(all_x[:, 0, :].flatten())
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
        za_mean = np.nanmean(all_x[:, 1, :].flatten())
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
        zu_mean = np.nanmean(all_x[:, 2, :].flatten())
        plt.text(0.02, 0.98, f'Mean: {zu_mean:.2e}\nVar: {zu_var:.2e}', transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        # Save state and reward distributions plot
        plt.savefig(f"figures/{filename_safe}_state_distributions_{N}_trajectories.svg", format='svg')
        # plt.show()


        # plot trajectories
        x, u, xi_p = generate_trajectories(policy, x0=(0, 0, 0), T=T)
        plot_trajectories(x, u, xi_p, filename=filename_safe + "_ex_trajectory", policy_name=name, T=T, mean=True, variance=True)
        plot_average_reward(x, u, xi_p, filename=filename_safe + "_ex_trajectory", policy_name=name, T=T)


def compare_policies(policy1, policy2, N=100):
    # generate random x of shape (3, N)
    x = np.random.randn(3, N)
    x[0, :] = np.clip(x[0, :], 0, 1)  # ensure x[0] in [0, 1]
    x[1, :] = np.clip(x[1, :], -0.01, 0.01)  # ensure x[1] in [-0.01, 1]
    x[2, :] = np.clip(x[2, :], -0.01, 0.01)  # ensure x[2] in [-5, 5]

    u1 = policy1(x)
    u2 = policy2(x)
    diff = np.abs(u1 - u2)
    print(f"Max difference between policies: {np.max(diff)}")
    print(f"Mean difference between policies: {np.mean(diff)}")