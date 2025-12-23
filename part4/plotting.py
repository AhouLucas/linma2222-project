

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import THETA, W_A, W_U, SIGMA_A, SIGMA_P, BETA_U, GAMMA_U, rng, reward, generate_trajectories, model_step

def plot_trajectories(x, u, xi_p, filename=None, mean=False, variance=False, T=None, policy_name=""):
    """Plot the trajectories of the state variables and actions over time in a 2x2 grid
       and add an additional plot for the average reward as a function of time."""
    # print(f"Plotting {filename} : reward = {average_reward(x, u, xi_p, x.shape[0]-1)}")

    if T is None:
        T = min(x.shape[0] - 1, u.shape[0])
        x = x[:T+1]
        u = u[:T]
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







def plot_average_reward(x, u, xi_p, T=None, filename=None, policy_name="", append=False, policy_label="", mode="avg"):
    # Plot average reward as a function of time
    if T is None: T = u.shape[0]
    if mode == "avg":
        avg_rewards = [np.nanmean(reward(x, u, xi_p, t)) for t in range(1, T+1)]
    if mode == "raw":
        avg_rewards = reward(x, u, xi_p, T)
    if mode == "cum":
        avg_rewards = np.cumsum(reward(x, u, xi_p, T))
    
    if not append:
        plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, T+1), avg_rewards, label='Average Reward ' + policy_label, color='purple' if not append else None)
    plt.title('Average Reward Over Time of ' + policy_name if mode == "avg" else 'Reward Over Time of ' + policy_name if mode == "raw" else 'Cumulative Reward Over Time of ' + policy_name)

    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.grid()

    plt.legend()
    if filename:
        plt.savefig("figures/" + filename + "_average_reward.svg", format='svg')
    # plt.show()



# Question 3.1.2
def run_trajectories(policy, N=100, name="policy", show_all=False, T=1000, show_progress=False, x0=(0, 0, 0), xi_a=None, xi_p=None):
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

    all_x, all_u, all_xi_p = generate_trajectories(policy, x0=x0, T=T, N=N, show_progress=show_progress, xi_a=xi_a, xi_p=xi_p)
    all_rewards = reward(all_x[:, :, :], all_u[:, :], all_xi_p[:, :], T).T
    # all_rewards[i] = reward(x, u, xi_p, T)


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

    if N != 1:
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
        x, u, xi_p = generate_trajectories(policy, x0=x0, T=T, xi_a=xi_a, xi_p=xi_p)
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


def plot_reward_distribution(policy_list, name="", n_traj=1000, n_traj_mpc=1, T=1000, deterministic=False, x0=(0, 0, 0), constrained=False):
    print(f"Evaluation of all policies over {n_traj} trajectories ...")
    title_suffix = " (Constrained)" if constrained else ""

    # all_rewards = np.zeros((len(policy_list), n_traj))
    # default to nan
    all_rewards = np.full((len(policy_list), n_traj), np.nan)

    xi_a = np.zeros((T, n_traj)) if deterministic else np.random.normal(0, 1, size=(T, n_traj))
    xi_p = np.zeros((T, n_traj)) if deterministic else np.random.normal(0, 1, size=(T, n_traj))
    for i in range(len(policy_list)):
        if constrained:
            # constraint q + u between 0 and 1
            policy = lambda x: np.clip(policy_list[i][0](x), -x[0], 1 - x[0])
        else:
            policy = policy_list[i][0]
        _n_used = n_traj_mpc if "mpc" in policy_list[i][1].lower() else n_traj
        xi_p_used = xi_p[:, :_n_used]
        x, u, _ = generate_trajectories(policy, x0=x0, T=T, N=_n_used, show_progress=True, xi_a=xi_a, xi_p=xi_p_used, desc=f"Evaluating {policy_list[i][1]:20s}")
        rewards_by_traj = np.nanmean(reward(x, u, xi_p_used, T), axis=0)
        # all_rewards[i] = rewards_by_traj
        all_rewards[i, :len(rewards_by_traj)] = rewards_by_traj
        # print(f"{policy_list[i][1]:30s}  Number of trajectories = {x.shape[2]}, trajectory length = {x.shape[0]}, reward shape = {reward(x, u, xi_p, T).shape}, avg reward shape = {np.nanmean(reward(x, u, xi_p, T), axis=0).T.shape}, number of non nan rewards = {np.sum(~np.isnan(all_rewards[i]))}")
        print(f"{policy_list[i][1]:30s}  Average reward = {np.nanmean(all_rewards[i]):.8f} , Std = {np.nanstd(all_rewards[i]):.8f} n_used = {np.sum(~np.isnan(all_rewards[i]))}")


    print(f"\nPOLICY REWARD SUMMARY: (constrained={constrained})")
    mean_rewards = [(f_name, np.nanmean(all_rewards[i]), np.nanstd(all_rewards[i]), np.sum(~np.isnan(all_rewards[i]))) for i, (policy, f_name) in enumerate(policy_list)]
    mean_rewards.sort(key=lambda x: x[1], reverse=True)
    for f_name, mean_reward, std_reward, count in mean_rewards:
        print(f"  Policy: {f_name:20s} | Mean Reward: {mean_reward:10.6f} | Std Dev: {std_reward:10.6f} | Samples: {count}")
    
    # bar plot of the mean rewards with error bars
    plt.figure(figsize=(10, 6))
    labels = [mean_rewards[i][0] for i in range(len(mean_rewards))]
    means = [mean_rewards[i][1] for i in range(len(mean_rewards))]
    stds = [mean_rewards[i][2] for i in range(len(mean_rewards))]
    plt.bar(labels, means, yerr=stds, capsize=5, color='skyblue', alpha=0.7)
    plt.ylabel('Average Reward')
    plt.title('Average Rewards for Different Policies' + title_suffix)
    plt.grid(axis='y')
    plt.xticks(rotation=30, ha='right', fontsize=6)
    plt.savefig(f"figures/policy_comparison_bar_{name}.svg", format='svg')


    # plot all the rewards
    plt.figure(figsize=(10, 6))
    labels = [policy_list[i][1] for i in range(len(policy_list))]
    # start the box to have 90% of the data
    # plt.hist(all_rewards.T, bins=30, label=labels, alpha=1, density=True)
    
    all_rewards_without_nan = all_rewards[~np.isnan(all_rewards)]
    start, end = np.percentile(all_rewards_without_nan, 5), np.percentile(all_rewards_without_nan, 95)

    plt.hist(all_rewards.T, bins=30, range=(start, end), label=labels, alpha=1, density=True)
    # there is some nan in the hist :
    # plt.hist([all_rewards[i][~np.isnan(all_rewards[i])] for i in range(all_rewards.shape[0])], bins=30, range=(start, end), label=labels, alpha=1, density=True)
    # add the mean of each distribution
    for i in range(all_rewards.shape[0]):
        plt.axvline(np.nanmean(all_rewards[i]), color=f'C{i}', linestyle='dashed', linewidth=1)
    plt.xlabel('Average Reward')

    plt.title('Distribution of Average Rewards for Different Policies' + title_suffix)
    plt.ylabel('Average Reward')
    plt.grid()
    ## clip to have 90% of the data
    plt.xlim(np.percentile(all_rewards_without_nan, 2), np.percentile(all_rewards_without_nan, 98))
    plt.legend()
    plt.savefig(f"figures/policy_comparison_histogram_{name}.svg", format='svg')


    # the integral plot
    plt.figure(figsize=(10, 6))
    for i in range(all_rewards.shape[0]):
        sorted_rewards = np.sort(all_rewards[i])
        cumulative = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
        plt.plot(sorted_rewards, cumulative, label=labels[i])
        plt.axvline(np.nanmean(all_rewards[i]), color=f'C{i}', linestyle='dashed', linewidth=1)

    plt.xlabel('Average Reward')
    plt.title('Cumulative Distribution of Average Rewards for Different Policies' + title_suffix)
    plt.ylabel('Cumulative Density')
    plt.grid()
    plt.legend()
    plt.savefig(f"figures/policy_comparison_cumulative_distribution_{name}.svg", format='svg')

    return all_rewards # array of shape (n_policies, n_traj)



def graph_K_evolution(name_K_array, K_lqr, indexes=None, K_lqr_name="K_{lqr}", title_suffix="during E-PIA Iteration"):
    plt.figure(figsize=(10, 6))
    for j, (name, K_array) in enumerate(name_K_array.items()):
        for i in range(K_array.shape[1]):
            if indexes is None:
                indexes = range(K_array.shape[0])

            plt.plot(indexes, K_array[:, i], label='$' + name + f'[{i}]={K_array[-1, i]:.2f}$', color='C'+str(i), linestyle=["-", "--", ":", "-."][j % 4])

    for i in range(K_array.shape[1]):
        plt.hlines(K_lqr[0, i], indexes[0], np.max(indexes), linestyles='dashed', label="$" + K_lqr_name + f"[{i}]={K_lqr[0, i]:.2f}$", colors='C'+str(i))
        # plt.hlines(K_lqr[0, i], 0, len(K_array)-1, linestyles='dashed', label="$K_{lqr}" +f"[{i}]={K_lqr[0, i]:.2f}$", colors='C'+str(i))

    plt.xlabel('Iteration')
    plt.ylabel(f'values')
    plt.title(f"Evolution of ${','.join(name_K_array.keys())}$ " + title_suffix)
    # plt.ylim(np.min(K_lqr) - 1, np.max(K_lqr) + 1)
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/K_evolution_{title_suffix.replace(' ', '_')}.svg", format='svg')



    ## Plot the evolution of the error norm between Kk and Klqr
    # K_lqr = K_opt
    # error_norms = [np.linalg.norm(K - K_lqr) for K in K_list]
    plt.figure(figsize=(10, 6))
    for name, K_array in name_K_array.items():
        error_norms = np.linalg.norm(K_array - K_lqr, axis=1)
        plt.semilogy(error_norms, label=f"error norm of ${name}$")
    plt.xlabel('Iteration')
    plt.ylabel(f'Norm of Error $||K_k - {K_lqr_name}||$')
    plt.title(f'Error norm between ${",".join(name_K_array.keys())}$ and ${K_lqr_name}$ ' + title_suffix)
    plt.grid()
    plt.legend()
    plt.savefig(f"figures/convergence_{title_suffix.replace(' ', '_')}.svg", format='svg')



def plot_Xfn_vs_Yfn(X_function, Y_function, data_x=None, data_u=None, n_points=100,seed=0, title="Comparison of ", x_label=r"X", y_label=r"Y"):

    rng = np.random.default_rng(seed)

    if data_x is None or data_u is None:
        from model import SIGMA_A, BETA_U
        scale = 0.01
        # around the origin
        ### just take random (x,u) pairs
        data_x = rng.normal(0, 1, size=(1000, 3)) * np.array([1.0, SIGMA_A, BETA_U]) * scale
        data_u = rng.normal(0, 1, size=(1000,)) * 0.1 * scale

    N = min(data_x.shape[0], data_u.shape[0])
    idxs = rng.choice(N, size=min(n_points, N), replace=False)

    X_vals = []
    Y_vals  = []
    xu_sizes    = []

    for k in idxs:
        x_k = data_x[k]
        u_k = data_u[k]

        X_vals.append(X_function(x_k, u_k))
        Y_vals.append(Y_function(x_k, u_k))
        xu_sizes.append(np.linalg.norm(np.concatenate([x_k.reshape(-1), np.array([u_k])])))

    X_vals = np.array(X_vals)
    Y_vals  = np.array(Y_vals)
    xu_sizes    = np.array(xu_sizes)
    
    offset = np.mean(Y_vals - X_vals)
    print(f"Poisson Q offset applied: {offset:.4f}")

    X_vals += offset


    plt.figure(figsize=(6, 6))
    ### Polyfit
    try:
        coeffs = np.polyfit(X_vals, Y_vals, deg=1)
        poly_fit = np.poly1d(coeffs)
        x_fit = np.linspace(X_vals.min(), X_vals.max(), 100)
        y_fit = poly_fit(x_fit)
        plt.plot(x_fit, y_fit, color='red', linestyle='-', label=f"Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.2f}")
        print(f"Polyfit : y = {coeffs[0]:.4f} x + {coeffs[1]:.4f}")
    except Exception as e:
        print(f"Polyfit failed: {e}")
        
    # plt.scatter(Q_lspe_vals, Q_hat_vals, s=15, alpha=0.7)
    # color by xu_sizes
    scatter = plt.scatter(X_vals, Y_vals, c=xu_sizes, s=30, cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(scatter)

    lo = min(X_vals.min(), Y_vals.min())
    hi = max(X_vals.max(), Y_vals.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title} {x_label} vs {y_label}")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    labels = (title + x_label + " vs " + y_label).replace("$", "").replace("{", "").replace("}", "").replace("\\", "").replace(" ", "_")
    plt.savefig(f'figures/comparison_{labels}.svg', format='svg')

