import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def make_agent_graphs(rewards_test_best, rewards_test_rand, rewards_test,
                          utilities_test_best, utilities_test_rand, utilities_test,
                          wealth_test_best, wealth_test_rand, wealth_test):

    block_utilities_test = np.mean(
        np.array(utilities_test).reshape(1000, -1), 0)
    block_utilities_test_rand = np.mean(
        np.array(utilities_test_rand).reshape(1000, -1), 0)
    block_utilities_test_best = np.mean(
        np.array(utilities_test_best).reshape(1000, -1), 0)

    block_wealth_test_rand = np.mean(
        np.array(wealth_test_rand).reshape(1000, -1), 0)
    block_wealth_test_best = np.mean(
        np.array(wealth_test_best).reshape(1000, -1), 0)
    block_wealth_test = np.mean(np.array(wealth_test).reshape(1000, -1), 0)

    block_rewards_test = np.mean(np.array(rewards_test).reshape(1000, -1), 0)
    block_rewards_test_rand = np.mean(
        np.array(rewards_test_rand).reshape(1000, -1), 0)
    block_rewards_test_best = np.mean(
        np.array(rewards_test_best).reshape(1000, -1), 0)

    # Calculate sharpe ratios instead of longitudinally do it at the end blocks of 1000 again

    #mu - rf/ sigma
    wr = np.array(wealth_test_rand).reshape(100, -1)
    wr_sharpe = (wr.mean(axis=0)/100-1)/(wr.std(axis=0)/100)

    wt = np.array(wealth_test).reshape(100, -1)
    wt_sharpe = (wt.mean(axis=0)/100-1)/(wt.std(axis=0)/100)

    wb = np.array(wealth_test_best).reshape(100, -1)
    wb_sharpe = (wb.mean(axis=0)/100-1)/(wb.std(axis=0)/100)

    sns.distplot(block_rewards_test_best, label="Merton optimal")
    sns.distplot(block_rewards_test, label="Trained Agent")
    sns.distplot(block_rewards_test_rand, label="Random agent")
    plt.title(
        'Distribution of Final rewards Merton v Trained Agent v Random (per 1000 episodes)')
    plt.xlabel('Episode Rewards')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.violinplot(
        [block_rewards_test_rand, block_rewards_test, block_rewards_test_best])
    plt.xticks([1, 2, 3], ["Random Agent", "Trained Agent",
               "Merton Optimal"], rotation=60, size=12)
    plt.ylabel("Test Rewards", size=12)
    plt.xlabel("Agent", size=12)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(0)
    plt.title("Violin plot of Rewards - Random v Merton v Agent")
    plt.show()

    sns.distplot(block_utilities_test_best, label="Merton optimal")
    sns.distplot(block_utilities_test, label="Trained Agent")
    sns.distplot(block_utilities_test_rand, label="Random agent")
    plt.title(
        'Distribution of Final Utilities Merton v Trained Agent v Random (per 1000 episodes)')
    plt.xlabel('Episode Utilities')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.violinplot([block_utilities_test_rand,
                   block_utilities_test, block_utilities_test_best])
    plt.xticks([1, 2, 3], ["Random Agent", "Trained Agent",
               "Merton Optimal"], rotation=60, size=12)
    plt.ylabel("Test Utility", size=12)
    plt.xlabel("Agent", size=12)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(0)
    plt.title("Violin plot of utilities - Random v Merton v Agent")
    plt.show()

    plt.plot(np.convolve(utilities_test_best, np.ones(
        (10000,))/10000, mode='valid'), label='Merton Optimal')
    plt.plot(np.convolve(utilities_test_rand, np.ones(
        (10000,))/10000, mode='valid'), label='Random agent')
    plt.plot(np.convolve(utilities_test, np.ones((10000,)) /
             10000, mode='valid'), label='Trained agent')
    plt.title('Moving average 10,000 episode utilities')
    plt.ylabel('Utility')
    plt.xlabel('Episode')
    plt.legend()
    plt.show()

    plt.violinplot(
        [block_wealth_test_rand, block_wealth_test, block_wealth_test_best])
    plt.xticks([1, 2, 3], ["Random Agent", "Trained Agent",
               "Merton Optimal"], rotation=60, size=12)
    plt.ylabel("Test Wealth", size=12)
    plt.xlabel("Agent", size=12)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(0)
    plt.title("Violin plot of Wealth - Random v Merton v Agent")
    plt.show()

    sns.distplot(block_wealth_test_best, label="Merton optimal")
    sns.distplot(block_wealth_test, label="Trained Agent")
    sns.distplot(block_wealth_test_rand, label="Random agent")
    plt.title(
        'Distribution of Final Wealth Merton v Trained Agent v Random (per 1000 episodes)')
    plt.xlabel('Episode Wealth')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    plt.violinplot([wr_sharpe, wt_sharpe, wb_sharpe])
    plt.xticks([1, 2, 3], ["Random Agent", "Trained Agent",
               "Merton Optimal"], rotation=60, size=12)
    plt.ylabel("Test Sharpe", size=12)
    plt.xlabel("Agent", size=12)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(0)
    plt.title("Violin plot of Sharpe ratios (M period) - Random v Merton v Agent")
    plt.show()

    sns.distplot(wb_sharpe, label="Merton optimal")
    sns.distplot(wt_sharpe, label="Trained Agent")
    sns.distplot(wr_sharpe, label="Random agent")
    plt.title(
        'Distribution of Final Sharpe ratios  Merton v Trained Agent v Random (per 1000 episodes)')
    plt.xlabel('Episode Sharpe')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    return block_utilities_test_rand, block_utilities_test, block_utilities_test_best, \
        block_rewards_test_rand, block_rewards_test, block_rewards_test_best, \
        block_wealth_test_rand, block_wealth_test, block_wealth_test_best, \
        wr_sharpe, wt_sharpe, wb_sharpe
