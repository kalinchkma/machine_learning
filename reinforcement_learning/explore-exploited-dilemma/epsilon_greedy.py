import matplotlib.pyplot as plt
import numpy as np
import random
import math

# setup random seed
np.random.seed(777)


class Bandit:
    def __init__(self, p) -> None:
        self.p = p
        self.p_estimate = np.random.random()
        self.N = np.random.random()

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N += x
        self.p_estimate = np.random.random()


BANDIT_PROBABILITIES = [0.3, 0.5, 0.8, 0.4]
NUM_TRIALS = 10
EPS = 0.1


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)

    num_of_time_explored = 0
    num_of_time_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])

    for i in range(NUM_TRIALS):
        # Simple explore and exploited
        if np.random.random() < EPS:
            num_of_time_explored += 1
            j = int(np.floor(random.sample(range(0, 4), 1))[0])
        else:
            num_of_time_exploited += 1
            j = int(np.floor(random.sample(range(0, 4), 1))[0])

        print(j, optimal_j)
        if j == optimal_j:
            num_optimal += 1
        x = bandits[j].pull()
        print(x)
        rewards[i] = x
        print("Rewards", rewards[i])
        bandits[j].update(x)

    # Print mean estimates
    print("Mean Estimate:")
    for b in bandits:
        print("     ", b.p_estimate)

    # Print total reward
    print("total reward earnd:", rewards.sum(), " Reward", rewards)
    print("Overall win rate:", rewards.sum() / NUM_TRIALS)
    print("#num of times explored:", num_of_time_explored)
    print("#num of times exploited:", num_of_time_exploited)
    print("#num of times selected optimal bandit:", num_optimal)

    # plot the result
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    print(cumulative_rewards)
    print(np.arange(NUM_TRIALS) + 1)
    print("win rates", win_rates)
    plt.figure(figsize=(5, 5))
    plt.plot(win_rates, label="win rates")
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES), label="Trials")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    experiment()
