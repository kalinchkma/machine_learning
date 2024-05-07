import matplotlib.pyplot as plt
import numpy as np
import random
import math

np.random.seed(42)

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p) -> None:
        # p: True win rate
        self.p = p
        self.p_estimate = 0.0
        self.N = 0.0  # number of samples collected so far

    def pull(self):
        # Draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.0
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_of_time_explored = 0
    num_of_time_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    print("optimal j:", optimal_j)

    for i in range(NUM_TRIALS):
        # use epsilon-greedy to select the next bandit
        if np.random.random() < EPS:
            num_of_time_explored += 1
            j = int(np.floor(random.sample(range(0, 3), 1))[0])
        else:
            num_of_time_exploited += 1
            j = int(np.floor(random.sample(range(0, 3), 1))[0])

        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # Update reward log
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print mean estimates for each bandit
    for b in bandits:
        print("mean estimate:", b.p_estimate)

    # print total reward
    print("total reward earnd:", rewards.sum(), " reward", rewards[0:5])
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num_times_explored:", num_of_time_explored)
    print("num_times_exploited:", num_of_time_exploited)
    print("num times selected optimal bandit:", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.figure(figsize=(5, 5))
    plt.plot(win_rates, label="win rates")
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES), label="Trials")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    experiment()
