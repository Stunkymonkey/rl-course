#!/usr/bin/env python3

import gym
import numpy as np
import matplotlib.pyplot as plt


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    Q_values = np.zeros((env.observation_space.n, env.action_space.n))
    Q_values.fill(0.5)
    # make terminal states zero
    Q_values[((env.desc == b'H') | (env.desc == b'G')).flatten(), :] = 0

    for i in range(num_ep):
        state = env.reset()
        t = 0
        T = np.inf
        action = env.action_space.sample()

        actions = [action]
        states = [state]
        TD = []
        rewards = [0]
        while True:
            if t < T:
                state, reward, done, _ = env.step(action)
                states.append(state)
                rewards.append(reward)

                if done:
                    T = t + 1
                else:
                    action = env.action_space.sample()
                    actions.append(action)
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n + 1, T + 1)):
                    G += np.power(gamma, i - tau - 1) * rewards[i]
                if tau + n < T:
                    state_action = (states[tau + n], actions[tau + n])
                    G += np.power(gamma, n) * Q_values[state_action[0]][state_action[1]]
                state_action = (states[tau], actions[tau])
                Q_values[state_action[0]][state_action[1]] += alpha * (G - Q_values[state_action[0]][state_action[1]])
                TD.append(G)

            if tau == T - 1:
                break

            t += 1
    return Q_values, np.sqrt(np.mean(np.square(TD)))


env = gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
# print(nstep_sarsa(env, n=4, alpha=0.4, num_ep=10))

true_state_values = np.arange(-20, 22, 2) / 20.0

errors = dict()
TD_n = []
alpha_step = 31
for n in np.power(2, range(10)):
    TD_alpha = []
    for alpha in np.linspace(0, 1, alpha_step):  # 11
        _, TD_tmp_value = nstep_sarsa(env, n=n, alpha=alpha, num_ep=int(1e3))
        TD_alpha.append(TD_tmp_value)
    TD_n.append(TD_alpha)

for i in range(len(TD_n)):
    plt.plot(np.linspace(0, 1, alpha_step), TD_n[i], label=i)
plt.legend()
plt.show()
# plot
