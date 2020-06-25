#!/usr/bin/env python3
import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ return probabilities for actions under softmax action selection """
    scores = state.dot(theta)
    e = np.exp(scores)
    s = e.sum()
    return e / s


def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters

    ep_lengths = []
    mean_lengths = {'ep': [], 'length': []}

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        # keep track of previous 100 episode lengths and compute mean
        ep_lengths.append(len(states))

        if not e % 100:
            if e == 0:
                average_length = ep_lengths[0]
            else:
                average_length = sum(ep_lengths[-100:]) / 100
            mean_lengths['ep'].append(e)
            mean_lengths['length'].append(average_length)
            print(f'Episode: {e:>5d}, average length: {average_length:>3.1f}')

        # TODO: implement the reinforce algorithm to improve the policy weights

    plt.plot(mean_lengths['ep'], mean_lengths['length'], label="aggregated min rewards of 100 episodes")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('length.png')
    # plt.show()


def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()
