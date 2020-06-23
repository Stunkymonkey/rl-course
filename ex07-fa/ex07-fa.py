#!/usr/bin/env python3
import gym
import numpy as np
import matplotlib.pyplot as plt


RENDER_EVERY = 2000
STATS_EVERY = 50

BUCKET_AMOUNT = [20, 20]
env = gym.make('MountainCar-v0')
BUCKET_SIZE = (env.observation_space.high - env.observation_space.low) / BUCKET_AMOUNT


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / BUCKET_SIZE
    return tuple(discrete_state.astype(np.int))


# tune learning rate
def qlearning(env, q_table, alpha=0.1, gamma=0.9, epsilon=0.1,
              initial_learning_rate=1.0, min_learning_rate=0.005, num_ep=int(25000)):

    ep_rewards = []
    ep_lengths = []
    ep_goal = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
    aggr_ep_goal = {'ep': [], 'amount': []}

    for episode in range(num_ep):
        episode_reward = 0
        episode_length = 0
        reached_goal = 0

        state = env.reset()
        discrete_state = get_discrete_state(state)
        done = False

        learning_rate = max(min_learning_rate, initial_learning_rate * (0.85 ** (episode // 100)))

        while not done:
            if np.random.uniform(0, 1) > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            if (episode + 1) % RENDER_EVERY == 0:
                env.render()

            q_table[discrete_state + (action,)] += learning_rate * (reward + gamma *
                                                                    np.max(q_table[new_discrete_state + (action,)]) -
                                                                    q_table[discrete_state + (action,)])

            if new_state[0] >= env.goal_position:
                reached_goal += 1

            discrete_state = new_discrete_state

            episode_reward += reward
            episode_length += 1

        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_length)
        ep_goal.append(reached_goal)

        if not episode % STATS_EVERY:
            if episode == 0:
                average_reward = sum(ep_rewards)
            else:
                average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
            if episode == 0:
                average_goal = sum(ep_goal)
            else:
                average_goal = sum(ep_goal[-STATS_EVERY:]) / STATS_EVERY
            aggr_ep_goal['ep'].append(episode)
            aggr_ep_goal['amount'].append(average_goal)
            print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, learning_rate: {learning_rate:>0.2f}')

    env.close()

    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'],
             label="aggregated average rewards of " + str(STATS_EVERY) + " episodes")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'],
             label="aggregated max rewards of " + str(STATS_EVERY) + " episodes")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'],
             label="aggregated min rewards of " + str(STATS_EVERY) + " episodes")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('reward.png')
    # plt.show()
    plt.clf()

    return aggr_ep_goal['ep'], aggr_ep_goal['amount']


def main():
    q_table = np.random.uniform(low=-2, high=0, size=(BUCKET_AMOUNT + [env.action_space.n]))
    episodes, reached_goals = qlearning(env, q_table)

    plt.plot(episodes, reached_goals, label="aggregated reaching goal of " + str(STATS_EVERY) + " episodes")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig('goal.png')
    # plt.show()
    env.close()


if __name__ == "__main__":
    main()
