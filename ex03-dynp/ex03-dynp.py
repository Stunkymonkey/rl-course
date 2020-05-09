import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n
print(n_states)

def value_iteration(max_iterations=10000):
    V_states = np.zeros(n_states)  # init values as zero
    policy = V_states.copy()
    theta = 1e-8
    gamma = 0.8
    for i in range(max_iterations):
        last_V_states = V_states.copy()
        for state in range(n_states):
            action_values = []
            for action in range(n_actions):
                state_value = 0
                for action_step in env.P[state][action]:
                    p, n_state, r, is_terminal = action_step
                    state_value += p * (r + gamma * V_states[n_state])
                action_values.append(state_value)
            best_action = np.argmax(np.asarray(action_values))
            V_states[state] = action_values[best_action]
            policy[state] = best_action
        delta = np.abs(np.sum(np.array(V_states)) - np.sum(np.array(last_V_states)))
        if (delta < theta):
            print("#Steps to converge: ", i)
            break
                
    return np.array(policy).astype(int)





    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break


if __name__ == "__main__":
    main()
