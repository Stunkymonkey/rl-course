import gym
import numpy as np
import itertools
import matplotlib.pyplot as plt

def policy(obs, sum = 20):
    return 0 if obs[0] >= sum else 1

def playEpisode(obs, env, done=False):
    state = obs[0]
    while not done:
        action = policy(obs)
        obs, reward, done, _ = env.step(action)

    return state, reward, obs[1], obs[2]

def plot_blackjack_values(V):

    def get_Z(x, y, usable_ace, runtime):
        if (x,y,usable_ace, runtime) in V:
            return V[x ,y ,usable_ace, runtime]
        else:
            return 0

    def get_figure(usable_ace, ax, xlim, runtime):
        
        x_range = np.arange(12, xlim)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([get_Z(x,y,usable_ace, runtime) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_zlim(-1,1)
        ax.set_ylim(1,10)
        ax.set_xlim(12,xlim-1)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot(221, projection='3d')
    ax.set_title('Usable Ace 10.000 episodes')
    get_figure(True, ax, 22, 10000)

    ax = fig.add_subplot(222, projection='3d')
    ax.set_title('No Usable Ace 10.000 episodes')
    get_figure(False, ax, 21, 10000)

    ax = fig.add_subplot(223, projection='3d')
    ax.set_title('Usable Ace 500.000 episodes')
    get_figure(True, ax, 22, 500000)

    ax = fig.add_subplot(224, projection='3d')
    ax.set_title('No Usable Ace 500.000 episodes')
    get_figure(False, ax, 21, 500000)
    plt.savefig('Ex2a.pdf', dpi=1000)
    plt.show()
    
def main():
    runtimes = [10000, 500000]
    states = [i for i in range(12,22)]
    dealer_states = [i for i in range(1,11)]
    states = list(itertools.product(states, dealer_states, [True, False], runtimes))
    returns = dict(zip(states, [[] for i in range(len(states))]))
    for runtime in runtimes:
        env = gym.make('Blackjack-v0')
        for _ in range(runtime):
            obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
            state, reward, dealercard,  usableAce = playEpisode(obs, env)
            state = 12 if state <= 12 else state
            returns[(state, dealercard, usableAce, runtime)].append(float(reward))
    V = {}
    for key, value in returns.items():
        if len(value) > 0:
            V[key] = (sum(value)/len(value))
    plot_blackjack_values(V)
    


if __name__ == "__main__":
    main()

