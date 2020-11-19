import numpy
import numpy as np
import gym
import matplotlib.pyplot as plt
import datetime


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action


def discretize_space(space):
    state_n = (space.high - space.low) * \
              np.array([10, 100])
    state_n = np.round(state_n, decimals=0).astype(int) + 1
    return state_n


def discretize_state(state):
    # print(state)
    state_d = (state - env.observation_space.low) * np.array([10, 100])
    state_d = np.round(state_d, 0).astype(int)
    # print(state_d)
    # print()
    return state_d


def QLearning(env,
              episode_n,
              noisy_episode_n,
              gamma=0.9,
              alpha=0.2):
    state_n = discretize_space(env.observation_space)
    action_n = env.action_space.n

    # [0] - position
    # [1] - speed
    Q = np.zeros((state_n[0], state_n[1], action_n))
    epsilon = 0.8

    total_rewards = []
    total_rewards_avg = []
    for episode in range(episode_n):
        total_reward = 0
        state = env.reset()
        state = discretize_state(state)
        done = False
        while not done:
            action = get_epsilon_greedy_action(
                Q[state[0], state[1]],
                epsilon,
                action_n)
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state)
            # if done and next_state[0] >= 0.5:
            #     Q[state[0], state[1], action] = reward
            # else:
            Q[state[0], state[1]][action] += alpha * (
                reward + gamma * np.max(Q[next_state[0], next_state[1]])
                - Q[state[0], state[1]][action])

            total_reward += reward

            if done:
                break

            state = next_state

        epsilon = max(0, epsilon - 1 / noisy_episode_n)

        total_rewards.append(total_reward)

        if episode % 100 == 0:
            rewards_avg = np.mean(total_rewards)
            total_rewards_avg.append(rewards_avg)
            total_rewards = []
            print('Episode ', episode)
            print('Avg reward ', rewards_avg)

    return total_rewards_avg


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    episode_n = 5000
    noisy_episode_n = 5000
    gamma = 0.9
    alpha = 0.2
    total_rewards = QLearning(env,
                              episode_n=episode_n,
                              noisy_episode_n=noisy_episode_n,
                              gamma=gamma,
                              alpha=alpha)
    fig, ax = plt.subplots()
    ax.plot(100 * (np.arange(len(total_rewards)) + 1), total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Q-Learning')

    textstr = '\n'.join((
        f'episode_n={episode_n}',
        f'noisy_episode_n={noisy_episode_n}',
        f'gamma={gamma}',
        f'alpha={alpha}',
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)


    plt.savefig(f'rewards{datetime.datetime.now()}.jpg')
    plt.show()
