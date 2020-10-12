from enum import Enum
import numpy as np
import gym
from policy_updaters import PolicyUpdater, PolicySmoother, LaplaceSmoother


class Agent:
    def __init__(self, state_n, action_n, policy_updater: PolicyUpdater):
        self.state_n = state_n
        self.action_n = action_n

        self.policy_updater = policy_updater
        self.policy = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        prob = self.policy[state]
        action = np.random.choice(np.arange(self.action_n), p=prob)
        return int(action)

    def update_policy(self, elite_sessions):
        new_policy = np.zeros((self.state_n, self.action_n))
        for session in elite_sessions:
            for state, action in zip(session['states'], session['actions']):
                new_policy[state][action] += 1
        self.policy_updater.update(elite_sessions, self, new_policy)


def get_state(obs):
    return obs


def get_session(env, agent, session_len, visual=False):
    session = {}
    states, actions = [], []
    total_reward = 0

    obs = env.reset()
    for _ in range(session_len):
        state = get_state(obs)
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    session['states'] = states
    session['actions'] = actions
    session['total_reward'] = total_reward
    return session


def get_elite_sessions(sessions, q_param):
    total_rewards = np.array([session['total_reward'] for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session['total_reward'] > quantile:
            elite_sessions.append(session)

    return elite_sessions


def learn(episode_n, env, agent, session_len, session_n, q_param, visual=False):
    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len) for _ in range(session_n)]
        mean_total_reward = np.mean([session['total_reward'] for session in sessions])

        elite_sessions = get_elite_sessions(sessions, q_param)
        if len(elite_sessions) > 0:
            agent.update_policy(elite_sessions)

        if visual:
            print(f'Episode: {episode + 1}')
            print(f'\tReward mean: {mean_total_reward}')
            print(f'\tElite sessions ({len(elite_sessions)})')


def main():
    env = gym.make("Taxi-v3")
    policy_smoother = PolicySmoother(0.1111)
    laplace_smother = LaplaceSmoother(0.5)
    agent = Agent(env.observation_space.n, env.action_space.n, policy_smoother)

    total_epochs = 50
    total_sessions = 200
    max_steps = 30
    q_param = 0.777

    learn(total_epochs, env, agent, max_steps, total_sessions, q_param, visual=True)

    last_sesh = get_session(env, agent, max_steps, visual=True)
    print(f'Last session:'
          f'\n\t{len(last_sesh["actions"])} moves'
          f'\n\t{last_sesh["total_reward"]} score')

    env.close()


if __name__ == "__main__":
    main()
