import gym
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt


class CrossEntropyAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_layers_dim),
            nn.ReLU(),
            nn.Linear(hidden_layers_dim, action_dim),
        )
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, input):
        return self.network(input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.forward(state)
        action_prob = self.softmax(action).detach().numpy()
        action = np.random.choice(len(action_prob), p=action_prob)
        return action

    def update_policy(self, elite_sessions):
        elite_states, elite_actions = [], []
        for session in elite_sessions:
            elite_states.extend(session['states'])
            elite_actions.extend(session['actions'])

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        loss = self.loss(self.network(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


EPSILON = 1e-3


def get_session(env, agent, session_len, visual=False):
    session = {}
    states, actions = [], []
    total_reward = 0

    state = env.reset()
    for sesh in range(session_len):
        states.append(state)
        action = agent.get_action(state)
        actions.append(action)

        if visual:
            env.render()

        state, reward, done, _ = env.step(action)
        reward += 1
        total_reward += reward

        position, speed = state
        if abs(position - 0.5) < EPSILON:
            print(position, speed, reward, 'GOAL REACHED at step', sesh)
            if speed - 0.001 < EPSILON:
                total_reward += 1
                break
            else:
                print('WAS TOO FAST')
            total_reward += 1
            break

        # todo uncomment if you want 200 step max session length
        # if done:
        #     break

    session['states'] = states
    session['actions'] = actions
    session['total_reward'] = total_reward
    return session


def get_elite_sessions(sessions, q_param):
    total_rewards = np.array([session['total_reward'] for session in sessions])
    quantile = np.quantile(total_rewards, q_param)

    elite_sessions = []
    for session in sessions:
        if session['total_reward'] > quantile or session['total_reward'] > 0:
            elite_sessions.append(session)

    return elite_sessions


def learn(episode_n, env, agent, session_len, session_n, q_param,
          visual=False):
    rewards = []
    for episode in range(episode_n):
        sessions = [get_session(env, agent, session_len) for _ in
                    range(session_n)]
        mean_total_reward = np.mean(
            [session['total_reward'] for session in sessions])

        elite_sessions = get_elite_sessions(sessions, q_param)
        if len(elite_sessions) > 0:
            agent.update_policy(elite_sessions)

        if visual:
            print(f'Episode: {episode + 1}')
            print(f'\tReward mean: {mean_total_reward}')
            print(
                f'\tElite sessions ({len(elite_sessions)}/{len(sessions)}) quantile={q_param}')

        rewards.append(mean_total_reward)
    return rewards


env = gym.make("MountainCar-v0")
agent = CrossEntropyAgent(env.observation_space.shape[0],
                          env.action_space.n,
                          hidden_layers_dim=100)

total_epochs = 20
session_n = 1
session_len = 8000
q_param = 0.1

rewards = learn(total_epochs, env, agent, session_len, session_n, q_param,
                visual=True)

last_sesh = get_session(env, agent, session_len, visual=True)
print(f'Last session:'
      f'\n\t{len(last_sesh["actions"])} moves'
      f'\n\t{last_sesh["total_reward"]} score')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(rewards) + 1), rewards)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

get_session(env, agent, session_len, visual=True)
