class PolicyUpdater:
    def __init__(self, param):
        pass

    def update(self, elite_sessions, agent, new_policy):
        pass


class PolicySmoother(PolicyUpdater):
    def __init__(self, param):
        if 0 < param <= 1:
            self.lambda_param = param
        else:
            raise ValueError(f'Lambda for policy smoothing must be in (0, '
                             f'1] but was {param}')

    def update(self, elite_sessions, agent, new_policy):
        for state in range(agent.state_n):
            if sum(new_policy[state]) == 0:
                new_policy[state] += 1 / agent.action_n
            else:
                new_policy[state] /= sum(new_policy[state])
        agent.policy = self.lambda_param * new_policy + \
                       (1 - self.lambda_param) * agent.policy


class LaplaceSmoother(PolicyUpdater):
    def __init__(self, param):
        if 0 < param:
            self.lambda_param = param
        else:
            raise ValueError(f'Lambda for Laplace smoothing must be > 0 but '
                             f'was {param}')

    def update(self, elite_sessions, agent, new_policy):
        for state in range(agent.state_n):
            new_policy[state] = (new_policy[state] + self.lambda_param) / (
                    sum(new_policy[state])
                    + self.lambda_param * agent.action_n)
        agent.policy = new_policy
