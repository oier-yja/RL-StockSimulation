from agents.Agent import Agent


class SAC(Agent):
    def __init__(self, obs_size, action_size, hidden_size):
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size

    def action(self, obs):
        pass

    def observe(self, obs, reward, done, reset):
        pass