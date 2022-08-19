import numpy as np
import pfrl
import torch
from torch import nn

from agents.Agent import Agent


class PPO(Agent):
    def __init__(self, obs_size, action_size, hidden_size):
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        policy = torch.nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )
        vf = torch.nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        model = pfrl.nn.Branched(policy, vf)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        self.agent = pfrl.agents.PPO(
            model=model,
            optimizer=optimizer,
            gpu=0,
        )

    def action(self, obs):
        action = self.agent.act(obs.astype(np.float32))
        return action

    def observe(self, obs, reward, done=False, reset=False):
        self.agent.observe(obs.astype(np.float32), reward, done, reset)
