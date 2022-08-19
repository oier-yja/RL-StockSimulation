import warnings
from typing import Optional, Union, Tuple
from scipy import stats
import gym
import numpy as np
from gym import spaces
from gym.core import ObsType, ActType


class InvestorEnvironment(gym.Env):

    def __init__(self, initial_money, window):
        self.INITIAL_MONEY = initial_money
        self.WINDOW = window
        self.TAX_PERCENT = 0.03

        self.action_space = spaces.Discrete(41, start=-20)
        warnings.simplefilter("ignore")
        self.observation_space = spaces.Box(
            # window + position + available_money
            low=np.array([0.] * (self.WINDOW + 2)),
            high=np.array([2000.] * (self.WINDOW + 2))
        )

        self.position = 0
        self.available_money = self.INITIAL_MONEY
        self.history_available_money = [self.INITIAL_MONEY] * self.WINDOW
        self.prices = []
        self.state = np.array(self.prices + [self.position, self.available_money])

    def preStep(self, prices):
        self.prices = prices

    def preReset(self, prices):
        self.prices = prices

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        current_price = self.prices[-2]
        future_price = self.prices[-1]
        if action * current_price * (1 + self.TAX_PERCENT) > self.available_money:
            self.history_available_money.append(self.available_money)
            self.history_available_money.pop(0)
            return self.state, self.available_money - action * current_price, False, {"Failed action": 1}

        elif self.position + action < 0:
            self.history_available_money.append(self.available_money)
            self.history_available_money.pop(0)
            return self.state, current_price * (-(self.position + action)), False, {"Failed action": 1}

        else:
            self.position = self.position + action
            if action >= 0:
                self.available_money = self.available_money - action * current_price * (1 + self.TAX_PERCENT)
            else:
                self.available_money = self.available_money - action * current_price * (1 - self.TAX_PERCENT)
            self.state = np.array(self.prices + [self.position, self.available_money], dtype=np.float32)

            # Update history_available_money
            self.history_available_money.append(self.available_money)
            # reward = self.available_money + self.position * future_price - self.INITIAL_MONEY

            # Calculate reward
            x = np.arange(self.WINDOW)
            y = np.arange(self.WINDOW)
            for i in range(self.WINDOW):
                y[i] = self.history_available_money[i+1] - self.history_available_money[i]

            slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
            if int(slope) == 0:
                reward = slope - 2
            else:
                reward = slope
            self.history_available_money.pop(0)
            return self.state, reward, False, {"Successful action": 1}

    def reset(self):
        self.position = 0
        self.available_money = self.INITIAL_MONEY
        self.state = np.array(self.prices + [self.position, self.available_money], dtype=np.float32)
        return self.state

    def render(self, mode="human"):
        pass
