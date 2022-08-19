from scipy import stats
import random
import warnings
from typing import Optional, Union, Tuple

import gym
import numpy as np
from gym.core import ObsType, ActType
from gym.vector.utils import spaces


class MarketControllerEnvironment(gym.Env):

    def __init__(self, window, investor_num, initial_money):
        self.WINDOW = window
        self.INVESTOR_NUM = investor_num
        self.INITIAL_MONEY = initial_money

        self.action_space = spaces.Discrete(41, start=-20)
        warnings.simplefilter("ignore")
        self.observation_space = spaces.Box(
            # window * investor_num + current_price
            low=np.array([0.] * (self.WINDOW * self.INVESTOR_NUM + 1)),
            high=np.array([2000.] * (self.WINDOW * self.INVESTOR_NUM + 1))
        )

        self.future_price = 0.
        self.available_moneys = [self.INITIAL_MONEY] * (self.WINDOW * self.INVESTOR_NUM)
        self.state = np.array(self.available_moneys + [self.future_price])

    def preStep(self, available_moneys):
        self.available_moneys = available_moneys

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        if action + self.future_price < 0:
            return self.state, action + self.future_price, False, {"Failed action": action}
        else:
            # Change future price
            self.future_price = self.future_price + action
            self.state = np.array(self.available_moneys + [self.future_price], dtype=np.float32)
            # Calculate reward
            # reward = 0
            # for i in range(self.WINDOW - 1, len(self.available_moneys), self.WINDOW):
            #     reward = reward - (self.available_moneys[i] - self.available_moneys[i-1])
            reward = 0
            x = np.arange(self.WINDOW)
            for i in range(self.WINDOW - 1, len(self.available_moneys), self.WINDOW):
                y = np.array(self.available_moneys[i+1-self.WINDOW:i+1])
                slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
                if int(slope) >= 0:
                    reward = reward - 2 - slope ** 3
                else:
                    reward = reward + 2 - slope ** 3

            return self.state, reward, False, {"Successful action": 1}

    def reset(self):
        self.future_price = random.uniform(1, 20)
        self.available_moneys = [self.INITIAL_MONEY] * (self.WINDOW * self.INVESTOR_NUM)
        self.state = np.array(self.available_moneys + [self.future_price], np.float32)
        return self.state

    def render(self, mode="human"):
        pass