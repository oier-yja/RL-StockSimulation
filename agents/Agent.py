import abc


class Agent(metaclass = abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, obs_size, action_size, hidden_size):
        pass

    @abc.abstractmethod
    def action(self, obs):
        pass

    @abc.abstractmethod
    def observe(self, obs, reward, done, reset):
        pass
