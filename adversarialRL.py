import matplotlib.pyplot as plt
import numpy as np

from agents.PPO import PPO
from environments.InvestorEnvironment import InvestorEnvironment
from environments.MarketControllerEnvironment import MarketControllerEnvironment


class adversarialRL:
    def __init__(self, investor_num, window, initial_money):
        self.WINDOW = window
        self.INVESTOR_NUM = investor_num
        self.INITIAL_MONEY = initial_money

        # Initialize env
        self.investor_envs = []
        for i in range(self.INVESTOR_NUM):
            self.investor_envs.append(InvestorEnvironment(self.INITIAL_MONEY, self.WINDOW))
        self.market_controller_env = MarketControllerEnvironment(self.WINDOW, self.INVESTOR_NUM, self.INITIAL_MONEY)

        # Initialize agents
        self.investors = []
        for i in range(investor_num):
            self.investors.append(PPO(self.investor_envs[i].observation_space.shape[0], 1, 64))
        self.market_controller = PPO(self.market_controller_env.observation_space.shape[0], 1, 64)

        self.x = []
        self.y = []
        self.available_money_y = []
        self.profits_y = []

    def startGamble(self, days):
        for l in range(100):
            plt.ion()
            plt.title("day-price graph")
            # Reset environment
            window_prices = np.random.uniform(low=0, high=20, size=self.WINDOW)
            window_available_moneys = np.full((self.INVESTOR_NUM, self.WINDOW), self.INITIAL_MONEY)

            prices = window_prices.tolist()
            investor_obs = []
            for investor_env in self.investor_envs:
                investor_env.preReset(prices)
                investor_obs.append(investor_env.reset())
            market_controller_obs = self.market_controller_env.reset()

            # Start gamble
            for i in range(days):

                # Agents take action
                investor_actions = []
                for j in range(self.INVESTOR_NUM):
                    investor_actions.append(self.investors[j].action(investor_obs[j]))
                market_controller_action = self.market_controller.action(market_controller_obs)

                # Add future price to calculate reward
                window_prices = np.append(window_prices, market_controller_obs[-1])
                window_prices = np.delete(window_prices, [0])

                # Step investor environment
                investor_obs.clear()
                investor_reward = 0
                for j in range(self.INVESTOR_NUM):

                    # Environment step
                    self.investor_envs[j].preStep(window_prices.tolist())
                    investor_ob, investor_reward, done, _ = self.investor_envs[j].step(investor_actions[j])
                    investor_obs.append(investor_ob)

                    # Agent observe
                    self.investors[j].observe(investor_ob, investor_reward)

                    # Update available_moneys
                    available_money = investor_ob[-1]
                    window_available_moneys[j] = np.roll(window_available_moneys, -1)[0]
                    window_available_moneys[j][self.WINDOW - 1] = available_money

                # Step market controller environment/
                available_moneys = window_available_moneys.reshape(-1, self.INVESTOR_NUM * self.WINDOW)[0].tolist()
                self.market_controller_env.preStep(available_moneys)
                market_controller_obs, market_controller_reward, done, _ = self.market_controller_env.step(
                    market_controller_action)
                # print(market_controller_reward, _)
                self.market_controller.observe(market_controller_obs, market_controller_reward)

                # Store graph data
                self.x.append(i)
                self.y.append(window_prices[-1])
                self.available_money_y.append(available_moneys[self.WINDOW-1]/100)
                self.profits_y.append(investor_obs[0][-2])
                self.drawPicture()

                if int(available_moneys[self.WINDOW-1]) is 0:
                    plt.close()
                    break

    def getInvestors(self):
        # return self.investors
        pass

    def drawPicture(self):
        plt.clf()
        plt.plot(self.x, self.y, color='blue')
        plt.plot(self.x, self.profits_y, color='brown')
        plt.plot(self.x, self.available_money_y, color='yellow')
        plt.pause(0.001)
        plt.ioff()
        # plt.plot(self.x, self.profits_y, color='brown')


if __name__ == '__main__':
    adversarial_RL = adversarialRL(1, 7, 2000)
    adversarial_RL.startGamble(10000)
    # adversarial_RL.drawPicture()
