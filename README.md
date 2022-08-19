# RL-StockSimulation
An idea about using reinforcement learning to simulate stock market

## Quickly start
1. Download code and install libraries

2. Open adversarialRL.py and start run

3. Get more information in source code and annotation

---

## Operational process
### Basic idea
I propose a hypothesis that in a given market, witch contains two major players. 
One is market controller and another one is investor.

The goal of market controller is to change the future prices to minimize the available money of investors, 
so that market controller can get money from investors.

On the contrast, the goal of investor is to maximize their available money, by taking actions, 
such as buy stocks or sell stocks.

Although the hole model was not rigorous tested, the result and the interaction was quite interesting. 
The graph of prices and other data just like a real stock market. 
Maybe this idea can help you come up with better model.

Any constructive suggestion or bug feedback are appreciated. 

:)
### General process
Object: MarketController, Investor, MarketControllerEnvironment, and InvestorEnvironment are created at initialization.
The whole process was divided to every signal days. In each day, 
MarketController change the future prices based on the history available money of investor, 
and Investor take an action to buy or sell a certain amount of stocks based on the history prices of this stock.

Then, MarketControllerEnvironment and InvestorEnvironment feedback the MarketController's action and Investor's action, 
enabling MarketController rejudge whether price changing is good or not, 
and making Investor know the action is correct or wrong.

Finally, relative data like Investor's available money, position, and history prices were updated. 
After that, the next day is coming and repeat the whole process.
### Other words
The whole process was based on the guidance of pfrl. If you can not fully understand what I said above, 
you can also to go to the [pfrl](https://github.com/pfnet/pfrl) to have a look.

---

## Libraries
python 3.7.0

gym 0.22.0

matplotlib 3.2.2

pytorch 1.11.0

pfrl 0.3.0

---
Any submission issue or discussion about this model are welcomed.