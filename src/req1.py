import agents as ag
import auctions as au
import environments as envi
from utils import *

import numpy as np
import matplotlib.pyplot as plt

n_days = 500
users_per_day = [10 for i in range(n_days)] #20 users per day
ctrs = [0.8, 0.5, 0.9, 1] # company + competitors
lambdas = [1, 0.9] # 2 slots
budget = 1000
product_cost = 0.1
valuation = 1
ucb_bidding_agent = True  #if true then the bidding agent is ucb-like, else it is multiplicative pacing

n_trials = 10


problem_params = {"n_days" : n_days, "users_per_day" : users_per_day, "ctrs" : ctrs, "lambdas" : lambdas,
                  "ucb_bidding_agent" : ucb_bidding_agent, "budget" : budget, "product_cost" : product_cost, "valuation" : valuation}
if __name__ == '__main__':
    # Initialize the agents
    agent1 = ag.MultiplicativePacingAgent(1, 1, 1, 1)
    agent2 = ag.UCB1BiddingAgent(10, [.5, .6], 100, 1)
    agent3 = ag.GPUCBAgent(100)
    # Run the agents
    agent1.bid()
    arm = agent2.pull_arm()
    agent1.update(1, 1)
    agent2.update(1, 1)
    print(arm, 'Done')

    auction1 = au.SecondPriceAuction([.5, .6])

    env1 = envi.StochasticBiddingCompetitors(1,1)
    set_seed(1)