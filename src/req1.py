import agents as ag
import auctions as au
import environments as envi

import numpy as np
import matplotlib.pyplot as plt

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
