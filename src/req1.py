import agents.MultiplicativePacingAgent as mpa
import agents.UCB1BiddingAgent as ucb1
import numpy as np

if __name__ == 'main':
    # Initialize the agents
    agent1 = mpa.MultiplicativePacingAgent(1, 1, 1, 1)
    agent2 = ucb1.UCB1BiddingAgent(1, 1, 1, 1)
    # Run the agents
    # agent1.bid()
    # agent2.bid()
    # agent1.update(1, 1)
    # agent2.update(1, 1)
    print('Done')