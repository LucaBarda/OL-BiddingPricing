import argparse
import numpy as np
import random

import agents as ag
import environments as envi
import auctions as au
from utils import *

import warnings
"""
We have 3 interpretations of the requirement:
[scenario_1]. There are 3 groups of bidders, each of n_bidders // 3 size. 
    a. primal dual truthful bidders
    b. primal dual non-truthful bidders
    c. UCB bidders
[scenario_2]. There are 3 bidders of types a, b, c and n-3 stochastic bidders
[scenario_3]. There are 3 bidders of types a, b, c and n-3 adversarial bidders
"""
class Requirement:
    def __init__(self, args, n_iters):
        self.args = args
        #extract all args in members
        for key, value in vars(args).items():
            setattr(self, key, value)

        if self.ctrs is None:
            self.ctrs = np.random.uniform(0.4, 0.9, self.num_participants)
        else:
            assert len(self.ctrs) == self.num_participants, "Number of CTRs must match number of bidders"

        self.T_bidding = self.num_auctions


    def main(self):
        # report = PDFReport("prova.pdf", 4)

        num_participants = self.num_participants
        if num_participants % 3 != 0:
            warnings.warn(f"Number of competitors must be divisible by 3, decreasing num_competitors to reach divisibility by 3: reaching number {num_participants - num_participants % 3}")
            num_participants -= num_participants % 3
            self.ctrs = self.ctrs[:num_participants]

        if self.valuation is None:
            valuations = np.random.uniform(0.7, 0.8, num_participants)
        else:
            valuations = np.ones(num_participants) * self.valuation #same valuation for all bidders


        eps = self.T_bidding**(-1/3)
        K = int(1/eps + 1)

        eta = 1/np.sqrt(self.T_bidding)

        min_bid = 0
        #max bis are the valuation of the bidders minus an epsilon
        max_bids = valuations - 0.05

        #also available bids changes depending on what valuation the bidder has
        available_bids = np.zeros(shape = (num_participants, K))
        for i in range(num_participants):
            available_bids[i] = np.linspace(min_bid, max_bids[i], K)

        idx_trut = range(0, num_participants // 3)
        idx_non_trut = range(num_participants // 3, 2 * num_participants // 3)
        idx_ucb = range(2 * num_participants // 3, num_participants)

        bidders = []
        for i in idx_trut:
            bidders.append(ag.StochasticPacingAgent(valuations[i], self.budget, self.T_bidding, eta))
        for j in idx_non_trut:
            valuation = np.random.uniform(0.4, 0.95)

            bidders.append(ag.AdversarialPacingAgent(available_bids[j], valuations[j], self.budget, self.T_bidding, eta))
        for z in idx_ucb:
            bidders.append(ag.UCB1BiddingAgent(self.budget, bids=[0 for i in range(K)], T = self.T_bidding, range=1))

        auction = au.FirstPriceAuction(self.ctrs)

        '''LOGGING'''
        all_bids = np.zeros((num_participants, self.T_bidding))
        m_ts = np.zeros((num_participants, self.T_bidding))
        my_utilities = np.zeros(shape = (num_participants, self.T_bidding))
        total_wins_types = np.zeros(3)
        total_utility_types = np.zeros(3)
        total_spent_types = np.zeros(3)

        for t in range(self.T_bidding):

            all_bids_t = np.zeros(num_participants)
            for i, bidder in enumerate(bidders):
                if i < num_participants // 3: #truthful bidders: I get the bid to the closes of the available bids otherwise impossible to construct regret 
                    bid = bidder.bid()
                    all_bids_t[i] = available_bids[i][np.abs(available_bids[i] - bid).argmin()]
                else:
                    all_bids_t[i] = bidder.bid()
            all_bids[:, t] = all_bids_t

            m_t_1 = max(all_bids_t)
            #now take the second highest bid
            m_t_2 = all_bids_t[np.argsort(all_bids_t)[-2]]


            # get winner and payments
            winner, _ = auction.round(all_bids_t)


            for i, agent in enumerate(bidders):
                #m_t is actually max bid of the opponents
                if all_bids_t[i] == m_t_1:
                    m_t = copy.deepcopy(m_t_2)
                else:
                    m_t = copy.deepcopy(m_t_1)
                

                has_won = (winner == i)
                f_t = (valuations[i] - all_bids_t[i]) * has_won
                c_t = all_bids_t[i] * has_won
                agent.update(f_t, c_t, m_t)
                
                total_wins_types[i//(num_participants//3)] += has_won
                total_utility_types[i//(num_participants//3)] += f_t
                total_spent_types[i//(num_participants//3)] += c_t

                m_ts[i, t] = m_t
                my_utilities[i, t] = f_t
            
            

            print(f"Auction {t+1}: Winner type: {winner//3}, winning bid {m_t_1}, Utility: {f_t}, Payment: {c_t}")
        print("\n\nFinal results: \n")
        print(f"Total wins for truthful bidders: {total_wins_types[0]}, Total utility: {total_utility_types[0]}, Total spent on average: {total_spent_types[0]/(num_participants//3)}")
        print(f"Total wins for non-truthful bidders: {total_wins_types[1]}, Total utility: {total_utility_types[1]}, Total spent on average: {total_spent_types[1]/(num_participants//3)}")
        print(f"Total wins for UCB bidders: {total_wins_types[2]}, Total utility: {total_utility_types[2]}, Total spent on average: {total_spent_types[2]/(num_participants//3)}")         


        ''' ADVERSARIAL CLAIRVOYANT '''
        clairvoyant_utilities = np.zeros((num_participants, self.T_bidding))
        for i in range(num_participants):
            _, clairvoyant_utilities[i], _ = get_clairvoyant_non_truthful_adversarial(self.budget, valuations[i], self.T_bidding, available_bids[i], all_bids, auction_agent=auction, idx_agent=i)
        
        #now average the utilities for each of the 3 types of bidders
        clairvoyant_utilities_types = np.zeros((3, self.T_bidding))
        my_utilities_types = np.zeros((3, self.T_bidding))
        for i in range(self.T_bidding):
            #for each time t I shoulde have the average utility of the 3 types of bidders
            clairvoyant_utilities_types[0, i] = np.mean(clairvoyant_utilities[idx_trut, i])
            clairvoyant_utilities_types[1, i] = np.mean(clairvoyant_utilities[idx_non_trut, i])
            clairvoyant_utilities_types[2, i] = np.mean(clairvoyant_utilities[idx_ucb, i])
            #same for my_utilities
            my_utilities_types[0, i] = np.mean(my_utilities[idx_trut, i])
            my_utilities_types[1, i] = np.mean(my_utilities[idx_non_trut, i])
            my_utilities_types[2, i] = np.mean(my_utilities[idx_ucb, i])
        
        regret_per_trial_bidding_nont = []
        regret_per_trial_bidding_t = []
        regret_per_trial_bidding_ucb = []
        regret_per_trial_bidding_t.append(np.cumsum(clairvoyant_utilities_types[0] - my_utilities_types[0]))
        regret_per_trial_bidding_nont.append(np.cumsum(clairvoyant_utilities_types[1] - my_utilities_types[1]))
        regret_per_trial_bidding_ucb.append(np.cumsum(clairvoyant_utilities_types[2] - my_utilities_types[2]))

        plt.plot(np.arange(self.T_bidding), regret_per_trial_bidding_nont[0], label='Average Regret Bidding')
        plt.title('Cumulative regret of bidding')
        # plt.fill_between(np.arange(self.T_bidding),
        #                 average_regret_bidding-regret_sd_bidding/np.sqrt(self.n_iters),
        #                 average_regret_bidding+regret_sd_bidding/np.sqrt(self.n_iters),
        #                 alpha=0.3,
        #                 label='Uncertainty')
        #plt.plot((0,T-1), (average_regret_bidding[0], average_regret_bidding[-1]), 'ro', linestyle="--")
        plt.xlabel('$t$')
        plt.legend()
        plt.savefig('just_bidding_regret.png')        


        # regret_per_trial_bidding.append(np.cumsum(clairvoyant_utilities - my_utilities))        

    def stochastic(self):
        pass

    def adversarial(self):
        pass 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--valuation", dest="valuation", type=float, default=None)
    parser.add_argument("--num_auctions", dest="num_auctions", type=int, default = 1000)
    parser.add_argument("--budget", dest="budget", type=float, default=100)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=100)
    parser.add_argument("--num_participants", dest="num_participants", type=int, default=10)
    parser.add_argument("--ctrs", dest = "ctrs", type=list, default = None)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--scenario", dest="scenario", type=str, choices=['solo', 'stochastic', 'adversarial'], default='solo')

    args = parser.parse_args()    

    req = Requirement(args, 100)

    if args.scenario == 'solo':
        req.main()
    elif args.scenario == 'stochastic':
        req.stochastic()
    elif args.scenario == 'adversarial':
        req.adversarial()
    else:
        print("Invalid scenario")
        exit(1)