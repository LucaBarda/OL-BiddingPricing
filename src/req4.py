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

        eps = self.T_bidding**(-1/3)
        K = int(1/eps + 1)

        eta = 1/np.sqrt(self.T_bidding)

        min_bid = 0.4
        max_bid = 0.8
        available_bids = np.linspace(min_bid, max_bid, K)

        idx_trut = range(0, num_participants // 3)
        idx_non_trut = range(num_participants // 3, 2 * num_participants // 3)
        idx_ucb = range(2 * num_participants // 3, num_participants)

        bidders = []
        for _ in idx_trut:
            bidders.append(ag.StochasticPacingAgent(self.valuation, self.budget, self.T_bidding, eta))

        for _ in idx_non_trut:
            bidders.append(ag.AdversarialPacingAgent(available_bids, self.valuation, self.budget, self.T_bidding, eta))

        for _ in idx_ucb:
            bidders.append(ag.UCB1BiddingAgent(self.budget, bids=[0 for i in range(K)], T = self.T_bidding, range=1))

        auction = au.FirstPriceAuction(self.ctrs)


        total_wins_types = np.zeros(3)
        total_utility_types = np.zeros(3)
        total_spent_types = np.zeros(3)
        for t in range(self.T_bidding):

            all_bids_t = np.zeros(num_participants)
            for i, bidder in enumerate(bidders):
                all_bids_t[i] = bidder.bid()
            m_t = max(all_bids_t)


            # get winner and payments
            winner, payments_per_click = auction.round(all_bids_t)

            for i, agent in enumerate(bidders):
                has_won = (winner == i)
                f_t = (self.valuation - all_bids_t[i]) * has_won
                c_t = all_bids_t[i] * has_won
                agent.update(f_t, c_t, m_t)
                
                total_wins_types[i//3] += has_won
                total_utility_types[i//3] += f_t
                total_spent_types[i//3] += c_t

            print(f"Auction {t+1}: Winner type: {winner//3}, winning bid {m_t}, Utility: {f_t}, Payment: {c_t}")
        
        # print(f"Total wins: {total_wins}, Total utility: {total_utility}, Total spent: {total_spent}")      
        # 
        # print now the final results: how many wins for each type, total utility and total spent for each type
        print("\n\nFinal results: \n")
        print(f"Total wins for truthful bidders: {total_wins_types[0]}, Total utility: {total_utility_types[0]}, Total spent: {total_spent_types[0]}")
        print(f"Total wins for non-truthful bidders: {total_wins_types[1]}, Total utility: {total_utility_types[1]}, Total spent: {total_spent_types[1]}")
        print(f"Total wins for UCB bidders: {total_wins_types[2]}, Total utility: {total_utility_types[2]}, Total spent: {total_spent_types[2]}")         


    def stochastic(self):
        pass

    def adversarial(self):
        pass 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--valuation", dest="valuation", type=float, default=0.8)
    parser.add_argument("--num_auctions", dest="num_auctions", type=int, default = 100)
    parser.add_argument("--budget", dest="budget", type=float, default=100)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=100)
    parser.add_argument("--num_participants", dest="num_participants", type=int, default=10)
    parser.add_argument("--ctrs", dest = "ctrs", type=list, default = None)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--scenario", dest="scenario", type=str, choices=['solo', 'stochastic', 'adversarial'], default='solo')

    args = parser.parse_args()    

    req = Requirement(args, 100)

    if args.scenario == 'solo':
        print('luha was here')
        req.main()
    elif args.scenario == 'stochastic':
        req.stochastic()
    elif args.scenario == 'adversarial':
        req.adversarial()
    else:
        print("Invalid scenario")
        exit(1)