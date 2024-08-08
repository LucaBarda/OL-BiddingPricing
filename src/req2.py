import argparse
import numpy as np
import random

import agents as ag
import environments as envi
import auctions as au
from utils import *
#set seed in numpy

class Requirement:
    def __init__(self, args, n_iters):
        self.args = args
        #extract all args in members
        for key, value in vars(args).items():
            setattr(self, key, value)

        #pricing members
        self.T_pricing = self.num_days



        #bidding members
        self.auctions_per_day = [self.auctions_per_day for _ in range(self.num_days)] #since it is 1 slot auction, 1 bid equals 1 user 
        self.auctions_per_day = [int(i + np.random.uniform(-5, 5)) for i in self.auctions_per_day] #add noise 

        self.competitors_per_day = [100 for _ in range(self.num_days)]

        if self.ctrs is None:
            self.ctrs = np.random.uniform(0.4, 0.9, self.num_competitors+1)
        else:
            assert len(self.ctrs) == self.num_competitors+1, "Number of CTRs must match number of bidders"

        self.T_bidding = np.sum(self.auctions_per_day)

    def main(self):
        pass

    def bidding(self):
        num_competitors = self.num_competitors
        budget = 100

        K = 100
        min_bid = 0
        max_bid = 1
        available_bids = np.linspace(min_bid, max_bid, K)

        # in this case we are just considering bidding so no need to separate for the different days.
        n_auctions = sum(self.auctions_per_day)

        # learning rate from theory
        eta = 1/np.sqrt(n_auctions)
        
        my_ctr = self.ctrs[0]
        other_ctrs = self.ctrs[1:]
        my_valuation = 0.8
        
        #In this case we are just considering bidding so no need to separete for the different days.
        total_auctions = sum(self.auctions_per_day)

        other_bids = np.random.uniform(0, 1, size=(num_competitors, total_auctions))
        # matrix of bids for each competitor in each auction

        agent = ag.AdversarialPacingAgent(available_bids, my_valuation, budget, total_auctions, eta)
        envir = envi.AdversarialBiddingCompetitors(other_bids, num_competitors, total_auctions)
        auction = au.SecondPriceAuction(self.ctrs)

        utilities = np.array([])
        my_bids = np.array([])
        my_payments = np.array([])
        total_wins = 0

        for t in range(total_auctions):
            # agent chooses bid
            bid_t = agent.bid()
            # get bids from other competitors
            other_bids_t = envir.round()
            m_t = other_bids_t.max()

            # get winner and payments
            bids = np.append(bid_t, other_bids_t)
            winner, payments_per_click = auction.round(bids)
            my_win = (winner == 0)

            f_t = (my_valuation - bid_t) * my_win
            c_t = bid_t * my_win
            # update agent with full feedback (m_t)
            agent.update(f_t, c_t, m_t)

            print(f"Auction {t+1}: Bid: {bid_t}, Utility: {f_t}, Payment: {c_t}, Winner: {winner}")


    def pricing(self):
        
        num_buyers = self.num_buyers
        item_cost = 0.1
        min_price = item_cost
        max_price = 1 # price at which the conversion probability is 0

        eps = self.T_pricing**(-1/3)
        K = int(1/eps + 1)
        
        discr_prices = np.linspace(min_price, max_price, K)

        hedge_lr = np.sqrt(np.log(K) / self.T_pricing)
        hedge_ag = ag.HedgeAgent(K, hedge_lr)

        conversion_probability = lambda p, theta: (1 - p)**theta

        theta_seq = generate_adv_sequence(self.T_pricing, 0.5, 2)
        envir = envi.AdversarialPricingEnvironment(conversion_probability, theta_seq, item_cost)

        # demand = conversion_probability * num_buyers
        # reward_func = lambda price, demand: demand * (price - item_cost)

        for t in range(self.T_pricing):
            #pull arm
            arm_t = hedge_ag.pull_arm()
            #get price
            price_t = discr_prices[arm_t]
            
            losses_t = np.array([])
            #full-feedback: compute losses for each possible price
            for price in discr_prices:
                d_t, r_t = envir.round(price, num_buyers, theta_seq[t])
                # vector of normalized losses
                losses_t = np.append(losses_t, 1 - r_t/num_buyers)

            #update
            hedge_ag.update(losses_t)

            print(f"Day {t+1}: Price: {price_t}, Losses: {losses_t}, Theta: {theta_seq[t]}")
        


        

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_days", dest="num_days", type=int, default=90)
    parser.add_argument("--auctions_per_day", dest="auctions_per_day", type=int, default = 10)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=100)
    parser.add_argument("--num_competitors", dest="num_competitors", type=int, default=10)
    parser.add_argument("--ctrs", dest = "ctrs", type=list, default = None)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--run_type", dest="run_type", type=str, choices=['main', 'bidding', 'pricing'], default='bidding')

    #for pricing only
    parser.add_argument("--num_buyers", dest="num_buyers", type = int, default = 100)

    args = parser.parse_args()    

    req = Requirement(args, 100)

    if args.run_type == 'main':
        req.main()
    elif args.run_type == 'bidding':
        req.bidding()
    elif args.run_type == 'pricing':
        req.pricing()
    else:
        print("Invalid run type")
        exit(1)
    