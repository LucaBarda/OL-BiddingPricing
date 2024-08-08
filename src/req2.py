import argparse
import numpy as np
import random

import agents as ag
import environments as envi
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
            self.ctrs = np.random.uniform(0.4, 0.9, self.num_competitors)
        else:
            assert len(self.ctrs) == self.num_competitors, "Number of CTRs must match number of competitors"

        self.T_bidding = np.sum(self.auctions_per_day)

    def main(self):
        pass

    def bidding(self):
        num_bidders = 10
        num_competitors = 10 - 1
        
        #In this case we are just considering bidding so no need to separete for the different days.
        #we sum for all days
        total_auctions = sum(self.auctions_per_day)
        #competitors randomly pick 1 from self.competitors_per_day
        competitors = random.choice(self.competitors_per_day)


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
    parser.add_argument("--run_type", dest="run_type", type=str, choices=['main', 'bidding', 'pricing'], default='pricing')

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
    