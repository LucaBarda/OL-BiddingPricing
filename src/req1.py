import argparse
import numpy as np
import random

import agents as ag
import environments as envi
from utils import *

class Requirement1:
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
        
        item_cost = 10
        min_price = item_cost
        max_price = 20 # price at which the conversion probability is 0
        n_customers = 100

        eps = self.T_pricing**(-1/3)
        K = int(1/eps + 1)
        K = 10000

        discr_prices = np.linspace(min_price, max_price, K)

        conversion_probability = lambda p: 1 - p/max_price
        # such that the probability of conversion is 1 at price = 0 and 0 at price = max_price

        reward_function = lambda price, n_sales: (price - item_cost) * n_sales
        # the maximum possible profit is selling at the maximum price to all customers
        max_reward = reward_function(max_price, n_customers) 

        agent = ag.GPUCBAgent(T = self.T_pricing, discretization = K)
        envir = envi.StochasticPricingEnvironment(conversion_probability, item_cost)

        print(f"Max Reward: {max_reward}, Discretized Prices: {discr_prices}, K: {K}")

        for t in range(self.T_pricing): 
            # GP agent chooses price
            price_t = agent.pull_arm()
            # rescale price from [0,1] to [min_price, max_price]
            price_t = denormalize_zero_one(price_t, min_price, max_price)

            # get demand and reward from pricing environment
            d_t, r_t = envir.round(price_t, n_customers)
            # reward = total profit

            # update agent with profit per customer
            agent.update(r_t/n_customers)

            print(f"Day {t+1}: Price: {price_t}, Demand: {d_t}, Reward: {r_t}")

        

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_days", dest="num_days", type=int, default=365)
    parser.add_argument("--auctions_per_day", dest="auctions_per_day", type=int, default = 10)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=100)
    parser.add_argument("--num_competitors", dest="num_competitors", type=int, default=10)
    parser.add_argument("--ctrs", dest = "ctrs", type=list, default = None)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--run_type", dest="run_type", type=str, choices=['main', 'bidding', 'pricing'], default='pricing')

    #for pricing only
    parser.add_argument("--num_buyers", dest="num_buyers", type = int, default = 100)

    args = parser.parse_args()    

    set_seed(args.seed)

    req = Requirement1(args, 100)

    if args.run_type == 'main':
        req.main()
    elif args.run_type == 'bidding':
        req.bidding()
    elif args.run_type == 'pricing':
        req.pricing()
    else:
        print("Invalid run type")
        exit(1)
    