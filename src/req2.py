import argparse
import numpy as np
import random
import agents as ag
from utils import set_seed, generate_adv_conv_prob_sequence
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
        cost_per_good = 0.1

        eps = self.T_pricing**(-1/3)
        min_price, max_price = 0, 1
        discr_prices = np.linspace(min_price, max_price+eps, eps)

        K = len(discr_prices)
        hedge_lr = np.sqrt(np.log(K) / self.T_pricing)
        hedge_ag = ag.HedgeAgent(K, hedge_lr)

        conv_probs = generate_adv_conv_prob_sequence(self.T_pricing)
        demand = conv_probs * num_buyers
        reward_func = lambda price, demand: demand * (price - cost_per_good)


        for t in range(self.T_pricing): 
            #pull arm
            arm = hedge_ag.pull_arm()
            #get price
            price_t = discr_prices[arm]

            #full-feedback: compute reward for each possible price
            rewards = demand[t] * (discr_prices - cost_per_good)
            reward_t = rewards[arm]
            #normalize reward between 0 and 1ù
            rewards = (rewards - (min_price - cost_per_good)*num_buyers) / (max_price*num_buyers - min_price*num_buyers)
            losses = 1 - rewards
            #update
            hedge_ag.update(losses)
        


        

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_days", dest="num_days", type=int, default=90)
    parser.add_argument("--auctions_per_day", dest="auctions_per_day", type=int, default = 10)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=100)
    parser.add_argument("--num_competitors", dest="num_competitors", type=int, default=10)
    parser.add_argument("--ctrs", dest = "ctrs", type=list, default = None)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--run_type", dest="run_type", type=str, choices=['main', 'bidding', 'pricing'], default='main')

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
    