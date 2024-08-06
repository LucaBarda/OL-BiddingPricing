import argparse
import numpy as np
import random

class Requirement:
    def __init__(self, args, n_iters):
        self.args = args

        #general members
        self.n_iters = n_iters
        self.num_days = 90 # 3 months

        #bidding members
        self.auctions_per_day = [10 for _ in range(self.num_days)] #since it is 1 slot auction, 1 bid equals 1 user 
        self.auctions_per_day = [int(i + np.random.normal(0, 5)) for i in self.auctions_per_day] #add noise 

        self.competitors_per_day = [100 for _ in range(self.num_days)] 
        self.competitors_per_day = [int(i + np.random.normal(0, 10)) for i in self.competitors_per_day] #add noise



    
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
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_days", dest="num_days", type=int, default=90)
    parser.add_argument("--auctions_per_day", dest="auctions_per_day", type=int, default = 10)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=100)
    parser.add_argument("--run_type", dest="run_type", type=str, choices=['main', 'bidding', 'pricing'], default=1000)
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
    