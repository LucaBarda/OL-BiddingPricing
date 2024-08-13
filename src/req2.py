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
        report = PDFReport("prova.pdf", self.requirement)
 
        item_cost = 0.1
        min_price = item_cost
        max_price = 1

        # a round of pricing for each day
        T_pricing = self.num_days
        eps = T_pricing ** (-1 / 3)
        K = int(1/eps + 1)

        discr_prices = np.linspace(min_price, max_price, K)
        learning_rate = np.sqrt(np.log(K) / T_pricing)
        pricing_agent = ag.HedgeAgent(K, learning_rate)

        # parametric conversion probability        
        conversion_probability = lambda p, theta: (1 - p) ** theta
        theta_seq = generate_adv_sequence(T_pricing, 0.5, 2)
        pricing_envir = envi.AdversarialPricingFullEnvironment(conversion_probability, theta_seq, item_cost)

        num_competitors = self.num_competitors
        budget = 4000

        min_bid = 0
        max_bid = 1
        available_bids = np.linspace(min_bid, max_bid, K)

        T_bidding = np.sum(self.auctions_per_day)       
        eta = 1 / np.sqrt(T_bidding)
        my_ctr = self.ctrs[0]
        my_valuation = 0.8

        other_bids = np.random.uniform(0.4, 0.7, size=(num_competitors, T_bidding))
        bidding_agent = ag.AdversarialPacingAgent(available_bids, my_valuation, budget, T_bidding, eta)
        bidding_envir = envi.AdversarialBiddingCompetitors(other_bids, num_competitors, T_bidding)
        auction = au.FirstPriceAuction(self.ctrs)

        total_sales = 0
        total_profit = 0
        
        total_wins = 0
        total_utility = 0
        total_spent = 0

        for t in range(self.num_days):
            ### Pricing phase: setting the price
            arm_t = pricing_agent.pull_arm()
            price_t = discr_prices[arm_t]

            day_wins = 0
            n_clicks = 0
            ### Bidding phase: each auction is a user connecting to the site
            for auction_index in range(self.auctions_per_day[t]):
                
                bid_t = bidding_agent.bid()
                other_bids_t = bidding_envir.round()
                m_t = other_bids_t.max()
                bids = np.append(bid_t, other_bids_t)

                winner, payment_per_click = auction.round(bids)

                my_win = 0
                if winner == 0: # auction won
                    my_win = 1
                    day_wins += 1

                    user_clicked = np.random.binomial(1, self.ctrs[0])
                    n_clicks += user_clicked

                # utility and cost for the bidding agent are computed in expectation                
                f_t = (my_valuation - my_ctr * payment_per_click) * my_win
                c_t = my_ctr * payment_per_click * my_win
                bidding_agent.update(f_t, c_t, m_t)

                total_utility += f_t
                total_spent += c_t

            ### Pricing phase: updating the price
            # get full feedback from environment
            d_t, r_t = pricing_envir.round(discr_prices, n_clicks)
            # compute losses with normalized reward
            losses_t = 1 - r_t/n_clicks if n_clicks > 0 else np.ones(K)
            # update pricing agent
            pricing_agent.update(losses_t)

            # update sales and profit on the played price
            day_sales = d_t[arm_t]
            day_profit = r_t[arm_t]

            total_wins += day_wins
            total_sales += day_sales
            total_profit += day_profit

            print(f"Day {t+1}: Price: {price_t}, Day wins: {day_wins}, N.clicks: {n_clicks}, Day Sales: {day_sales}, Day Profit: {day_profit}")

        print(f"Total wins: {total_wins}, Total utility: {total_utility}, Total spent: {total_spent}, Total sales: {total_sales}, Total profit: {total_profit}")

    def bidding(self):
        num_competitors = self.num_competitors
        budget = 400
        


        eps = self.T_bidding**(-1/3)
        K = int(1/eps + 1)

        min_bid = 0.4
        max_bid = 0.8
        available_bids = np.linspace(min_bid, max_bid, K)

        # in this case we are just considering bidding so no need to separate for the different days.
        n_auctions = sum(self.auctions_per_day)

        # learning rate from theory
        eta = 1/np.sqrt(n_auctions)
        
        my_ctr = 0.9
        other_ctrs = self.ctrs[1:]
        my_valuation = 0.8
        
        #In this case we are just considering bidding so no need to separete for the different days.
        total_auctions = sum(self.auctions_per_day)

        other_bids = np.random.uniform(0.4, 0.6, size=(num_competitors, total_auctions))
        # matrix of bids for each competitor in each auction

        agent = ag.AdversarialPacingAgent(available_bids, my_valuation, budget, total_auctions, eta)
        envir = envi.AdversarialBiddingCompetitors(other_bids, num_competitors, total_auctions)
        auction = au.FirstPriceAuction(self.ctrs)

        utilities = np.array([])
        my_bids = np.array([])
        my_payments = np.array([])

        total_wins = 0
        total_utility = 0
        total_spent = 0
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

            total_wins += my_win
            total_utility += f_t
            total_spent += c_t

            print(f"Auction {t+1}: Bid: {bid_t}, Opponent bid {m_t}, Utility: {f_t}, Payment: {c_t}, Winner: {winner}")
        
        print(f"Total wins: {total_wins}, Total utility: {total_utility}, Total spent: {total_spent}")

    def pricing(self):
        
        num_buyers = self.num_buyers
        item_cost = 0.1
        min_price = item_cost
        max_price = 1

        eps = self.T_pricing**(-1/3)
        K = int(1/eps + 1)
        
        discr_prices = np.linspace(min_price, max_price, K)

        learning_rate = np.sqrt(np.log(K) / self.T_pricing)
        agent = ag.HedgeAgent(K, learning_rate)

        conversion_probability = lambda p, theta: (1 - p)**theta

        theta_seq = generate_adv_sequence(self.T_pricing, 0.5, 2)
        envir = envi.AdversarialPricingFullEnvironment(conversion_probability, theta_seq, item_cost)

        # demand = conversion_probability * num_buyers
        # reward_func = lambda price, demand: demand * (price - item_cost)

        total_sales = 0
        total_profit = 0
        for t in range(self.T_pricing):
            #pull arm
            arm_t = agent.pull_arm()
            #get price
            price_t = discr_prices[arm_t]
            
            losses_t = np.array([])
            #full-feedback: need feedback on all prices
            d_t, r_t = envir.round(discr_prices, num_buyers)
            # compute losses with normalized reward
            losses_t = 1 - r_t/num_buyers

            # update done only for the played price
            total_sales += d_t[arm_t]
            total_profit += r_t[arm_t]

            #update agent with full feedback
            agent.update(losses_t)

            print(f"Day {t+1}: Price: {price_t}, Losses: {losses_t}, Theta: {theta_seq[t]}, Demand: {d_t}, Profit: {r_t}")
        
        print(f"Total Sales: {total_sales}, Total Profit: {total_profit}")      

        

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
        print('luha was here')
        req.main()
    elif args.run_type == 'bidding':
        req.bidding()
    elif args.run_type == 'pricing':
        req.pricing()
    else:
        print("Invalid run type")
        exit(1)
    