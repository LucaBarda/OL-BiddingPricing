import argparse
import numpy as np
import math

from agents import GPUCBAgent
from agents import NonStationaryClairvoyantPricing

from environments import NonStationaryPricingEnvironment
from utils import *
import matplotlib.pyplot as plt


class Requirement3:

    def __init__(self, args):
        self.args = args
        #extract all args in members
        for key, value in vars(args).items():
            setattr(self, key, value)

        #set seed in numpy
        np.random.seed(self.seed)

        #pricing parameters
        self.T_pricing = 10000
        self.T_interval = 2000
        self.intervals = math.ceil(self.T_pricing/self.T_interval)
        self.conversion_probabilities = [
            lambda price: 1 - price / 40,
            lambda price: 1 - price / 60,
            lambda price: np.exp(- ((price-10) **2  )/25),
            lambda price: (1 / (2 * np.sqrt(0.05 * price - 0.3))) - 0.5,
            lambda price: 1 - (2.4 * ((price - 10) / 30) 
                               - 2.8 * ((price- 10) / 30) ** 2 
                               + 1.4 * ((price- 10) / 30) ** 3)
        ]
        #defining pricing cost parameters
        self.cost = 10
        self.max_price = 40
        self.min_price = self.cost

        #using discretization prescribed by theory but for each interval
        epsilon = self.T_interval**(-0.33)
        self.K = int(1/epsilon)

        self.prices = np.linspace(self.min_price, self.max_price, self.K)

        #defining the clairvoyant agent and get the best prices
        clairvoint_agent = NonStationaryClairvoyantPricing(self.num_buyers, self.conversion_probabilities, self.prices, self.cost, self.intervals)
        self.best_prices = clairvoint_agent.getBestPrices()

    def show_demand_curves(self):
        demand_curves = [self.num_buyers * self.conversion_probabilities[i](self.prices) for i in range(self.intervals)]
        profit_curves = [demand_curve * (self.prices - self.cost) for demand_curve in demand_curves]
        best_prices_indices = [np.argmax(profit_curve) for profit_curve in profit_curves]

        fig, axs = plt.subplots(1, 2)
        for i in range(self.intervals):
            axs[0].plot(self.prices, demand_curves[i], label=f'interval {i}')
            axs[0].scatter(self.best_prices[i], demand_curves[i][best_prices_indices[i]], color='red')
            axs[1].plot(self.prices, profit_curves[i], label=f'interval {i}')
            axs[1].scatter(self.best_prices[i], profit_curves[i][best_prices_indices[i]], color='red')
        axs[0].set_title('Demand curves')
        axs[0].set_xlabel('Price')
        axs[0].set_ylabel('Demand')
        axs[0].legend()
        axs[1].set_title('Profit curves')
        axs[1].set_xlabel('Price')
        axs[1].set_ylabel('Profit')
        axs[1].legend()
        #plt.tight_layout()
        plt.show()


    def main(self):

        agent = GPUCBAgent(self.T_pricing, discretization=self.K)
        env = NonStationaryPricingEnvironment(cost = self.cost, 
                                              conversion_probabilities = self.conversion_probabilities, 
                                              T_interval = self.T_interval, 
                                              seed = self.seed)

        
        agent_rewards = np.array([])
        for t in range(self.T_pricing):
            p_t = agent.pull_arm()
            p_t = denormalize_zero_one(p_t, self.min_price, self.max_price)
            d_t, r_t = env.round(p_t, n_t=self.num_buyers)
            agent.update(r_t/self.num_buyers)
            agent_rewards = np.append(agent_rewards, r_t)

        cumulative_regret = np.cumsum(expected_clairvoyant_rewards-agent_rewards)

        plt.figure()
        plt.plot(cumulative_regret)
        plt.title('Cumulative Regret of GP-UCB')
        plt.xlabel('$t$')
        plt.show()
        

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
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=100)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    #for pricing only
    parser.add_argument("--num_buyers", dest="num_buyers", type = int, default = 100)

    args = parser.parse_args()    

    req = Requirement3(args)

    req.show_demand_curves()
    