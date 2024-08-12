import numpy as np

class NonStationaryClairvoyantPricing:
    def __init__(self, num_buyers, conversion_probabilities, prices, cost, intervals):
        profit_curves = [num_buyers * (prices - cost) * conversion_probabilities[i](prices) for i in range(intervals)]

        best_prices_indices = [np.argmax(profit_curve) for profit_curve in profit_curves]
        self.best_prices = [prices[i] for i in best_prices_indices]

    def getBestPrices(self):
        return self.best_prices
    

