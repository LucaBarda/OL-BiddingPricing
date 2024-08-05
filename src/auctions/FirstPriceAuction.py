import numpy as np

from .AbstractAuction import Auction

class FirstPriceAuction(Auction):
    def __init__(self, ctrs):
        self.ctrs = ctrs
        self.n_adv = len(self.ctrs)
    
    def get_winners(self, bids):
        adv_values = self.ctrs*bids
        adv_ranking = np.argsort(adv_values)
        winner = adv_ranking[-1]
        return winner, adv_values
    
    def get_payments_per_click(self, winners, values, bids):
        # in first-price auctions the winner pays what he bid
        payment = bids[winners]
        # or equivalently: payment = values[winners]/self.ctrs[winners]
        return payment.round(2)