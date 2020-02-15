import numpy as np

from market_chart_widget import  PriceData
import os

class History:

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.history_len = config.history_len
        self.history =[ PriceData(config.price_data_size) for i in range(config.history_len)]

    def add(self, data):
        self.history[:-1] = self.history[1:]
        self.history[-1] = data

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history
