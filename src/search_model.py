
import numpy as np
from datetime import datetime, timedelta
from numpy import genfromtxt
import pandas as pd
import random
from threading import Thread,Event


class SearchModel:
    def __init__(self,state,forecasts, current_price,open_cost,unit,t):


        self.open_cost = open_cost
        self.unit = unit

        self.t_max=390

        self.n_actions = [0, 1, 2, 3, 4, 5, 6]
        self.action_labels = ['stay_neutral', 'buy_open', 'sell_close', 'hold_long', 'sell_open', 'buy_close',
                              'hold_short']

        self.position =state[0]
        self.order_price = state[1]
        self.current_price = current_price
        self.data=state[2:]
        self.forecast_window=len(state)-2
        self.forecasts=forecasts
        self.data_gen = (d for d in self.forecasts.tolist())

        self.t=t



    def get_valid_actions(self):
        # actions:
        # # 0= stay side line
        # # 1 = open buy, 2 = sell to close, 3 = hold long position
        # # 4 = sell to open, 5 = buy to close, 6 = hold short position
        #
        if self.position==0: #neutral
            return [0, 1, 4]  # noop, Buy to Open,   Sell to Open
        if self.position >0: #long
            return [2, 3]  #   sell to close, hold long
        if self.position <0: #short
            return [5,6]   # buy to close, hold short
        #
        # if self.position == 0:  # neutral
        #     return [0, 1, 2]  # hold, Buy , Sell
        # if self.position > 0:  # long
        #     return [0, 2]  # hold, sell to close,
        # if self.position < 0:  # short
        #     return [0, 1]  # hold, buy to close


    def step(self, action):

        done = False
        reward =0.

        d=next(self.data_gen)
        self.data[0]=d
        self.current_price = d

        unitcost = self.open_cost / self.unit



        if action == 0:  # noop
            reward = 0.
            if self.position !=0:
                raise ValueError('can not take this action:' + str(action))


        elif action == 1:  # Buy Open

            if self.position != 0:
                raise ValueError('can not take this action:' + str(action))

            if self.position == 0:  # open a long position
                self.position = 1
                self.order_price = self.current_price
                reward = -unitcost


        elif action == 2:  # Sell Close
            if self.position <= 0:
                raise ValueError('can not take this action:' + str(action))

            if self.position > 0:  # close long position
                reward = self.position * (self.current_price - self.order_price) - unitcost
                self.position = 0



        elif action == 3:  # hold long
            if self.position <= 0:
                raise ValueError('can not take this action (double down): ' + str(action))


        elif action == 4:  # sell to Open
            if self.position != 0:
                raise ValueError('can not take this action (double down): ' + str(action))

            if self.position == 0:  # open a short position
                self.position = -1
                self.order_price = self.current_price
                reward = -unitcost


        elif action == 5:  # Sell Close
            if self.position >= 0:
                raise ValueError('can not take this action:' + str(action))

            if self.position < 0.:  # close a short position
                reward = self.position * (self.current_price - self.order_price) - unitcost
                self.position = 0


        elif action == 6:  # hold short
            if self.position >= 0:
                raise ValueError('can not take this action:' + str(action))


        else:
            raise ValueError('no such action: ' + str(action))


        self.t += 1
        self.action=action
        self.terminal = self.t==self.t_max
        self.reward = reward

        s = np.append([self.position,self.order_price], self.data[0:self.forecast_window])
        return s, self.position, self.order_price,self.current_price, reward, self.t==self.t_max, self.get_valid_actions()

