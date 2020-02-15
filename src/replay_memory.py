import numpy as np
import pandas as pd
import random
import os
from market_chart_widget import  PriceData

class ReplayMemory:

    def __init__(self, config):
        self.config = config
        self.actions = np.empty(self.config.mem_size, dtype=np.int32)
        self.rewards = np.empty(self.config.mem_size, dtype=np.int32)

        self.opens = np.empty((self.config.mem_size,config.observation_window), dtype=np.float16)
        self.highs = np.empty((self.config.mem_size, config.observation_window), dtype=np.float16)
        self.lows = np.empty((self.config.mem_size, config.observation_window), dtype=np.float16)
        self.closes = np.empty((self.config.mem_size, config.observation_window), dtype=np.float16)
        self.volumes = np.empty((self.config.mem_size, config.observation_window), dtype=np.float16)
        self.todays = np.empty((self.config.mem_size, config.observation_window), dtype=np.bool)

        self.terminals = np.empty(self.config.mem_size, dtype=np.float16)

        self.dates  =np.empty(self.config.mem_size, dtype='datetime64[us]')
        self.positions = np.empty(self.config.mem_size, dtype=np.int32)
        self.order_prices = np.empty(self.config.mem_size, dtype=np.float16)
        self.est_current_prices = np.empty(self.config.mem_size, dtype=np.float16)
        self.time_since_open = np.empty(self.config.mem_size, dtype=np.int16)


        self.count = 0
        self.current = 0
        self.dir_save = config.dir_save + "memory/"

        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)

    def save(self):
        np.save(self.dir_save + "actions.npy", self.actions)
        np.save(self.dir_save + "rewards.npy", self.rewards)
        np.save(self.dir_save + "terminals.npy", self.terminals)

        np.save(self.dir_save + "opens.npy", self.opens)
        np.save(self.dir_save + "highs.npy", self.highs)
        np.save(self.dir_save + "lows.npy", self.lows)
        np.save(self.dir_save + "closes.npy", self.closes)
        np.save(self.dir_save + "volumes.npy", self.volumes)
        np.save(self.dir_save + "todays.npy", self.todays)


        np.save(self.dir_save + "dates.npy", self.dates)
        np.save(self.dir_save + "positions.npy", self.positions)
        np.save(self.dir_save + "order_prices.npy", self.order_prices)
        np.save(self.dir_save + "est_current_prices.npy", self.est_current_prices)
        np.save(self.dir_save + "time_since_open.npy", self.time_since_open)

    def load(self):
        self.actions = np.load(self.dir_save + "actions.npy")
        self.rewards = np.load(self.dir_save + "rewards.npy")
        self.terminals = np.load(self.dir_save + "terminals.npy")

        self.opens=np.load(self.dir_save + "opens.npy")
        self.highs =np.load(self.dir_save + "highs.npy")
        self.lows =np.load(self.dir_save + "lows.npy")
        self.closes =np.load(self.dir_save + "closes.npy")
        self.volumes =np.load(self.dir_save + "volumes.npy")
        self.todays  =np.load(self.dir_save + "todays.npy")

        self.dates  = np.load(self.dir_save + "dates.npy")
        self.positions =  np.load(self.dir_save + "positions.npy")
        self.order_prices =  np.load(self.dir_save + "order_prices.npy")
        self.est_current_prices =  np.load(self.dir_save + "est_current_prices.npy")
        self.time_since_open =  np.load(self.dir_save + "time_since_open.npy")


class DQNReplayMemory(ReplayMemory):

    def __init__(self, config):
        super(DQNReplayMemory, self).__init__(config)

        self.opens_pre = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.highs_pre = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.lows_pre = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.closes_pre = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.volumes_pre = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.todays_pre = np.empty((self.config.batch_size, config.observation_window), dtype=np.bool)

        self.opens_post = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.highs_post = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.lows_post = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.closes_post = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.volumes_post = np.empty((self.config.batch_size, config.observation_window), dtype=np.float16)
        self.todays_post = np.empty((self.config.batch_size, config.observation_window), dtype=np.bool)

    def getState(self, index):

        index = index % self.count
        return  self.opens[index],self.highs[index],self.lows[index],self.closes[index],self.volumes[index],self.todays[index]

    def add(self, open,high,low,close,volume, today, reward, action, terminal,position,date,order_price,current_price,time_since):

        self.actions[self.current] = action
        self.rewards[self.current] = reward

        self.opens[self.current] = open
        self.highs[self.current] = high
        self.lows[self.current] = low
        self.closes[self.current] = close
        self.volumes[self.current] = volume
        self.todays[self.current] = today

        self.terminals[self.current] = float(terminal)

        self.dates[self.current] = np.datetime64(date)
        self.positions[self.current] = position
        self.order_prices[self.current] = order_price
        self.est_current_prices[self.current] = current_price
        self.time_since_open[self.current] = time_since

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size

    def sample_batch(self):

        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(1, self.count-1)

                if self.terminals[index]:
                    continue
                break

            self.opens_pre[len(indices)], self.highs_pre[len(indices)], self.lows_pre[len(indices)], \
            self.closes_pre[len(indices)], \
            self.volumes_pre[len(indices)], self.todays_pre[len(indices)] = self.getState(index - 1)

            self.opens_post[len(indices)], self.highs_post[len(indices)], self.lows_post[len(indices)], \
            self.closes_post[len(indices)], \
            self.volumes_post[len(indices)], self.todays_post[len(indices)] = self.getState(index)

            indices.append(index)

        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        positions = self.positions[indices]
        dates = self.dates[indices]
        order_prices = self.order_prices[indices]
        current_prices = self.est_current_prices[indices]
        time_steps_since = self.time_since_open[indices]


        return self.opens_pre,self.highs_pre,self.lows_pre,self.closes_pre,self.volumes_pre,self.todays_pre, \
               actions,rewards,\
               self.opens_post,self.highs_post,self.lows_post,self.closes_post,self.volumes_post,self.todays_post, \
               terminals,positions,dates,order_prices,current_prices,time_steps_since

class DRQNReplayMemory(ReplayMemory):

    def __init__(self, config):
        super(DRQNReplayMemory, self).__init__(config)

        self.timesteps = np.empty((self.config.mem_size), dtype=np.int32)

        self.opens_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1,config.observation_window),
                                  dtype=np.float16)
        self.highs_ = np.empty((self.config.batch_size,self.config.min_history + self.config.states_to_update +1, config.observation_window),
                                  dtype=np.float16)
        self.lows_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1,config.observation_window),
                                 dtype=np.float16)
        self.closes_ = np.empty((self.config.batch_size,self.config.min_history + self.config.states_to_update +1, config.observation_window),
                                   dtype=np.float16)
        self.volumes_ = np.empty((self.config.batch_size,self.config.min_history + self.config.states_to_update +1, config.observation_window),
                                    dtype=np.float16)
        self.todays_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1,config.observation_window),
                                   dtype=np.bool)

        self.actions_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))
        self.rewards_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))
        self.terminals_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))

        self.dates_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.positions_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.order_prices_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.current_prices_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.time_since_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))

    def add(self, open, high, low, close, volume, today, reward, action, terminal, position, date, order_price,
            current_price, time_since):
        self.actions[self.current] = action
        self.rewards[self.current] = reward

        self.opens[self.current] = open
        self.highs[self.current] = high
        self.lows[self.current] = low
        self.closes[self.current] = close
        self.volumes[self.current] = volume
        self.todays[self.current] = today

        self.terminals[self.current] = float(terminal)

        self.dates[self.current] = np.datetime64(date)
        self.positions[self.current] = position
        self.order_prices[self.current] = order_price
        self.est_current_prices[self.current] = current_price
        self.time_since_open[self.current] = time_since

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size


    def getState(self, index):

        index = index % self.count

        return self.opens[index - (self.config.min_history + self.config.states_to_update + 1): index], \
               self.highs[index - (self.config.min_history + self.config.states_to_update + 1): index], \
               self.lows[index - (self.config.min_history + self.config.states_to_update + 1): index], \
               self.closes[index - (self.config.min_history + self.config.states_to_update + 1): index], \
               self.volumes[index - (self.config.min_history + self.config.states_to_update + 1): index],\
               self.todays[index - (self.config.min_history + self.config.states_to_update + 1): index]

    def get_scalars(self, index):
        t = self.terminals[index - (self.config.min_history + self.config.states_to_update + 1): index]
        a = self.actions[index - (self.config.min_history + self.config.states_to_update + 1): index]
        r = self.rewards[index - (self.config.min_history + self.config.states_to_update + 1): index]
        d= self.dates[index - (self.config.min_history + self.config.states_to_update + 1): index]
        p=self.positions[index - (self.config.min_history + self.config.states_to_update + 1): index]
        o=self.order_prices[index - (self.config.min_history + self.config.states_to_update + 1): index]
        c=self.est_current_prices[index - (self.config.min_history + self.config.states_to_update + 1): index]
        s= self.time_since_open[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a, t, r,d,p,o,c,s

    def sample_batch(self):
        assert self.count > self.config.min_history + self.config.states_to_update

        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(self.config.min_history, self.count-1)
                if index >= self.current and index - self.config.min_history < self.current:
                    continue
                if index < self.config.min_history + self.config.states_to_update + 1:
                    continue
                if self.timesteps[index] < self.config.min_history + self.config.states_to_update:
                    continue
                break

            self.opens_[len(indices)] ,self.highs_[len(indices)] ,self.lows_[len(indices)] , \
            self.closes_[len(indices)] , self.volumes_[len(indices)] , self.todays_[len(indices)]  =  self.getState(index)

            self.actions_[len(indices)], self.terminals_[len(indices)], self.rewards_[len(indices)],\
            self.dates_[len(indices)], self.positions_[len(indices)], self.order_prices_[len(indices)],\
            self.current_prices_[len(indices)], self.time_since_[len(indices)]  = self.get_scalars(index)

            indices.append(index)


        return self.opens_,self.highs_,self.lows_, self.closes_, self.volumes_, self.todays_, \
               self.actions_, self.rewards_, self.terminals_,self.dates_, self.positions_, \
               self.order_prices_,self.current_prices_,self.time_since_

class DRQN2ReplayMemory(ReplayMemory):

    def __init__(self, config):
        super(DRQNReplayMemory, self).__init__(config)

        self.timesteps = np.empty((self.config.mem_size), dtype=np.int32)

        self.opens_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1,config.observation_window),
                                  dtype=np.float16)
        self.highs_ = np.empty((self.config.batch_size,self.config.min_history + self.config.states_to_update +1, config.observation_window),
                                  dtype=np.float16)
        self.lows_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1,config.observation_window),
                                 dtype=np.float16)
        self.closes_ = np.empty((self.config.batch_size,self.config.min_history + self.config.states_to_update +1, config.observation_window),
                                   dtype=np.float16)
        self.volumes_ = np.empty((self.config.batch_size,self.config.min_history + self.config.states_to_update +1, config.observation_window),
                                    dtype=np.float16)
        self.todays_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1,config.observation_window),
                                   dtype=np.bool)

        self.actions_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))
        self.rewards_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))
        self.terminals_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))

        self.dates_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.positions_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.order_prices_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.current_prices_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.time_since_ = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))

    def add(self, open, high, low, close, volume, today, reward, action, terminal, position, date, order_price,
            current_price, time_since):
        self.actions[self.current] = action
        self.rewards[self.current] = reward

        self.opens[self.current] = open
        self.highs[self.current] = high
        self.lows[self.current] = low
        self.closes[self.current] = close
        self.volumes[self.current] = volume
        self.todays[self.current] = today

        self.terminals[self.current] = float(terminal)

        self.dates[self.current] = np.datetime64(date)
        self.positions[self.current] = position
        self.order_prices[self.current] = order_price
        self.est_current_prices[self.current] = current_price
        self.time_since_open[self.current] = time_since

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size


    def getState(self, index):

        index = index % self.count

        return self.opens[index - (self.config.min_history + self.config.states_to_update + 1): index], \
               self.highs[index - (self.config.min_history + self.config.states_to_update + 1): index], \
               self.lows[index - (self.config.min_history + self.config.states_to_update + 1): index], \
               self.closes[index - (self.config.min_history + self.config.states_to_update + 1): index], \
               self.volumes[index - (self.config.min_history + self.config.states_to_update + 1): index],\
               self.todays[index - (self.config.min_history + self.config.states_to_update + 1): index]

    def get_scalars(self, index):
        t = self.terminals[index - (self.config.min_history + self.config.states_to_update + 1): index]
        a = self.actions[index - (self.config.min_history + self.config.states_to_update + 1): index]
        r = self.rewards[index - (self.config.min_history + self.config.states_to_update + 1): index]
        d= self.dates[index - (self.config.min_history + self.config.states_to_update + 1): index]
        p=self.positions[index - (self.config.min_history + self.config.states_to_update + 1): index]
        o=self.order_prices[index - (self.config.min_history + self.config.states_to_update + 1): index]
        c=self.est_current_prices[index - (self.config.min_history + self.config.states_to_update + 1): index]
        s= self.time_since_open[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a, t, r,d,p,o,c,s

    def sample_batch(self):
        assert self.count > self.config.min_history + self.config.states_to_update

        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(self.config.min_history, self.count-1)
                if index >= self.current and index - self.config.min_history < self.current:
                    continue
                if index < self.config.min_history + self.config.states_to_update + 1:
                    continue
                if self.timesteps[index] < self.config.min_history + self.config.states_to_update:
                    continue
                break

            self.opens_[len(indices)] ,self.highs_[len(indices)] ,self.lows_[len(indices)] , \
            self.closes_[len(indices)] , self.volumes_[len(indices)] , self.todays_[len(indices)]  =  self.getState(index)

            self.actions_[len(indices)], self.terminals_[len(indices)], self.rewards_[len(indices)],\
            self.dates_[len(indices)], self.positions_[len(indices)], self.order_prices_[len(indices)],\
            self.current_prices_[len(indices)], self.time_since_[len(indices)]  = self.get_scalars(index)

            indices.append(index)


        return self.opens_,self.highs_,self.lows_, self.closes_, self.volumes_, self.todays_, \
               self.actions_, self.rewards_, self.terminals_,self.dates_, self.positions_, \
               self.order_prices_,self.current_prices_,self.time_since_