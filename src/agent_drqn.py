from base_agent  import BaseAgent
#from src.history import History
from replay_memory import DRQNReplayMemory
from networks.drqn import DRQN
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from time import sleep
import logging
import pprint as pp
import  tensorflow as tf
from tf_utils import shape
import os
from collections import deque


class DRQNAgent(BaseAgent):

    def __init__(self, config,environment):
        super(DRQNAgent, self).__init__(config,environment)
        self.replay_memory = DRQNReplayMemory(config)
        self.net = DRQN(len(self.env.n_actions), config)
        self.net.build()
        self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

        self.init_logging(self.net.dir_log)

        self.queue = deque(maxlen=self.config.mem_size)

        self.account_profit_loss = 0.
        self.forecast_window = config.forecast_window
        self.close_attempts = 0

        logging.info('all parameters:')
        logging.info(pp.pformat([(var.name, shape(var)) for var in tf.global_variables()]))

        logging.info('trainable parameters:')
        logging.info(pp.pformat([(var.name, shape(var)) for var in tf.trainable_variables()]))

        logging.info('trainable parameter count:')
        logging.info(str(np.sum(np.prod(shape(var)) for var in tf.trainable_variables())))



    def observe(self, t):
        #reward = max(self.min_reward, min(self.max_reward, self.env.reward))
        reward =self.env.reward
        dates,times,opens,highs,lows,closes,volumes = self.env.data.get_latest_n(self.net.series_length)

        stoday = datetime.strftime(self.env.today, '%m/%d/%Y')
        todays = [date == stoday for date in dates]

        #
        #
        # if self.env.action in (self.env.action_labels.index("buy_open"),
        #                        self.env.action_labels.index("hold_long"),
        #                        self.env.action_labels.index("sell_open"),
        #                        self.env.action_labels.index("hold_short")):
        #     self.queue.appendleft((opens, highs, lows, closes, volumes, todays, reward,
        #                            self.env.action, self.env.terminal,
        #                            self.env.position, self.env.today, self.env.order_price,
        #                            self.env.current_price, self.env.time_since))
        #
        # if self.env.action in (self.env.action_labels.index("sell_close")
        #                         ,self.env.action_labels.index("buy_close")):
        #     self.queue.appendleft(( opens,  highs, lows, closes,  volumes, todays, reward,
        #                            self.env.action, self.env.terminal,
        #                            self.env.position, self.env.today, self.env.order_price,
        #                            self.env.current_price, self.env.time_since))
        #     #if reward >0, add to replay memory, otherwise discard
        #
        #     #we apply a eligibility trace to all steps lead to this reward based on price changes
        #     c=self.queue.copy()
        #     p=[c[i][12] for i in range(len(c))]
        #     p.insert(0, p[0])
        #     p.append(p[-1])
        #     if((p[-1] - p[0]) !=0):
        #         r = [reward*(p[i + 1] - p[i - 1]) / (2 * (p[-1] - p[0])) for i in range(1, len(c)+1)]
        #     else:
        #         r=[0 for i in range(1, len(c)+1)]
        #
        #     #print('Time:',self.env.time,'reward:', reward, self.env.action, self.env.position, self.env.order_price, self.env.current_price, )
        #     i=0
        #     while self.queue :
        #
        #         d = self.queue.pop()
        #         l=list(d)
        #         #print('before:', d[6],d[11],d[12])
        #         l[6] = r[i]
        #         i += 1
        #         d = tuple(l)
        #         #print('added to memory:',d[6])
        #         self.replay_memory.add(*(d))


        self.replay_memory.add(opens, highs, lows, closes, volumes, todays, reward,
                               self.env.action, self.env.terminal,
                               self.env.position, self.env.today, self.env.order_price,
                               self.env.current_price, self.env.time_since)

        if self.i < self.config.epsilon_decay_episodes:
            self.epsilon -= self.config.epsilon_decay

        if self.i % self.config.train_freq == 0 and self.i > self.config.train_start:
            opens_, highs_, lows_,closes_, volumes_, todays_, \
            actions_, rewards_,terminals_, dates_,positions_, \
            order_prices_, current_prices_, time_since_ = self.replay_memory.sample_batch()

            q, loss= self.net.train_on_batch_target(opens_, highs_, lows_,closes_, volumes_, todays_,
            actions_, rewards_,terminals_, dates_,positions_,
            order_prices_, current_prices_, time_since_, self.i)
            self.total_q += q
            self.total_loss += loss
            self.update_count += 1

        if self.i % self.config.update_freq == 0:
            self.net.update_target()

    def policy(self):
        self.random = False
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.env.get_valid_actions())
            return action
        else:
            dates, times, opens, highs, lows, closes, volumes = self.env.data.get_latest_n(self.net.series_length)

            stoday = datetime.strftime(self.env.today, '%m/%d/%Y')
            is_today = [date == stoday for date in dates]

            a, self.lstm_state_c, self.lstm_state_h = self.net.sess.run(
                [self.net.q_action, self.net.state_output_c, self.net.state_output_h],{
                #self.net.state : [[state]],
                self.net.opens: [opens],
                self.net.highs: [highs],
                self.net.lows: [lows],
                self.net.closes: [closes],
                self.net.volumes: [volumes],
                self.net.todays: [is_today],
                self.net.positions: [self.env.position],
                self.net.order_prices: [self.env.order_price],
                self.net.current_prices: [self.env.current_price],
                self.net.time_since: [self.env.time_since],
                self.net.lengths: [self.net.series_length],

                    self.net.c_state_train: self.lstm_state_c,
                self.net.h_state_train: self.lstm_state_h
            })

            action= a[0]
            while  action not in self.env.get_valid_actions():
                #print('invalid action called:',action)
                action = np.random.choice(self.env.get_valid_actions())

            return action



    def train(self, steps,eventSource):
        render = False
        self.env.random_past_day()
        num_game, self.update_count, ep_reward = 0,0,0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        ep_rewards, actions = [], []
        t = 0
        self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single

        for self.i in tqdm(range(self.i, steps)):
            action = self.policy()


            # end of day logic
            # if it is close to the end of day, we need to try to close out our position on a good term,
            # not to wait to be forced to close at the end of day, which is a fast rule of this algorithm
            # here is a logic, 10 allowances once within self.forecast_window minutes of closing minute, close on any change of the first nine
            # if the action needed agrees with the prediction, which is to close, execute it and then stay neutral the rest of day
            # or forcefully close the position at 10th allowance then stay neutral.
            grace_period = timedelta(minutes=15)
            end_time = datetime.strptime("16:00", '%H:%M')
            # print(end_time, self.time, end_time - self.time)
            if end_time - self.env.time < grace_period:
                if (self.env.position > 0.):  # a long position
                    if action != self.env.action_labels.index("sell_close"):  # close long
                        self.close_attempts += 1
                        if self.close_attempts > 10:
                            action = self.env.action_labels.index("sell_close")
                if (self.env.position < 0.):  # a short position
                    if action != self.env.action_labels.index("buy_close"):  # close a short
                        if self.close_attempts > 10:
                            action = self.env.action_labels.index("buy_close")
                if (self.env.position == 0.):
                    action = self.env.action_labels.index("stay_neutral")
                if self.close_attempts > 10:
                    if (self.env.position > 0.):  # a long position
                        action = self.env.action_labels.index("sell_close")  # close long
                    if (self.env.position < 0.):  # a long position
                        action = self.env.action_labels.index("buy_close")  # close short

            # Beginning of day logic, don't trade the first fifteen minutes
            fifteen_minute = timedelta(minutes=15)
            start_time = datetime.strptime("9:30", '%H:%M')
            # print(end_time, self.time, end_time - self.time)
            if self.env.time - start_time <= fifteen_minute:
                action = self.env.action_labels.index("stay_neutral")


            self.env.step(action)

            # if self.random:
            #     dates, times, opens, highs, lows, closes, volumes = self.env.data.get_latest_n(self.net.series_length)
            #
            #     stoday = datetime.strftime(self.env.today, '%m/%d/%Y')
            #     todays = [date == stoday for date in dates]
            #
            #     self.lstm_state_c, self.lstm_state_h = self.net.sess.run([self.net.state_output_c, self.net.state_output_h], {
            #         self.net.opens: [opens],
            #         self.net.highs: [highs],
            #         self.net.lows: [lows],
            #         self.net.closes: [closes],
            #         self.net.volumes: [volumes],
            #
            #         self.net.todays: [todays],
            #         self.net.positions: [self.env.position],
            #         self.net.order_prices: [self.env.order_price],
            #         self.net.current_prices: [self.env.current_price],
            #         self.net.time_since: [self.env.time_since],
            #         self.net.lengths: [self.net.series_length],
            #
            #         self.net.c_state_train : self.lstm_state_c,
            #         self.net.h_state_train: self.lstm_state_h
            #     })

            self.observe(t)


            if action not in (  self.env.action_labels.index("stay_neutral"),
                                self.env.action_labels.index("hold_long"),
                                self.env.action_labels.index("hold_short")):
                if action in (self.env.action_labels.index("buy_open"),self.env.action_labels.index("sell_open")):
                    pl =  - self.env.open_cost
                if action ==self.env.action_labels.index("sell_close"):
                    pl=self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost

                if action ==self.env.action_labels.index("buy_close"):
                    pl=-1*self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost

                self.log_trade(self.env.action_labels[action], self.env.today, self.env.time, self.env.unit, self.env.order_price,
                               self.env.current_price,pl)

                self.account_profit_loss += pl


            forecasts = np.zeros(self.forecast_window, dtype=np.float32)
            forecast_history = np.zeros(self.forecast_window, dtype=np.float32)
            #sleep(1)
            #eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts, forecast_history)


            if self.env.terminal:
                t = 0
                self.close_attempts = 0
                self.env.random_past_day()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
                self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single
            else:
                ep_reward += self.env.reward
                t += 1
            actions.append(action)
            total_reward += self.env.reward

            if self.i >= self.config.train_start:
                if self.i % self.config.test_step == self.config.test_step -1:
                    avg_reward = total_reward / self.config.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    sum_dict = {
                        'average_reward': avg_reward,
                        'average_loss': avg_loss,
                        'average_q': avg_q,
                        'ep_max_reward': max_ep_reward,
                        'ep_min_reward': min_ep_reward,
                        'ep_num_game': num_game,
                        'learning_rate': self.net.learning_rate,
                        'ep_rewards': ep_rewards,
                        'ep_actions': actions
                    }
                    self.net.inject_summary(sum_dict, self.i)
                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []

            if self.i % 500000 == 0 and self.i > 0:
                j = 0
                print('Saving model at:',self.i)
                self.save()
            if self.i % 100000 == 0:
                j = 0
                render = True

            if render:

                j += 1
                if j == 1000:
                    render = False

    def play(self, episodes, eventSource):
        self.net.restore_session()
        self.env.random_past_day()

        self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single
        i = 0
        episode_steps = 0
        while i < episodes:
            dates, times, opens, highs, lows, closes, volumes = self.env.data.get_latest_n(self.net.series_length)

            stoday = datetime.strftime(self.env.today, '%m/%d/%Y')
            todays = [date == stoday for date in dates]
            a, self.lstm_state_c, self.lstm_state_h = self.net.sess.run(
                [self.net.q_action, self.net.state_output_c, self.net.state_output_h], {
                    # self.net.state : [[state]],
                    self.net.opens: [opens],
                    self.net.highs: [highs],
                    self.net.lows: [lows],
                    self.net.closes: [closes],
                    self.net.volumes: [volumes],
                    self.net.todays: [todays],
                    self.net.positions: [self.env.position],
                    self.net.order_prices: [self.env.order_price],
                    self.net.current_prices: [self.env.current_price],
                    self.net.time_since: [self.env.time_since],
                    self.net.lengths: [self.net.series_length],
                    self.net.c_state_train: self.lstm_state_c,
                    self.net.h_state_train: self.lstm_state_h
                })

            action = a[0]
            while action not in self.env.get_valid_actions():
                # print('invalid action called:',action)
                action = np.random.choice(self.env.get_valid_actions())

            # end of day logic
            # if it is close to the end of day, we need to try to close out our position on a good term,
            # not to wait to be forced to close at the end of day, which is a fast rule of this algorithm
            # here is a logic, 10 allowances once within self.forecast_window minutes of closing minute, close on any change of the first nine
            # if the action needed agrees with the prediction, which is to close, execute it and then stay neutral the rest of day
            # or forcefully close the position at 10th allowance then stay neutral.
            grace_period = timedelta(minutes=15)
            end_time = datetime.strptime("16:00", '%H:%M')
            # print(end_time, self.time, end_time - self.time)
            if end_time - self.env.time < grace_period:
                if (self.env.position > 0.):  # a long position
                    if action != self.env.action_labels.index("sell_close"):  # close long
                        self.close_attempts += 1
                        if self.close_attempts > 10:
                            action = self.env.action_labels.index("sell_close")
                if (self.env.position < 0.):  # a short position
                    if action != self.env.action_labels.index("buy_close"):  # close a short
                        if self.close_attempts > 10:
                            action = self.env.action_labels.index("buy_close")
                if (self.env.position == 0.):
                    action = self.env.action_labels.index("stay_neutral")
                if self.close_attempts > 10:
                    if (self.env.position > 0.):  # a long position
                        action = self.env.action_labels.index("sell_close")  # close long
                    if (self.env.position < 0.):  # a long position
                        action = self.env.action_labels.index("buy_close")  # close short

            # Beginning of day logic, don't trade the first fifteen minutes
            fifteen_minute = timedelta(minutes=15)
            start_time = datetime.strptime("9:30", '%H:%M')
            # print(end_time, self.time, end_time - self.time)
            if self.env.time - start_time <= fifteen_minute:
                action = self.env.action_labels.index("stay_neutral")

            self.env.step(action)

            if action not in (  self.env.action_labels.index("stay_neutral"),
                                self.env.action_labels.index("hold_long"),
                                self.env.action_labels.index("hold_short")):
                if action in (self.env.action_labels.index("buy_open"), self.env.action_labels.index("sell_open")):
                    pl = - self.env.open_cost
                if action == self.env.action_labels.index("sell_close"):
                    pl = self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost

                if action == self.env.action_labels.index("buy_close"):
                    pl = -1 * self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost

                self.log_trade(self.env.action_labels[action], self.env.today, self.env.time, self.env.unit, self.env.order_price,
                               self.env.current_price, pl)

                self.account_profit_loss += pl

            forecasts = np.zeros(self.forecast_window, dtype=np.float32)
            forecast_history = np.zeros(self.forecast_window, dtype=np.float32)
            sleep(1)
            eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts, forecast_history)

            episode_steps += 1
            if episode_steps > self.config.max_steps:
                self.env.terminal = True
            if self.env.terminal:
                episode_steps = 0
                i += 1
                self.env.random_past_day()
                self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single

    def init_logging(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)

        # reload(logging)  # bad
        logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        level=logging.INFO,
        format='[[%(asctime)s]] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())