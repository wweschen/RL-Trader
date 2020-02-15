from base_agent  import BaseAgent
#from src.history import History
from replay_memory import DQNReplayMemory
from networks.dqn import DQN
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from time import sleep
import  os

class DQNAgent(BaseAgent):

    def __init__(self, config, environment):
        super(DQNAgent, self).__init__(config,environment)
        #self.history = History(config)
        self.replay_memory = DQNReplayMemory(config)
        self.net = DQN(len(environment.n_actions), config)
        self.net.build()
        self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

        self.account_profit_loss = 0.
        self.forecast_window=config.forecast_window
    def observe(self):

        reward = max(self.min_reward, min(self.max_reward, self.env.reward))
        data = self.env.data

        self.stoday = datetime.strftime(self.env.today, '%m/%d/%Y')
        todays = [date == self.stoday for date in data.dates]

        self.replay_memory.add(data.opens,data.highs,data.lows,data.closes,data.volumes, todays, reward, self.env.action, self.env.terminal,
                               self.env.position, self.env.today,self.env.order_price,
                               self.env.current_price, self.env.time_since)

        if self.i < self.config.epsilon_decay_episodes:
            self.epsilon -= self.config.epsilon_decay
        if self.i % self.config.train_freq == 0 and self.i > self.config.train_start:
            opens, highs, lows, closes, volumes, todays, action, reward, \
            opens_, highs_, lows_, closes_, volumes_, todays_, \
            terminal,positions,dates,order_prices,current_prices,\
                    time_steps_since = self.replay_memory.sample_batch()

            q, loss= self.net.train_on_batch_target( opens,highs,lows,closes,volumes, todays, action, reward,
                                                     opens_, highs_, lows_, closes_, volumes_, todays_, terminal,
                                                     self.i, positions,dates,order_prices,current_prices,time_steps_since)
            self.total_q += q
            self.total_loss += loss
            self.update_count += 1
        if self.i % self.config.update_freq == 0:
            print('update the training target at:', self.i)
            self.net.update_target()

    def policy(self):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.env.get_valid_actions())

            return action
        else:

            feed_dict=self.net.get_feed_dict(self.env.data, self.env.reward, self.env.action, self.env.terminal,
                               self.env.position, self.env.today,self.env.order_price, self.env.current_price, self.env.time_since )

            a = self.net.q_action.eval(
                feed_dict=feed_dict
                , session=self.net.sess)

            action  = a[0]

            while  action not in self.env.get_valid_actions():
                #print('invalid action called:',action)
                action = np.random.choice(self.env.get_valid_actions())
            return action


    def train(self, steps, eventSource):
        render = False
        self.env.random_past_day()
        num_game, self.update_count, ep_reward = 0,0,0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        ep_rewards, actions = [], []
        t = 0

        #for _ in range(self.config.history_len):
        #    self.history.add(self.env.data)

        for self.i in tqdm(range(self.i, steps)):

            action = self.policy()
            self.env.step(action)

            self.observe()

            if action in (1, 2):
                direction = 1 if  action == 2 else -1
                pl = direction * self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost
                self.log_trade(action, self.env.today, self.env.time, self.env.unit, self.env.order_price,
                               self.env.current_price,pl)

                self.account_profit_loss += pl
            forecasts = np.zeros(self.forecast_window, dtype=np.float32)
            forecast_history = np.zeros(self.forecast_window, dtype=np.float32)
            sleep(1)
            eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts, forecast_history)

            if self.env.terminal:
                t = 0
                self.env.random_past_day()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
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

                    print('log to tensorboard at:', self.i)
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
                print('Saving the parameters at:',self.i)
                self.save()
            if self.i % 100000 == 0:
                j = 0
                render = True

            if render:
                #self.env_wrapper.env.render()
                j += 1
                if j == 1000:
                    render = False


    def play(self, episodes, net_path):
        self.net.restore_session(path=net_path)
        self.env.new_game()
        i = 0
        #for _ in range(self.config.history_len):
        #    self.history.add(self.env.data)
        episode_steps = 0
        while i < episodes:
            feed_dict=self.net.get_feed_dict(self.env.data, self.env.reward, self.env.action, self.env.terminal,
                               self.env.position, self.env.today,self.env.order_price, self.env.current_price, self.env.time_since )


            a = self.net.q_action.eval(
             feed_dict=feed_dict,
                session=self.net.sess)

            action = a[0]
            self.env.step(action)
            #self.history.add(self.env.data)
            episode_steps += 1
            if episode_steps > self.config.max_steps:
                self.env.terminal = True
            if self.env.terminal:
                episode_steps = 0
                i += 1
                self.env.new_play_game()
                #for _ in range(self.config.history_len):
                #    data = self.env.data
                #    self.history.add(data)

