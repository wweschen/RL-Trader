from base_agent  import BaseAgent
#from src.history import History
from replay_memory import DQNReplayMemory
from networks.dqn import DQN
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from time import sleep
import  os
from pykalman import KalmanFilter

class KAgent(BaseAgent):

    def __init__(self, config, environment):
        super(KAgent, self).__init__(config,environment)
        #self.history = History(config)
        # self.replay_memory = DQNReplayMemory(config)
        #self.net = DQN(len(environment.n_actions), config)
        #self.net.build()

        #self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

        self.account_profit_loss = 0.
        self.forecast_window=config.forecast_window
        self.close_attempts = 0
        self.q_learning_rate=0.01

        self.policy = self.make_epsilon_greedy_policy(self.QFunc, self.epsilon, len(self.env.n_actions))


    def QFunc(self, s,p,c):
        q0 = 0
        q1 = 0  # buy open
        q2 = 0  # sell close
        q3 = 0  # hold long
        q4 = 0  # sell open
        q5 = 0  # buy close
        q6 = 0  # hold short

        s=s[::-1]

        self.time_since=0
        self.escape_count=0
        if self.time_since % 4==0 and self.time_since >0:
            self.escape_count+=1

        n_in = c[0]
        n_out = c[1]

        n_in_buy_offset=c[2]
        n_in_sell_offset = c[3]

        n_out_buy_offset=c[4]
        n_out_sell_offset = c[5]

        insig = ''.join(s[0:n_in])
        outsig = ''.join(s[0:n_out])

        print('in:',insig,'out:',outsig)



        if p==0.0:
            q0=1

            if insig.count('c') == 0 and insig.count('a') >= n_in - n_in_buy_offset:
                q1=1
                q0=0
                self.time_since+=1
            if insig.count('a')==0 and insig.count('c') >= n_in - n_in_sell_offset:
                q4=1
                q0=0
                self.time_since += 1

        if p==1.0:
            q0 =0
            if outsig.count('a')==0 and outsig.count('c') >= n_out-n_out_sell_offset:
                if self.escape_count>0:
                    self.escape_count-=1
                else:
                    q2=1
                    self.time_since = 0
            else:
                q3=1
                self.time_since += 1
        if p==-1.0:
            q0=0
            if outsig.count('c')==0 and outsig.count('a') >= n_out- n_out_buy_offset:
                if self.escape_count > 0:
                    self.escape_count -= 1
                else:
                    q5=1
                    self.time_since =0
            else:
                q6=1
                self.time_since += 1



        q=[q0, q1, q2, q3, q4, q5, q6]

        return q

    def norm_it(self,d):
        d=d.flatten()
        d=d[::-1]
        x = np.diff(d)
        return x #/ np.linalg.norm(x)

    def get_state(self,data):
        #using Kalman filter to get the true value/price in a minute observations, open,high,low,close prices
        d = np.log(np.array([data.get_t_data(-i)[2:6] for i in range(self.config.observation_window)]))
        mean = d.mean()
        kf = KalmanFilter(initial_state_mean=mean, n_dim_obs=4)
        v = kf.em(d)
        h=self.norm_it(v.smooth(d)[0])

        s = h.std()
        h = h / s
        o = ''
        for i in range(len(h)):
            if h[i] > 0.43:
                o = o + 'a '
            elif h[i] < -.43:
                o = o + 'c '
            else:
                o = o + 'b '

        return o.split()

    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Args:
            Q: A function returns a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """

        def policy_fn(observation,p):
            A = np.ones(nA, dtype=float) * epsilon / nA
            q=Q(observation,p)

            best_action = np.argmax(q)
            print("action called:",self.env.action_labels[best_action])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn



    def train(self, steps, eventSource):
        render = False
        self.env.random_past_day()
        num_game, self.update_count, ep_reward = 0,0,0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        ep_rewards, actions = [], []
        t = 0

        self.theta = np.zeros([self.config.observation_window-1, len(self.env.n_actions)], dtype=np.float32)
        self.zetha = np.ones([self.config.observation_window-1, len(self.env.n_actions)], dtype=np.float32)
        self.discount = 0.99

        s = self.get_state(self.env.data)

        action_probs = self.policy(s, self.env.position)

        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        while action not in self.env.get_valid_actions():
            print('invalid action called.',action)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for self.i in tqdm(range(self.i, steps)):


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
                self.close_attempts += 1
                if (self.env.position > 0.):  # a long position
                    if action != self.env.action_labels.index("sell_close"):  # close long
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

            print('action taken:',self.env.action_labels[action])

            s_prime= self.get_state(self.env.data)

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
            #if render:
                #sleep(.2)
                #eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts, forecast_history)

            #delta = self.env.reward + self.QFunc(s_prime,self.theta)[action] - self.QFunc(s,self.theta)[action]
            #print('delta:',delta)
            # a = np.zeros([len(self.env.n_actions)])
            # a[action] = self.q_learning_rate

            #self.theta = self.theta + self.q_learning_rate * delta * self.zetha
            #print('theta 1:', self.theta )

            rsum = np.absolute(self.theta).sum(axis=1)
            rsum[rsum == 0] = 1
            self.theta = self.theta / rsum[:, None]
            #print('theta 2:', self.theta)
            #self.zetha = self.discount * self.zetha + np.array(s)[:, None]
            #print('zetha:', self.zetha)

            if action in ( self.env.action_labels.index("stay_neutral"),
                           self.env.action_labels.index("sell_close"),
                           self.env.action_labels.index("buy_close")):
                #reset eligibility trace
                self.zetha = np.ones([self.config.observation_window - 1, len(self.env.n_actions)], dtype=np.float32)

            action_probs = self.policy(s_prime,self.env.position)
            #print('s:', s_prime)
            #print('p:',self.env.position)
            #print(action_probs)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            while action not in self.env.get_valid_actions():
                print('invalid action called.', action,action_probs)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


            if self.env.terminal:
                t = 0
                self.close_attempts = 0
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
                    avg_loss = self.total_loss / self.config.test_step
                    avg_q = self.total_q / self.config.test_step

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
                        #'learning_rate': self.net.learning_rate,
                        'ep_rewards': ep_rewards,
                        'ep_actions': actions
                    }

                    print('log to tensorboard at:', self.i)
                    #self.net.inject_summary(sum_dict, self.i)
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
            if self.i % 2000 == 0:
                j = 0
                render = True

            if render:
                #self.env_wrapper.env.render()
                j += 1
                if j == 1000:
                    render = False


    def play(self, episodes, eventSource):
        #self.net.restore_session()

        self.env.random_past_day()

        i = 0
        #for _ in range(self.config.history_len):
        #    self.history.add(self.env.data)
        episode_steps = 0
        while i < episodes:
            s = self.get_state(self.env.data)
            action_probs = self.policy(s, self.env.position)

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            while action not in self.env.get_valid_actions():
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            self.env.step(action)
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

            if action not in (self.env.action_labels.index("stay_neutral"),
                              self.env.action_labels.index("hold_long"),
                              self.env.action_labels.index("hold_short")):
                if action in (self.env.action_labels.index("buy_open"), self.env.action_labels.index("sell_open")):
                    pl = - self.env.open_cost
                if action == self.env.action_labels.index("sell_close"):
                    pl = self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost

                if action == self.env.action_labels.index("buy_close"):
                    pl = -1 * self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost

                self.log_trade(self.env.action_labels[action], self.env.today, self.env.time, self.env.unit,
                               self.env.order_price,
                               self.env.current_price, pl)

                self.account_profit_loss += pl

            forecasts = np.zeros(self.forecast_window, dtype=np.float32)
            forecast_history = np.zeros(self.forecast_window, dtype=np.float32)
            sleep(1)
            eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts,
                                         forecast_history)

            episode_steps += 1
            if episode_steps > self.config.max_steps:
                self.env.terminal = True
            if self.env.terminal:
                episode_steps = 0
                i += 1
                self.env.random_past_day()

