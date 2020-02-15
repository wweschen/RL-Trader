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

class KSAgent(BaseAgent):

    def __init__(self, config, environment):
        super(KSAgent, self).__init__(config,environment)
        #self.history = History(config)
        # self.replay_memory = DQNReplayMemory(config)
        #self.net = DQN(len(environment.n_actions), config)
        #self.net.build()

        #self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

        self.account_profit_loss = 0.
        self.previous_profit_loss =0.
        self.forecast_window=config.forecast_window
        self.close_attempts = 0
        self.q_learning_rate=0.01

        self.policy = self.make_epsilon_greedy_policy(self.QFunc, self.epsilon, len(self.env.n_actions))



    def QFunc(self, s,c,op,cp,std):
        q0 = 0  # stay neutral
        q1 = 0  # buy open
        q2 = 0  # sell close
        q3 = 0  # hold long
        q4 = 0  # sell open
        q5 = 0  # buy close
        q6 = 0  # hold short

        s = s[::-1]

        n_in = c[0]
        n_out = c[1]

        n_buy_in = c[2]
        n_sell_out = c[3]

        n_sell_in=c[4]
        n_buy_out = c[5]

        insig = ''.join(s[0:n_in])
        outsig = ''.join(s[0:n_out])

        out=''

        if self.env.position==0:
            q0=1
            #forced outed so reset these position counters
            self.buy_out_escape_count=0
            self.sell_out_escape_count=0
            self.time_since = 0

            if insig.count('c') == 0 and insig.count('a') >= n_buy_in:
                q1=1
                q0=0
                self.floater = cp
                self.time_since += 1
            if insig.count('a') == 0 and insig.count('c') >= n_sell_in:
                q4=1
                q0=0
                self.floater = cp
                self.time_since += 1

        if  self.env.position == 1:
            q0 =0
            if cp>self.floater:
                self.floater =cp

            if self.time_since % n_buy_in == 0 and self.time_since > 0:
                self.time_since = 0
                self.sell_out_escape_count += 1
                self.sigma = self.sigma_min+(self.sigma_max-self.sigma_min) * np.exp(
                    -0.693*self.sell_out_escape_count / 8)

            if self.floater>=np.exp(std*self.sigma)*cp:
                q2 = 1  # sell out
                self.time_since = 0
                self.sell_out_escape_count = 0
                self.floater = 0
                self.sigma = self.sigma_max
                out = '*********float'
            else:
                if outsig.count('a') ==0 and outsig.count('c') >= n_sell_out:
                    if self.sell_out_escape_count > 0:
                        q3 = 1
                        self.sell_out_escape_count =0
                    else:
                        q2 = 1 #sell out
                        out='*********pattern'
                        self.time_since = 0
                        self.sell_out_escape_count =0
                        self.floater = 0
                        self.sigma = self.sigma_max
                else:
                    q3=1
                    self.time_since += 1

        if  self.env.position == -1:
            q0=0

            if cp < self.floater:
                self.floater = cp

            if self.time_since % n_sell_in == 0 and self.time_since > 0:
                self.time_since = 0
                self.buy_out_escape_count += 1
                self.sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * np.exp(
                    -0.693 * self.buy_out_escape_count/8)

            if cp >= np.exp(std * self.sigma) * self.floater:
                q5 = 1  # buy out
                self.time_since = 0
                self.buy_out_escape_count = 0
                self.floater = 0
                self.sigma=self.sigma_max

                out = '*********float'
            else:
                if outsig.count('c') == 0 and outsig.count('a') >= n_buy_out:
                    if self.buy_out_escape_count > 0:
                        q6 = 1
                        self.buy_out_escape_count = 0

                    else:
                        q5 = 1 #buy out
                        self.time_since =0
                        self.floater =0
                        self.sigma = self.sigma_max
                        self.buy_out_escape_count = 0
                        out = '********pattern'

                else:
                    q6=1
                    self.time_since += 1

        q=[q0, q1, q2, q3, q4, q5, q6]

        print('in:', insig, 'out:', outsig, self.env.position, self.time_since, self.sell_out_escape_count,
              self.buy_out_escape_count, round(op,2),round(cp,2),round(self.floater,2),self.sigma, np.exp(std * self.sigma), out)

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

        return o.split(),s

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

        def policy_fn(observation,c,op,cp,std):
            #greedy for now
            #A = np.ones(nA, dtype=float) * epsilon / nA
            q=Q(observation,c,op,cp,std)

            best_action = np.argmax(q)
            print("action called:",self.env.action_labels[best_action],q)
            #A[best_action] += (1.0 - epsilon)
            return best_action

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


        self.env.position = 0

        self.time_since = 0
        self.a_count_since = 0
        self.c_count_since = 0
        self.sell_out_escape_count = 0
        self.buy_out_escape_count = 0
        self.floater =0.
        self.sigma = 16.
        self.sigma_max = 16.
        self.sigma_min = 2

        c=[5,5,4,5,4,5]
        s,std = self.get_state(self.env.data)

        action = self.policy(s,c, self.env.order_price,self.env.current_price,std)

        for self.i in tqdm(range(self.config.train_steps)):

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
            fifteen_minute = timedelta(minutes=5)
            start_time = datetime.strptime("9:30", '%H:%M')
            # print(end_time, self.time, end_time - self.time)
            if self.env.time - start_time <= fifteen_minute:
                action = self.env.action_labels.index("stay_neutral")

            self.env.step(action)

            print('action taken:', self.env.action_labels[action])

            s_prime,std = self.get_state(self.env.data)

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

            sleep(.2)
            eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts, forecast_history)


            action = self.policy(s_prime, c,self.env.order_price,self.env.current_price,std)

            if self.env.terminal:
                self.time_since = 0
                self.a_count_since = 0
                self.c_count_since = 0
                self.sell_out_escape_count = 0
                self.buy_out_escape_count = 0

                self.log_trade_summary(self.env.today, c, self.account_profit_loss-self.previous_profit_loss)

                self.log_trade(''.join(str(i) for i in c), self.env.today, None, 0, 0, 0, self.account_profit_loss-self.previous_profit_loss)

                self.close_attempts = 0
                self.previous_profit_loss = self.account_profit_loss
                self.env.random_past_day()

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
    def log_trade_summary(self,day,c,pl):
        import errno
        sdate = day.strftime('%Y-%m-%d')

        path = os.path.join(self.config.env_name, 'trade_summary',  "trade_summary.csv")
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        if os.path.exists(path):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        with open(path, append_write) as f:
            f.write(
                '{},{:1d},{:1d},{:1d},{:1d},{:1d},{:1d},{:0.2f}\n'.format(sdate, c[0], c[1], c[2], c[3], c[4], c[5],
                                                                          pl))

    def log_search_result(self,day,c,pl):
        import errno
        sdate = day.strftime('%Y-%m-%d')

        path = os.path.join(self.config.env_name, 'search_logs', sdate, "search_log.csv")
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        if os.path.exists(path):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        with open(path, append_write) as f:
            f.write(
                '{},{:1d},{:1d},{:1d},{:1d},{:1d},{:1d},{:0.2f}\n'.format(sdate, c[0], c[1],c[2],c[3],c[4],c[5],pl))
