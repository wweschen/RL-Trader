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
from wavelets import modwt

class KSWsearchAgent(BaseAgent):

    def __init__(self, config, environment):
        super(KSWsearchAgent, self).__init__(config,environment)
        #self.history = History(config)
        # self.replay_memory = DQNReplayMemory(config)
        #self.net = DQN(len(environment.n_actions), config)
        #self.net.build()

        #self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

        self.account_profit_loss = 0.
        self.forecast_window=config.forecast_window
        self.close_attempts = 0
        self.q_learning_rate=0.01
        self.steps = 0
        self.previous_flag=0

        self.policy = self.make_epsilon_greedy_policy(self.QFunc, self.epsilon, len(self.env.n_actions))

    def QFunc(self, s,  op, cp, std):
        q0 = 0  # stay neutral
        q1 = 0  # buy open
        q2 = 0  # sell close
        q3 = 0  # hold long
        q4 = 0  # sell open
        q5 = 0  # buy close
        q6 = 0  # hold short

        current_flag = s[-1]
        action_flag  =self.previous_flag * current_flag




        out=''

        if self.env.position == 0:
            q0 = 1
            # forced outed so reset these position counters
            self.buy_out_escape_count = 0
            self.sell_out_escape_count = 0
            self.time_since = 0

            if action_flag > 0.20 and current_flag > 0 :
                q1 = 1
                q0 = 0
                self.floater = cp
                self.time_since += 1
                out = 'threshold:{:0.2f},std:{}'.format(action_flag, std)
            if action_flag > 0.20  and current_flag < 0 :#(action_flag< 0. and current_flag < 0) or (self.previous_flag < 0 and current_flag < 0) :
                q4 = 1
                q0 = 0
                self.floater = cp
                self.time_since += 1
                out = 'threshold:{:0.2f},std:{}'.format(action_flag, std)

        if self.env.position == 1:
            q0 = 0
            if action_flag<= 0.0 :# or (self.previous_flag < 0 and current_flag < 0) : # signal_window.count(1)>0 and window.count(-1)==0:
                q2 = 1  # sell out
                self.time_since = 0
                self.sell_out_escape_count = 0
                self.floater = 0
                self.sigma = self.sigma_max
                out = 'threshold:{:0.2f},std:{}'.format(action_flag, std)
            else:
                q3 = 1
                self.time_since += 1

        if self.env.position == -1:
            q0 = 0

            if action_flag<=0.0 :#or  (self.previous_flag > 0 and current_flag > 0) :
                q5 = 1  # buy out
                self.time_since = 0
                self.buy_out_escape_count = 0
                self.floater = 0
                self.sigma = self.sigma_max
                out='threshold:{:0.2f},std:{}'.format(action_flag,std)
            else:
                q6 = 1
                self.time_since += 1

        print('signal:', self.previous_flag, current_flag, action_flag, self.env.position,  std, out)

        self.previous_flag = current_flag

        q = [q0, q1, q2, q3, q4, q5, q6]



        return q, out

    def norm_it(self, d):
        d = d.flatten()
        d = d[::-1]
        x = np.diff(d)
        return x  # / np.linalg.norm(x)

    def get_state(self, data):

        self.steps += 1

        if self.steps < 16:
            fetch = 16
        else:
            fetch = self.steps
        if self.steps > 32:
            fetch = 32

        d = np.log(np.array([self.env.data.get_t_data(-i)[2:6] for i in range(fetch)]))

        mean = d.mean()
        kf = KalmanFilter(initial_state_mean=mean, n_dim_obs=4)
        v = kf.em(d)
        p = v.smooth(d)[0]
        d = p.flatten()
        d = d[::-1]

        s = d.std()

        vo = []
        for i in range(fetch):
            vo.append([self.env.data.get_t_data(-i)[3] - self.env.data.get_t_data(-i)[4]])

        vo = np.log(np.array(vo[::-1])+1)

        s='{:0.6f}'.format(s )

        di = []
        for i in range(fetch):
            di.append([self.env.data.get_t_data(-i)[5] - self.env.data.get_t_data(-i)[2]])
        di = np.array(di[::-1])

        signal=di/np.linalg.norm(di)


        x5 = []
        n5 = int(fetch / 16)
        m5 = fetch % 16

        print(n5, m5)
        aa = signal[0:m5]
        y5 = modwt(aa, 'db2', 5)[5]
        nd5 = np.linalg.norm(y5)
        if nd5 != 0:
            y5 = 2 * y5 / nd5
        x5 = x5 + list(y5)

        for i in range(n5):
            aa = signal[m5 + i * 16:m5 + (i + 1) * 16]
            y5 = modwt(aa, 'db2', 5)[5]
            nd5 = np.linalg.norm(y5)
            if nd5 != 0:
                y5 = 2 * y5 / nd5
            x5 = x5 + list(y5)


        return  x5,s

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

        def policy_fn(observation, op, cp, std):
            # greedy for now
            # A = np.ones(nA, dtype=float) * epsilon / nA
            q,cause = Q(observation,   op, cp, std)

            best_action = np.argmax(q)
            #print("action called:", self.env.action_labels[best_action], q)
            # A[best_action] += (1.0 - epsilon)
            return best_action,cause

        return policy_fn

    def train(self, steps, eventSource):
        render = False

        #best days
        #aday='08/07/2008'
        #aday = '06/13/2006'
        # aday = '07/24/2008'
        # aday = '03/26/2009'
        #aday ='06/15/2016'
        #aday = '09/09/2015'
        #aday = '06/29/2011'

        #mid negative
        #  aday='05/13/2015'
        #aday ='08/25/2014'
        #aday= '11/03/2011'
        #aday='12/14/2016'
        #aday='03/05/2004'


        #wrost days
        #aday='06/10/2013'
        #aday= '04/27/2011'
        #aday='03/07/2008'
        #aday = '07/16/2008'
        #aday = '11/01/2007'
        #aday = '06/27/2007'
        #aday = '03/09/2006'
        aday='10/24/2012'

        self.env.a_given_past_day(aday)


        # #for self.i in tqdm(range(self.i, steps)):
        #
        # #grid search for the parameters to get a profit trading day:
        # max_in_SAX_length=6
        # min_in_SAX_length=4
        #
        # max_out_SAX_length=6
        # min_out_SAX_length =4
        #
        # for n_in in range(min_in_SAX_length,max_in_SAX_length+1): # SAX string length used to decide to get in
        #     for n_out in range(min_out_SAX_length,max_out_SAX_length+1): # SAX string length used to decide to get out
        #         for n_buy_in in range(min_in_SAX_length,n_in+1): #count of buy in signal character a
        #             for n_sell_out  in range(min_out_SAX_length,n_out+1): #count of sell out signal character c
        #                 for n_sell_in in range(min_in_SAX_length,n_in+1):  #count of sell in signal character c
        #                     for n_buy_out  in range(min_out_SAX_length,n_out+1): #count of buy out signal character a

        self.env.position = 0

        self.time_since = 0
        self.a_count_since = 0
        self.c_count_since = 0
        self.sell_out_escape_count = 0
        self.buy_out_escape_count = 0
        self.floater = 0.
        self.sigma = 32.
        self.sigma_max = 32.
        self.sigma_min = 2

        method='kalman-sax-wavelets-sigma32'

        s, std = self.get_state(self.env.data)
        cause =''

        action, cause = self.policy(s,  self.env.order_price, self.env.current_price, std)

        for self.i in tqdm(range(391)):

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
                            cause = 'forced'
                if (self.env.position < 0.):  # a short position
                    if action != self.env.action_labels.index("buy_close"):  # close a short
                        if self.close_attempts > 10:
                            action = self.env.action_labels.index("buy_close")
                            cause = 'forced'
                if (self.env.position == 0.):
                    action = self.env.action_labels.index("stay_neutral")
                if self.close_attempts > 10:
                    if (self.env.position > 0.):  # a long position
                        action = self.env.action_labels.index("sell_close")  # close long
                        cause = 'forced'
                    if (self.env.position < 0.):  # a long position
                        action = self.env.action_labels.index("buy_close")  # close short
                        cause='forced'
            # Beginning of day logic, don't trade the first fifteen minutes
            fifteen_minute = timedelta(minutes=16)
            start_time = datetime.strptime("9:30", '%H:%M')
            # print(end_time, self.time, end_time - self.time)
            if self.env.time - start_time <= fifteen_minute:
                action = self.env.action_labels.index("stay_neutral")


            self.env.step(action)

            print('action taken:',self.env.action_labels[action])

            s_prime,std= self.get_state(self.env.data)

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
                               self.env.current_price,pl,cause)

                self.account_profit_loss += pl
            #
            # forecasts = np.zeros(self.forecast_window, dtype=np.float32)
            # forecast_history = np.zeros(self.forecast_window, dtype=np.float32)
            #
            # sleep(1.2)
            # eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts, forecast_history)


            action,cause  = self.policy(s_prime,  self.env.order_price, self.env.current_price, std)



            if self.env.terminal:

                self.log_search_result(self.env.today, method, self.account_profit_loss)

                self.log_trade(method, self.env.today, None, 0,0,0, self.account_profit_loss, '')

                self.i = 0
                self.close_attempts = 0
                #self.env.a_given_past_day(aday)
                self.account_profit_loss = 0

    # def play(self, episodes, eventSource):
    #     #self.net.restore_session()
    #
    #     self.env.random_past_day()
    #
    #     i = 0
    #     #for _ in range(self.config.history_len):
    #     #    self.history.add(self.env.data)
    #     episode_steps = 0
    #     while i < episodes:
    #         s = self.get_state(self.env.data)
    #         action_probs = self.policy(s, self.env.position)
    #
    #         action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    #         while action not in self.env.get_valid_actions():
    #             action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    #
    #
    #         # end of day logic
    #         # if it is close to the end of day, we need to try to close out our position on a good term,
    #         # not to wait to be forced to close at the end of day, which is a fast rule of this algorithm
    #         # here is a logic, 10 allowances once within self.forecast_window minutes of closing minute, close on any change of the first nine
    #         # if the action needed agrees with the prediction, which is to close, execute it and then stay neutral the rest of day
    #         # or forcefully close the position at 10th allowance then stay neutral.
    #         grace_period = timedelta(minutes=15)
    #         end_time = datetime.strptime("16:00", '%H:%M')
    #         # print(end_time, self.time, end_time - self.time)
    #         if end_time - self.env.time < grace_period:
    #             if (self.env.position > 0.):  # a long position
    #                 if action != self.env.action_labels.index("sell_close"):  # close long
    #                     self.close_attempts += 1
    #                     if self.close_attempts > 10:
    #                         action = self.env.action_labels.index("sell_close")
    #             if (self.env.position < 0.):  # a short position
    #                 if action != self.env.action_labels.index("buy_close"):  # close a short
    #                     if self.close_attempts > 10:
    #                         action = self.env.action_labels.index("buy_close")
    #             if (self.env.position == 0.):
    #                 action = self.env.action_labels.index("stay_neutral")
    #             if self.close_attempts > 10:
    #                 if (self.env.position > 0.):  # a long position
    #                     action = self.env.action_labels.index("sell_close")  # close long
    #                 if (self.env.position < 0.):  # a long position
    #                     action = self.env.action_labels.index("buy_close")  # close short
    #
    #         # Beginning of day logic, don't trade the first fifteen minutes
    #         fifteen_minute = timedelta(minutes=15)
    #         start_time = datetime.strptime("9:30", '%H:%M')
    #         # print(end_time, self.time, end_time - self.time)
    #         if self.env.time - start_time <= fifteen_minute:
    #             action = self.env.action_labels.index("stay_neutral")
    #
    #         self.env.step(action)
    #
    #         if action not in (self.env.action_labels.index("stay_neutral"),
    #                           self.env.action_labels.index("hold_long"),
    #                           self.env.action_labels.index("hold_short")):
    #             if action in (self.env.action_labels.index("buy_open"), self.env.action_labels.index("sell_open")):
    #                 pl = - self.env.open_cost
    #             if action == self.env.action_labels.index("sell_close"):
    #                 pl = self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost
    #
    #             if action == self.env.action_labels.index("buy_close"):
    #                 pl = -1 * self.env.unit * (self.env.current_price - self.env.order_price) - self.env.open_cost
    #
    #             self.log_trade(self.env.action_labels[action], self.env.today, self.env.time, self.env.unit,
    #                            self.env.order_price,
    #                            self.env.current_price, pl)
    #
    #             self.account_profit_loss += pl
    #
    #         forecasts = np.zeros(self.forecast_window, dtype=np.float32)
    #         forecast_history = np.zeros(self.forecast_window, dtype=np.float32)
    #         sleep(1)
    #         eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts,
    #                                      forecast_history)
    #
    #         episode_steps += 1
    #         if episode_steps > self.config.max_steps:
    #             self.env.terminal = True
    #         if self.env.terminal:
    #             episode_steps = 0
    #             i += 1
    #             self.env.random_past_day()

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
                '{},{},{:0.2f}\n'.format(sdate, c, pl))
