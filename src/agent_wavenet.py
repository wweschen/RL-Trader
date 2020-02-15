from base_agent  import BaseAgent
#from src.history import History
from replay_memory import DQNReplayMemory
from networks.wavenet import WaveNet
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from time import sleep
import  os
import logging
import pprint as pp
import  tensorflow as tf
from tf_utils import shape,save_pkl,load_pkl
from pykalman import KalmanFilter
from search_model import SearchModel

class WaveNetAgent(BaseAgent):

    def __init__(self, config, environment):
        super(WaveNetAgent, self).__init__(config,environment)
        #self.history = History(config)
        # self.replay_memory = DQNReplayMemory(config)
        self.net = WaveNet(  config)
        self.net.build()

        self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

        self.account_profit_loss = 0.
        self.forecast_window=config.forecast_window
        self.close_attempts = 0
        self.q_learning_rate=0.01

        #self.policy = self.make_epsilon_greedy_policy(self.dyna2_Q, self.epsilon, len(self.env.n_actions))

        self.policy_hat = self.make_epsilon_greedy_policy(self.dyna2_Q_hat, self.epsilon, len(self.env.n_actions))

        self.init_logging(self.net.dir_log)
        logging.info('all parameters:')
        logging.info(pp.pformat([(var.name, shape(var)) for var in tf.global_variables()]))

        logging.info('trainable parameters:')
        logging.info(pp.pformat([(var.name, shape(var)) for var in tf.trainable_variables()]))

        logging.info('trainable parameter count:')
        logging.info(str(np.sum(np.prod(shape(var)) for var in tf.trainable_variables())))

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

        def policy_fn(observation, actions):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q(observation, actions))
            #print("best_action:",self.env.action_labels[best_action])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn


    def norm_it(self,d):
        x = np.diff(d.flatten())
        return x / np.linalg.norm(x)

    def dyna2_Q(self, s,actions):
        q= np.einsum('ji,i->j', np.transpose(self.theta), s)

        n = len(self.env.n_actions)

        onehot = np.eye(n)*[1 if i in actions else 0 for i in range(n)]
        q=np.matmul(onehot,q)
        #print('S:', s)
        #print('Q:',q)
        return q

    def dyna2_Q_hat(self, s,  actions):

        q=np.einsum('ji,i->j', np.transpose(self.theta), s) + np.einsum('ji,i->j', np.transpose(self.theta_hat), s)
        n = len(self.env.n_actions)

        onehot = np.eye(n) * [1 if i in actions else 0 for i in range(n)]
        q = np.matmul(onehot, q)
        # print('S_hat:', s)
        # print('Q_hat:',q)
        return q

    def dyna2_search(self, state, forecasts, position, order_price, current_price):
        self.zetha_hat.fill(0.)
        i = 0


        search_model = SearchModel(state,forecasts, current_price, self.env.open_cost, self.env.unit, self.env.t)

        valid_actions =  search_model.get_valid_actions()
        s=state
        action_probs = self.policy_hat(s,valid_actions)
        # print(action_probs)

        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        while action not in valid_actions:
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        while i < len(forecasts):


            s_prime, position, order_price, current_price, reward, terminal, valid_actions  =    search_model.step(action)

            i += 1

            delta = reward + self.dyna2_Q_hat(s_prime,valid_actions)[action] - self.dyna2_Q_hat(s,valid_actions)[ action]


            self.theta_hat = self.theta_hat + self.q_learning_rate* delta * self.zetha_hat

            rsum = np.absolute(self.theta_hat).sum(axis=1)
            rsum[rsum == 0] = 1

            self.theta_hat = self.theta_hat / rsum[:, None]

            self.zetha_hat = self.discount * self.zetha_hat + np.array(s)[:, None]

            action_probs = self.policy_hat(s_prime,valid_actions)

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            while action not in valid_actions:
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            s = s_prime

    def get_state(self,data,position,order_price):
        #using Kalman filter to get the true value/price in a minute observations, open,high,low,close prices
        d = np.array([data.get_t_data(-i)[2:6] for i in range(self.config.forecast_window)])

        d0 = np.array([data.get_t_data(-i)[2:6] for i in range(self.config.price_data_size)])

        mean = d0.mean()

        kf = KalmanFilter(em_vars=['transition_covariance', 'observation_covariance'], initial_state_mean=mean,
                          n_dim_obs=4)
        v = kf.em(d)

        h = v.smooth(d)[0]

        h= np.append([position,order_price],h)

        return h



    def predict(self):

        forecasts = self.net.predict(self.env.data, self.env.today)

        return forecasts

    def observe(self):

        self.past_forecasts, self.targets, loss = self.net.train(self.env.data, self.env.today)

        # print('Targets:',self.targets)
        self.total_loss += loss
        self.update_count += 1

    def train(self, steps, eventSource):
        render = False

        self.load()

        self.env.random_past_day()
        num_game, self.update_count, ep_reward = 0,0,0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        ep_rewards, actions = [], []
        t = 0

        self.theta = np.zeros([self.config.forecast_window+2, len(self.env.n_actions)], dtype=np.float32)
        self.zetha = np.ones([self.config.forecast_window+2, len(self.env.n_actions)], dtype=np.float32)
        self.theta_hat = np.zeros([self.config.forecast_window+2, len(self.env.n_actions)], dtype=np.float32)
        self.zetha_hat = np.ones([self.config.forecast_window+2, len(self.env.n_actions)], dtype=np.float32)
        self.discount = 0.99
        self.q_learning_rate = 0.01

        self.theta = load_pkl(self.model_dir + "theta.pkl")
        if self.theta is None:
            self.theta = np.zeros([self.config.forecast_window + 2, len(self.env.n_actions)], dtype=np.float32)

        valid_actions= self.env.get_valid_actions()
        s = self.get_state(self.env.data,self.env.position,self.env.order_price)
        action_probs = self.policy_hat(s,valid_actions)

        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        while action not in valid_actions:
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for self.i in tqdm(range(self.i, steps)):

            # 1. predict

            self.forecasts = self.predict()

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
            #print('action taken:', self.env.action_labels[action])

            self.observe()

            s_prime= self.get_state(self.env.data,self.env.position,self.env.order_price)

            valid_actions_prime = self.env.get_valid_actions()


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



            #if render:
            #     #sleep(0.2)
            # print('forecasts:',self.forecasts)
            # print('past forecasts:', self.past_forecasts)
            eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss,
                                             self.forecasts, self.past_forecasts,self.targets)



            self.dyna2_search(s_prime,  self.forecasts, self.env.position, self.env.order_price,
                              self.env.current_price)

            # print(self.dyna2_Q_hat(s_prime)[self.action])

            delta = self.env.reward + self.dyna2_Q(s_prime,valid_actions_prime)[action] \
                                                - self.dyna2_Q(s,valid_actions)[action]
             #print('delta:',delta)
            # a = np.zeros([len(self.env.n_actions)])
            # a[action] = self.q_learning_rate
            # print('delt:', delta)
            # print('zetha:', self.zetha)
            # print('theta:', self.theta)

            self.theta = self.theta + self.q_learning_rate * delta * self.zetha
            #print('theta 1:', self.theta )

            rsum = np.absolute(self.theta).sum(axis=1)
            rsum[rsum == 0] = 1
            self.theta = self.theta / rsum[:, None]
            #print('theta 2:', self.theta)
            # print('S:',s)
            # print('S prime:', s_prime)
            self.zetha = self.discount * self.zetha + np.array(s)[:, None]
            #print('zetha:', self.zetha)

            if action in ( self.env.action_labels.index("stay_neutral"),
                           self.env.action_labels.index("sell_close"),
                           self.env.action_labels.index("buy_close")):
                #reset eligibility trace
                self.zetha = np.ones([self.config.forecast_window+2, len(self.env.n_actions)], dtype=np.float32)

            action_probs = self.policy_hat(s_prime,valid_actions_prime)

            s=s_prime
            valid_actions=valid_actions_prime

            #print(action_probs)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            while action not in self.env.get_valid_actions():
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
            if self.i % 5000 == 0 and self.i > 0:
                j = 0
                print('Saving the parameters at:',self.i)
                self.save()
                save_pkl(self.theta, self.model_dir + "theta.pkl")
            if self.i % 5000 == 0:
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
            action_probs = self.policy(s, self.theta)

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
            #sleep(1)
            #eventSource.data_signal.emit(self.env.data, self.env.position, self.account_profit_loss, forecasts,
            #                             forecast_history)

            episode_steps += 1
            if episode_steps > self.config.max_steps:
                self.env.terminal = True
            if self.env.terminal:
                episode_steps = 0
                i += 1
                self.env.random_past_day()

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