from __future__ import print_function
import logging
import os
import pprint as pp
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from functools import reduce
from emulator import PriceData
from base import BaseModel
from datetime import datetime, timedelta
from tf_utils import (time_distributed_dense_layer, temporal_convolution_layer,
    sequence_mean, sequence_smape, shape)

from utils import  save_pkl, load_pkl,pprint_dict
from time import sleep
from search_model import SearchModel
from collections import defaultdict

def QFunc():
    return np.zeros(3)

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)

    self.sess = sess
    self.env = environment

    self.weight_dir = 'weights'
    self.account_profit_loss=0.

    self.residual_channels = 32
    self.skip_channels = 32
    self.dilations = [2 ** i for i in range(4)]
    self.filter_widths = [2 for i in range(4)]
    self.num_decode_steps = self.forecast_window

    self.init_logging(self.log_dir)

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_wavenet()

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

      def policy_fn(observation):
          A = np.ones(nA, dtype=float) * epsilon / nA
          best_action = np.argmax(Q(observation))
          A[best_action] += (1.0 - epsilon)
          return A

      return policy_fn



  def init_logging(self, log_dir):
      if not os.path.isdir(log_dir):
          os.makedirs(log_dir)

      date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
      log_file = 'log_{}.txt'.format(date_str)

      #reload(logging)  # bad
      logging.basicConfig(
          filename=os.path.join(log_dir, log_file),
          level=logging.INFO,
          format='[[%(asctime)s]] %(message)s',
          datefmt='%m/%d/%Y %I:%M:%S %p'
      )
      logging.getLogger().addHandler(logging.StreamHandler())

  def get_features(self,targets,past_forecasts,forecasts,position,order_price,current_price):

      list=[]
      error = ( targets[0][-1] -  past_forecasts[0][-1]) / targets[0][-1]
      list.append(order_price)
      list.append(position)
      list.append( t/current_price for t in targets[0])
      list.append(1)
      list.append(forecasts[0][0])
      list.append(error)

      return np.array(list)

  def dyna2_Q(self,s):
    return np.einsum('ji,i->j', np.transpose(self.theta), s)

  def dyna2_Q_hat(self, s ):

    return np.einsum('ji,i->j', np.transpose(self.theta),s) + np.einsum('ji,i->j', np.transpose(self.theta_hat),  s)

  def dyna2_search(self,targets,past_forecasts,forecasts,position,order_price,current_price):
    #state: op,po,pt_14,pt_13,pt_12,pt_11,pt_10,pt_9,pt_8,pt_7,pt_6,pt_5,pt_4,pt_3,pt_2,pt_1,pt,pt1,error
    self.zetha_hat.fill(0.)
    i = 0
    valid_actions=self.search_model.get_valid_actions()
    s = [(order_price - current_price) / current_price, position / self.env.unit]
    s += [(t - current_price) / current_price for t in targets[0][0:15]]
    s += [(f - current_price) / current_price for f in forecasts[0][0:2]]

    action_probs = self.dyna_policy_hat(s)
    # print(action_probs)


    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    while action not in valid_actions:
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    while i<13:
      s = [(order_price-current_price)/current_price, position/self.env.unit]
      s += [(t-current_price) / current_price for t in targets[0][i:15]]
      s +=  [(f-current_price) /current_price for f in forecasts[0][0:i+2]]
      #s += [(targets[0][-1] - past_forecasts[0][-1]) / targets[0][-1]]



      data, position, order_price, current_price, reward,terminal, valid_actions =self.search_model.step(action)

      i += 1

      s_prime=[(order_price-current_price)/current_price,position/self.env.unit]
      s_prime += [(t-current_price)/current_price for t in targets[0][i:15]]
      s_prime += [(f-current_price) /current_price for f in forecasts[0][0:i+2]]
      #s_prime += [( targets[0][-1] -  past_forecasts[0][-1]) / targets[0][-1]]

      delta= reward+self.dyna2_Q_hat(s_prime)[self.action] - self.dyna2_Q_hat(s)[self.action]
      a = np.zeros([3])

      a[action] = self.dyna_learning_rate

      self.theta_hat=self.theta_hat+a *  delta * self.zetha_hat

      rsum = np.absolute(self.theta_hat).sum(axis=1)
      rsum[rsum == 0] = 1
      self.theta_hat = self.theta_hat / rsum[:, None]

      self.zetha_hat = self.dyna_discount*self.zetha_hat+np.array(s)[:,None]

      action_probs = self.dyna_policy_hat(s_prime)
      # print(action_probs)

      # mask = np.array([0, 0, 0])
      # mask[valid_actions] = 1
      # action_probs = action_probs * mask

      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      while action not in valid_actions:
          action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

  def train(self,eventSource):

    self.sess.run(self.init)
    self.load_model()
    start_step = self.step_op.eval()
    start_time = time.time()
    self.momentum=0
    self.prev_momentum =0
    self.acceleration =0
    self.potential =0
    self.prev_potential =0
    self.potential_momentum =0

    epsilon = 0.1

    #Dyna-2 implementation
    self.theta = np.zeros([5,3],dtype=np.float32)
    #self.theta_hat = np.zeros([5, 3], dtype=np.float32)

    # self.features_sample = np.zeros([19], dtype=np.float32)
    self.zetha = np.zeros([5,3],dtype=np.float32)
    # self.zetha_hat = np.zeros([19,3],dtype=np.float32)
    self.dyna_discount = 0.99
    self.dyna_learning_rate = 0.01
    #
    self.dyna_policy = self.make_epsilon_greedy_policy(self.dyna2_Q,epsilon,3)
    # self.dyna_policy_hat = self.make_epsilon_greedy_policy(self.dyna2_Q_hat, epsilon, 3)
    #
    self.theta = load_pkl(self.model_dir + "theta.pkl")

    if self.theta is None:
      self.theta = np.zeros([5, 3], dtype=np.float32)

    # self.Q =  load_pkl( self.model_dir+"Q.pkl")
    #
    # if self.Q is None:
    #   self.Q=defaultdict(QFunc)
    #
    # pprint_dict(self.Q)


    # The policy we're following
    #self.policy = self.make_epsilon_greedy_policy(self.Q, epsilon, 3)

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []
    self.past_forecasts = None
    self.close_attempts=0
    self.data, self.today, self.time, self.position,self.order_price, self.current_price, reward, terminal, self.actions_allowed = self.env.random_past_day()
    self.step=start_step
    self.observe(self.data, self.today, self.position, reward, terminal, self.actions_allowed)

    #self.theta_hat.fill(0.)
    self.zetha.fill(0.)

    self.forecasts = np.zeros(self.forecast_window, dtype=np.float32)
    forecast_history = np.zeros(self.forecast_window, dtype=np.float32)
    eventSource.data_signal.emit(self.data, self.position,  self.account_profit_loss, self.forecasts, forecast_history)  # <- Here you emit a signal!

    s  = np.zeros([5], dtype=np.float32).tolist()

    action_probs = self.dyna_policy(s)
    #print('action probs:',action_probs)
    self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    while self.action not in self.actions_allowed:
      self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)


    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):

      num_game, self.update_count, ep_reward = 0, 0, 0.
      total_reward, self.total_loss, self.total_q = 0., 0., 0.
      ep_rewards, actions = [], []

      # 1. predict

      self.forecasts = self.predict(self.data)
      #self.forecasts = self.predict_point_symtry(self.data)

      # if self.current_price !=0.:
      #   s = [(self.order_price-self.current_price)/self.current_price, self.position/self.env.unit] + \
      #       [(t-self.current_price) / self.current_price for t in self.targets[0][:-1]]
      #   s.append(0)
      #   s.append((self.forecasts[0][0]-self.current_price) / self.current_price)
      #   s.append((self.forecasts[0][1]-self.current_price) / self.current_price)
      #   #s.append((self.targets[0][-1] - self.past_forecasts[0][-1]) / self.targets[0][-1])

      # print(self.forecasts,self.past_forecasts)
      if self.momentum !=0:
          self.prev_momentum= self.momentum

      self.momentum= int(100* (np.asarray(self.forecasts[0]).flatten()  - np.asarray(self.past_forecasts[0]).flatten() ).mean())
      self.acceleration =self.momentum-self.prev_momentum
      self.prev_potential = self.potential
      self.potential =  (int((self.position / 10) * (self.current_price - self.order_price)))

      self.potential_momentum = self.potential - self.prev_potential

      self.state = (self.position/self.env.unit, self.momentum, self.acceleration,self.potential,self.potential_momentum )



      #end of day logic
      #if it is close to the end of day, we need to try to close out our position on a good term,
      # not to wait to be forced to close at the end of day, which is a fast rule of this algorithm
      #here is a logic, 10 allowances once within self.forecast_window minutes of closing minute, close on any change of the first nine
      # if the action needed agrees with the prediction, which is to close, execute it and then stay neutral the rest of day
      # or forcefully close the position at 10th allowance then stay neutral.
      grace_period = timedelta(minutes=15)
      end_time = datetime.strptime("16:00", '%H:%M')
      #print(end_time, self.time, end_time - self.time)
      if end_time - self.time < grace_period:
        if(self.position >0.): #a long position
          if self.action != 2: #close long
            self.close_attempts += 1
            if self.close_attempts>10:
               self.action = 2
        if(self.position<0.): #a short position
            if self.action !=1: #close a short
              if self.close_attempts > 10:
                self.action = 1
        if(self.position == 0.):
            self.action =0
        if self.close_attempts>10:
          if (self.position > 0.):  # a long position
             self.action  = 2  # close long
          if (self.position < 0.):  # a long position
               self.action = 1 # close long

      #Beginning of day logic, don't trade the first fifteen minutes
      fifteen_minute = timedelta(minutes=15)
      start_time = datetime.strptime("9:30", '%H:%M')
      # print(end_time, self.time, end_time - self.time)
      if self.time -start_time <= fifteen_minute:
            self.action = 0

        # 2. act
      self.data, self.today, self.time, self.position,self.order_price, self.current_price, reward, terminal, self.actions_allowed  = self.env.step(self.action)
      #print('step:', self.today,  ' ' , self.time, ' ' , self.data.dates[-1], ' ' ,  self.data.times[-1])
      # 3. observe
      self.observe(self.data, self.today, self.position, reward, terminal, self.actions_allowed )

      # we have a shift causing by unknown at this point (learning rate too big?), so hopefully the following can correct it meanwhile.
      shift= self.targets[0][0]-self.past_forecasts[0][0]
      self.forecasts=self.forecasts+shift
      self.past_forecasts=self.past_forecasts+shift




      if self.action in (1,2 ):
          direction = 1 if self.action==2 else -1
          pl=direction*self.env.unit*(self.current_price - self.order_price)-self.env.open_cost
          self.log_trade(self.action, self.today,self.time,self.env.unit, self.order_price,self.current_price, pl )
          self.account_profit_loss += pl

          #print('\nState traded:',s)

      eventSource.data_signal.emit(self.data, self.position, self.account_profit_loss, self.forecasts[0],
                                   self.past_forecasts[0])  # <- Here you emit a signal!

      #dyna learning
      # self.search_model = SearchModel(self.forecasts[0], self.order_price, self.position, self.env.open_cost,
      #                                 self.env.unit,self.env.t)
      #
      # s_prime =  [(self.order_price-self.current_price)/self.current_price, self.position/self.env.unit]\
      #            +[ (t-self.current_price) / self.current_price  for t in self.targets[0][:-1]]
      # s_prime.append(0)
      # s_prime.append((self.forecasts[0][0]-self.current_price)/self.current_price)
      # s_prime.append((self.forecasts[0][1]-self.current_price)/ self.current_price)
      # #s_prime.append((self.targets[0][-1] - self.past_forecasts[0][-1]) / self.targets[0][-1])
      #
      # self.dyna2_search(self.targets,self.past_forecasts,self.forecasts,self.position,self.order_price,self.current_price)
      #
      # #print(self.dyna2_Q_hat(s_prime)[self.action])
      #
      # delta = reward + self.dyna2_Q_hat(s_prime)[self.action] - self.dyna2_Q_hat(s)[self.action]
      # a= np.zeros([3])
      # a[self.action]=self.dyna_learning_rate
      # self.theta = self.theta + a* delta * self.zetha
      #
      # rsum=np.absolute(self.theta).sum(axis=1)
      # rsum[rsum == 0] = 1
      # self.theta= self.theta / rsum[:, None]
      #
      # self.zetha= self.dyna_discount * self.zetha + np.array(s)[:,None]
      # action_probs = self.dyna_policy_hat(s_prime)
      #
      # self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      # while self.action not in self.actions_allowed:
      #   self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      # if self.action != 0:
      #   print('\ntheta:', self.theta)
      #   print('\nzetha:', self.zetha)
      #   print('\naction_probs:', action_probs)

      # # TD Update
      # # print(self.forecasts,self.past_forecasts)

      if self.momentum != 0:
        self.prev_momentum = self.momentum

      self.momentum = int( 100 * (np.asarray(self.forecasts[0]).flatten() - np.asarray(self.past_forecasts[0]).flatten()).mean())
      self.prev_potential =self.potential
      self.potential =  (int((self.position / 10) * (self.current_price - self.order_price)))

      self.potential_momentum =self.potential - self.prev_potential

      self.acceleration = self.momentum - self.prev_momentum


      self.next_state = (self.position/self.env.unit, self.momentum, self.acceleration,self.potential,self.potential_momentum )

      delta = reward + self.dyna2_Q(self.next_state)[self.action] - self.dyna2_Q(s)[self.action]

      a= np.zeros([3])
      a[self.action]=self.dyna_learning_rate
      self.theta = self.theta + a* delta * self.zetha

      rsum=np.absolute(self.theta).sum(axis=1)
      rsum[rsum == 0] = 1
      self.theta= self.theta / rsum[:, None]

      self.zetha= self.dyna_discount * self.zetha + np.array(self.state)[:,None]

      action_probs = self.dyna_policy(self.next_state)

      self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      while self.action not in self.actions_allowed:
        self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      if self.action != 0:
        print('\ntheta:', self.theta)
        print('\nzetha:', self.zetha)
        print('\naction_probs:', action_probs)

      sleep(0.5)

      if terminal:
        self.data,  self.today, self.time, self.position, self.order_price, self.current_price,reward, terminal, actions  = self.env.random_past_day()
        forecasts =np.zeros(self.forecast_window, dtype=np.float32)
        forecast_history = np.zeros(self.forecast_window, dtype=np.float32)
        self.close_attempts=0
        eventSource.data_signal.emit(self.data, self.position,  self.account_profit_loss,  forecasts,
                                     forecast_history)  # <- Here you emit a signal!

        self.state  = np.zeros([5], dtype=np.float32).tolist()
        #
        #
        # self.theta_hat.fill(0.)
        # self.zetha.fill(0.)
        #
        action_probs = self.dyna_policy(self.state)

        self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        while self.action not in self.actions_allowed:
          self.action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(self.action)
      total_reward += reward

      #if self.step >= self.learn_start:
      if self.step % 391 ==390:
        avg_reward = total_reward / self.test_step
        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        try:
          max_ep_reward = np.max(ep_rewards)
          min_ep_reward = np.min(ep_rewards)
          avg_ep_reward = np.mean(ep_rewards)
        except:
          max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

        print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
            % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))

        if max_avg_ep_reward * 0.9 <= avg_ep_reward:
          self.step_assign_op.eval({self.step_input: self.step + 1})
          self.save_model(self.step + 1)

          save_pkl(self.theta,self.model_dir+"theta.pkl")

          max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

        if self.step > 390:
          self.inject_summary({
              'average.reward': avg_reward,
              'average.loss': avg_loss,
              'average.q': avg_q,
              'episode.max reward': max_ep_reward,
              'episode.min reward': min_ep_reward,
              'episode.avg reward': avg_ep_reward,
              'episode.num of game': num_game,
              'episode.rewards': ep_rewards,
              'episode.actions': actions,
              'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
            }, self.step)

          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []
  def get_input_sequences(self):
    self.open_encode = tf.placeholder(tf.float32, [None, None])
    self.high_encode = tf.placeholder(tf.float32, [None, None])
    self.low_encode = tf.placeholder(tf.float32, [None, None])
    self.close_encode = tf.placeholder(tf.float32, [None, None])
    self.volume_encode = tf.placeholder(tf.float32, [None, None])
    self.is_today = tf.placeholder(tf.int32, [None, None])
    self.encode_len = tf.placeholder(tf.int32, [None])
    self.position = tf.placeholder(tf.int32,[None]) # -1 short position, 0 neutral, +1 long position
    self.order_price = tf.placeholder(tf.float32,[None])
    self.est_current_price= tf.placeholder(tf.float32,[None])
    self.time_since_open = tf.placeholder(tf.int32,[None])
    #we can not have a position open overnight, so it can not be more than 390 minutes in theory.
    # But in reality, we won't be holding a position too long, so maybe let's say one hour max, 60 max

    #self.symbol = tf.placeholder(tf.int32, [None])
    #self.day = tf.placeholder(tf.int32, [None])

    self.y_decode = tf.placeholder(tf.float32, [None, self.num_decode_steps])
    self.decode_len = tf.placeholder(tf.int32, [None])


    self.keep_prob = tf.placeholder(tf.float32)
    self.is_training = tf.placeholder(tf.bool)

    self.log_x_encode_mean = sequence_mean(tf.log((self.high_encode+self.low_encode)/2. + 1), self.encode_len)
    self.log_x_encode = self.transform((self.high_encode+self.low_encode)/2.,self.log_x_encode_mean)
    self.log_x_target_mean = sequence_mean(tf.log((self.high_encode[-self.forecast_window:] + self.low_encode[-self.forecast_window:]) / 2. + 1), self.decode_len)

    self.log_open_encode_mean= sequence_mean(tf.log(self.open_encode + 1), self.encode_len)
    self.log_open_encode = self.transform(self.open_encode,self.log_open_encode_mean )

    self.log_high_encode_mean = sequence_mean(tf.log(self.high_encode + 1), self.encode_len)
    self.log_high_encode = self.transform(self.high_encode,self.log_high_encode_mean)

    self.log_low_encode_mean = sequence_mean(tf.log(self.low_encode + 1), self.encode_len)
    self.log_low_encode = self.transform(self.low_encode,self.log_low_encode_mean)

    self.log_close_encode_mean = sequence_mean(tf.log(self.close_encode + 1), self.encode_len)
    self.log_close_encode = self.transform(self.close_encode,self.log_close_encode_mean)

    self.log_volume_encode_mean = sequence_mean(tf.log(self.volume_encode + 1), self.encode_len)
    self.log_volume_encode = self.transform(self.volume_encode,self.log_volume_encode_mean)

    self.position = tf.placeholder(tf.int32, [None])

    self.log_order_price = tf.log(self.order_price + 1)- self.log_x_encode_mean

    self.log_est_current_price  =  tf.log(self.est_current_price + 1)- self.log_x_encode_mean


    self.x = tf.expand_dims(self.log_x_encode, 2)

    self.encode_features = tf.concat([
      tf.expand_dims( self.log_open_encode , 2),
      tf.expand_dims( self.log_high_encode , 2),
      tf.expand_dims( self.log_low_encode , 2),
      tf.expand_dims( self.log_close_encode , 2),
      tf.expand_dims( self.log_volume_encode , 2),

      tf.tile(tf.expand_dims(tf.one_hot(self.position+1, 3), 1), (1, tf.shape(self.open_encode)[1], 1)),
      tf.tile(tf.expand_dims(tf.one_hot(self.time_since_open, 60), 1), (1, tf.shape(self.open_encode)[1], 1)),

      tf.expand_dims( tf.cast(self.is_today,tf.float32) , 2),

      tf.tile(tf.reshape(self.log_open_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
      tf.tile(tf.reshape(self.log_high_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
      tf.tile(tf.reshape(self.log_low_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
      tf.tile(tf.reshape(self.log_close_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
      tf.tile(tf.reshape(self.log_volume_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
      tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
      tf.tile(tf.reshape(self.log_order_price, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
      tf.tile(tf.reshape(self.log_est_current_price, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),

    ], axis=2)


    return self.x

  def predict_point_symtry(self,data):
      self.targets = [(data.highs[-self.forecast_window:] + data.lows[-self.forecast_window:]) / 2.]
      self.current_price = self.targets[0][-1]
      d = self.current_price -np.array(self.targets[0])
      forecasts =np.array([self.current_price+d[14-i] for i in range(0,15)])
      #print(self.targets,forecasts)
      return [forecasts]

  def predict(self, data,  test_ep=None):
    # first we need to predict self.forecast_window mimutes of average prices via WaveNet
    # then we will predict an action to take with Q function approximator
    self.stoday = datetime.strftime(self.today, '%m/%d/%Y')
    is_today=[date ==self.stoday for date in data.dates]

    feed_dict={
        self.open_encode :[ data.opens]  ,
        self.high_encode:[ data.highs],
        self.low_encode :[ data.lows],
        self.close_encode : [ data.closes],
        self.volume_encode :[ data.volumes],
        self.encode_len : [200-self.forecast_window],
        self.is_today:[is_today],
        self.y_decode :[np.zeros([self.forecast_window], dtype=np.float32)],
        self.decode_len : [self.forecast_window],
        self.learning_rate_step: self.step,
      }

    forecasts = self.sess.run(
      fetches = self.preds,
      feed_dict=feed_dict
    )



    return  forecasts

  def observe(self, data, date, position, reward, terminal, actions ):
    #reward = max(self.min_reward, min(self.max_reward, reward))

    # self.history.add(screen)
    # self.memory.add(screen, reward, action, terminal)
    self.stoday = datetime.strftime(self.today, '%m/%d/%Y')
    is_today = [date == self.stoday for date in data.dates]

    feed_dict = {
      self.open_encode: [data.opens[:-self.forecast_window]],
      self.high_encode: [data.highs[:-self.forecast_window]],
      self.low_encode: [data.lows[:-self.forecast_window]],
      self.close_encode: [data.closes[:-self.forecast_window]],
      self.volume_encode: [data.volumes[:-self.forecast_window]],
      self.encode_len: [200 - 2 * self.forecast_window],
      self.is_today: [is_today[:-self.forecast_window]],
      self.y_decode: [(data.highs[-self.forecast_window:] + data.lows[-self.forecast_window:]) / 2.],
      self.decode_len: [self.forecast_window],
      self.learning_rate_step: self.step,
    }

    self.past_forecasts, self.targets, loss, _ = self.sess.run(
      fetches=[self.preds, self.labels, self.loss, self.optim],
      feed_dict=feed_dict
    )

    self.total_loss += loss
    # self.total_q += q_t.mean()
    self.update_count += 1

    # if self.step > self.learn_start:
    #   if self.step % self.train_frequency == 0:
    #     self.learning_mini_batch()

  def learning_mini_batch(self,data):
    # if self.memory.count < self.history_length:
    #   return
    # else:
    #   s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

    t = time.time()

    # price forecasting


    #Q-learing
    # pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
    #
    # q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
    #   self.target_s_t: s_t_plus_1,
    #   self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
    # })
    #
    # target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
    #
    # _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
    #   self.target_q_t: target_q_t,
    #   self.action: action,
    #   self.s_t: s_t,
    #   self.learning_rate_step: self.step,
    # })

    #self.writer.add_summary(summary_str, self.step)
    # self.total_loss += loss
    # self.total_q += q_t.mean()
    # self.update_count += 1

  def transform(self, x,mean):
    return tf.log(x + 1) - tf.expand_dims(tf.log(mean), 1)

  def inverse_transform(self, x,mean):
    return tf.exp(x + tf.expand_dims(tf.log(mean), 1)) - 1

  def update_parameters(self, loss):

    if self.regularization_constant != 0:
      l2_norm = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(param))) for param in tf.trainable_variables()])
      loss = loss + self.regularization_constant * l2_norm

    self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
    self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                       tf.train.exponential_decay(
                                         self.learning_rate,
                                         self.learning_rate_step,
                                         self.learning_rate_decay_step,
                                         self.learning_rate_decay,
                                         staircase=True))

    optimizer = self.get_optimizer(self.learning_rate_op)

    grads = optimizer.compute_gradients(loss)

    clipped = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) for g, v_ in grads]

    optim = optimizer.apply_gradients(clipped, global_step=self.global_step)

    if self.enable_parameter_averaging:
      maintain_averages_op = self.ema.apply(tf.trainable_variables())
      with tf.control_dependencies([optim]):
        self.optim = tf.group(maintain_averages_op)
    else:
      self.optim = optim


    logging.info('all parameters:')
    logging.info(pp.pformat([(var.name, shape(var)) for var in tf.global_variables()]))

    logging.info('trainable parameters:')
    logging.info(pp.pformat([(var.name, shape(var)) for var in tf.trainable_variables()]))

    logging.info('trainable parameter count:')
    logging.info(str(np.sum(np.prod(shape(var)) for var in tf.trainable_variables())))

  def get_optimizer(self, learning_rate):
    if self.optimizer == 'adam':
      return tf.train.AdamOptimizer(learning_rate)
    elif self.optimizer == 'gd':
      return tf.train.GradientDescentOptimizer(learning_rate)
    elif self.optimizer == 'rms':
      return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9)
    else:
      assert False, 'optimizer must be adam, gd, or rms'

  def encode(self, x, features):
    x = tf.concat([x, features], axis=2)

    inputs = time_distributed_dense_layer(
      inputs=x,
      output_units=self.residual_channels,
      activation=tf.nn.tanh,
      scope='x-proj-encode',
      reuse = tf.AUTO_REUSE
    )

    skip_outputs = []
    conv_inputs = [inputs]
    for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
      dilated_conv = temporal_convolution_layer(
        inputs=inputs,
        output_units=2 * self.residual_channels,
        convolution_width=filter_width,
        causal=True,
        dilation_rate=[dilation],
        scope='dilated-conv-encode-{}'.format(i),
        reuse = tf.AUTO_REUSE
      )
      conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
      dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

      outputs = time_distributed_dense_layer(
        inputs=dilated_conv,
        output_units=self.skip_channels + self.residual_channels,
        scope='dilated-conv-proj-encode-{}'.format(i),
        reuse = tf.AUTO_REUSE
      )
      skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

      inputs += residuals
      conv_inputs.append(inputs)
      skip_outputs.append(skips)

    skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))

    h = time_distributed_dense_layer(skip_outputs, 128,
                                    scope='dense-encode-1',
                                     activation=tf.nn.relu,
                                     reuse = tf.AUTO_REUSE)
    y_hat = time_distributed_dense_layer( h, 1,
                                          scope='dense-encode-2',
                                          reuse = tf.AUTO_REUSE)

    return y_hat, conv_inputs[:-1]

  def initialize_decode_params(self, x, features):
    x = tf.concat([x, features], axis=2)

    inputs = time_distributed_dense_layer(
      inputs=x,
      output_units=self.residual_channels,
      activation=tf.nn.tanh,
      scope='x-proj-decode',
        reuse = tf.AUTO_REUSE
    )

    skip_outputs = []
    conv_inputs = [inputs]
    for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
      dilated_conv = temporal_convolution_layer(
        inputs=inputs,
        output_units=2 * self.residual_channels,
        convolution_width=filter_width,
        causal=True,
        dilation_rate=[dilation],
        scope='dilated-conv-decode-{}'.format(i),
        reuse = tf.AUTO_REUSE
      )
      conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
      dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

      outputs = time_distributed_dense_layer(
        inputs=dilated_conv,
        output_units=self.skip_channels + self.residual_channels,
        scope='dilated-conv-proj-decode-{}'.format(i),
        reuse = tf.AUTO_REUSE
      )
      skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

      inputs += residuals
      conv_inputs.append(inputs)
      skip_outputs.append(skips)

    skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
    h = time_distributed_dense_layer(skip_outputs, 128, scope='dense-decode-1', activation=tf.nn.relu)
    y_hat = time_distributed_dense_layer(h, 1, scope='dense-decode-2')
    return y_hat

  def decode(self, x, conv_inputs, features):
    batch_size = tf.shape(x)[0]

    # initialize state tensor arrays
    state_queues = []
    for i, (conv_input, dilation) in enumerate(zip(conv_inputs, self.dilations)):
      batch_idx = tf.range(batch_size)
      batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
      batch_idx = tf.reshape(batch_idx, [-1])

      queue_begin_time = self.encode_len - dilation - 1
      temporal_idx = tf.expand_dims(queue_begin_time, 1) + tf.expand_dims(tf.range(dilation), 0)
      temporal_idx = tf.reshape(temporal_idx, [-1])

      idx = tf.stack([batch_idx, temporal_idx], axis=1)
      slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, shape(conv_input, 2)))

      layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation + self.num_decode_steps)
      layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
      state_queues.append(layer_ta)

    # initialize feature tensor array
    features_ta = tf.TensorArray(dtype=tf.float32, size=self.num_decode_steps)
    features_ta = features_ta.unstack(tf.transpose(features, (1, 0, 2)))

    # initialize output tensor array
    emit_ta = tf.TensorArray(size=self.num_decode_steps, dtype=tf.float32)

    # initialize other loop vars
    elements_finished = 0 >= self.decode_len
    time = tf.constant(0, dtype=tf.int32)

    # get initial x input
    current_idx = tf.stack([tf.range(tf.shape(self.encode_len)[0]), self.encode_len - 1], axis=1)
    initial_input = tf.gather_nd(x, current_idx)

    def loop_fn(time, current_input, queues):
      current_features = features_ta.read(time)
      current_input = tf.concat([current_input, current_features], axis=1)

      with tf.variable_scope('x-proj-decode', reuse = tf.AUTO_REUSE):
        w_x_proj = tf.get_variable('weights')
        b_x_proj = tf.get_variable('biases')
        x_proj = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

      skip_outputs, updated_queues = [], []
      for i, (conv_input, queue, dilation) in enumerate(zip(conv_inputs, queues, self.dilations)):
        state = queue.read(time)
        with tf.variable_scope('dilated-conv-decode-{}'.format(i), reuse = tf.AUTO_REUSE):
          w_conv = tf.get_variable('weights'.format(i))
          b_conv = tf.get_variable('biases'.format(i))
          dilated_conv = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(x_proj, w_conv[1, :, :]) + b_conv
        conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
        dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

        with tf.variable_scope('dilated-conv-proj-decode-{}'.format(i), reuse = tf.AUTO_REUSE):
          w_proj = tf.get_variable('weights'.format(i))
          b_proj = tf.get_variable('biases'.format(i))
          concat_outputs = tf.matmul(dilated_conv, w_proj) + b_proj
        skips, residuals = tf.split(concat_outputs, [self.skip_channels, self.residual_channels], axis=1)

        x_proj += residuals
        skip_outputs.append(skips)
        updated_queues.append(queue.write(time + dilation, x_proj))

      skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
      with tf.variable_scope('dense-decode-1', reuse = tf.AUTO_REUSE):
        w_h = tf.get_variable('weights')
        b_h = tf.get_variable('biases')
        h = tf.nn.relu(tf.matmul(skip_outputs, w_h) + b_h)

      with tf.variable_scope('dense-decode-2', reuse = tf.AUTO_REUSE):
        w_y = tf.get_variable('weights')
        b_y = tf.get_variable('biases')
        y_hat = tf.matmul(h, w_y) + b_y

      elements_finished = (time >= self.decode_len)
      finished = tf.reduce_all(elements_finished)

      next_input = tf.cond(
        finished,
        lambda: tf.zeros([batch_size, 1], dtype=tf.float32),
        lambda: y_hat
      )
      next_elements_finished = (time >= self.decode_len - 1)

      return (next_elements_finished, next_input, updated_queues)

    def condition(unused_time, elements_finished, *_):
      return tf.logical_not(tf.reduce_all(elements_finished))

    def body(time, elements_finished, emit_ta, *state_queues):
      (next_finished, emit_output, state_queues) = loop_fn(time, initial_input, state_queues)

      emit = tf.where(elements_finished, tf.zeros_like(emit_output), emit_output)
      emit_ta = emit_ta.write(time, emit)

      elements_finished = tf.logical_or(elements_finished, next_finished)
      return [time + 1, elements_finished, emit_ta] + list(state_queues)

    returned = tf.while_loop(
      cond=condition,
      body=body,
      loop_vars=[time, elements_finished, emit_ta] + state_queues
    )

    outputs_ta = returned[2]
    y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))
    return y_hat




  def build_wavenet(self):

    # training network
    self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
    self.global_step = tf.Variable(0, trainable=False)
    self.learning_rate_var = tf.Variable(0.0, trainable=False)

    x = self.get_input_sequences()

    y_hat_encode, conv_inputs = self.encode(x, features=self.encode_features)
    self.initialize_decode_params(x, features=self.decode_features)
    y_hat_decode = self.decode(y_hat_encode, conv_inputs, features=self.decode_features)

    x1 = tf.squeeze(x,2)
    col = tf.shape(x1)[1]
    row = tf.shape(x1)[0]
    tail =  (tf.slice(x1, [0, col - 1], [row, 1]))

    y_hat_decode = self.inverse_transform(tf.squeeze(y_hat_decode, 2),self.log_x_encode_mean)
    y_hat_decode = tf.nn.relu(y_hat_decode)

    self.labels = self.y_decode
    self.preds = y_hat_decode
    self.loss = sequence_smape(self.labels, self.preds, self.decode_len )

    self.update_parameters(self.loss)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q',
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar("%s" % tag, self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

      self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

    self.init = tf.global_variables_initializer()


  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def play(self, callback, n_step=10000, n_episode=100, test_ep=None, render=False):
    # if test_ep == None:
    #   test_ep = self.ep_end
    #
    # test_history = History(self.config)
    #
    # if not self.display:
    #   gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
    #   self.env.env.monitor.start(gym_dir)
    #
    # best_reward, best_idx = 0, 0
    # for idx in xrange(n_episode):
    #   screen, reward, action, terminal = self.env.new_random_game()
    #   current_reward = 0
    #
    #   for _ in range(self.history_length):
    #     test_history.add(screen)
    #
    #   for t in tqdm(range(n_step), ncols=70):
    #     # 1. predict
    #     action = self.predict(test_history.get(), test_ep)
    #     # 2. act
    #     screen, reward, terminal = self.env.act(action, is_training=False)
    #     # 3. observe
    #     test_history.add(screen)
    #
    #     current_reward += reward
    #     if terminal:
    #       break
    #
    #   if current_reward > best_reward:
    #     best_reward = current_reward
    #     best_idx = idx
    #
    #   print("="*self.forecast_window)
    #   print(" [%d] Best reward : %d" % (best_idx, best_reward))
    #   print("="*self.forecast_window)

    if not self.display:
      self.env.env.monitor.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
