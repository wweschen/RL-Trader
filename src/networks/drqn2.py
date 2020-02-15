import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug
from networks.base import  BaseModel

from datetime import datetime, timedelta

from tf_utils import (time_distributed_dense_layer, temporal_convolution_layer,
    sequence_mean,transform, fully_connected_layer, lstm_layer, sequence_smape, shape,stateful_lstm,huber_loss)



class DRQN2(BaseModel):

    def __init__(self, n_actions, config):
        super(DRQN2, self).__init__(config, "drqn2")
        self.n_actions = n_actions
        self.history_len = config.history_len
        self.cnn_format = config.cnn_format
 

        self.num_lstm_layers = config.num_lstm_layers
        self.lstm_size = config.lstm_size
        self.min_history = config.min_history
        self.states_to_update = config.states_to_update


        self.residual_channels =config.residual_channels
        self.skip_channels = config.skip_channels
        self.dilations = config.dilations
        self.filter_widths = config.filter_widths
        self.batch_size=config.batch_size
        self.series_length=config.observation_window


    def train_on_batch_target(self, opens, highs, lows,closes, volumes, todays,
            actions, rewards,terminals, dates,positions,
            order_prices, current_prices, time_since, steps):



        q, loss = np.zeros((self.batch_size, self.n_actions)), 0

        opens = np.transpose(opens, [1, 0, 2])
        highs = np.transpose(highs, [1, 0, 2])
        lows= np.transpose(lows, [1, 0, 2])
        closes= np.transpose(closes, [1, 0, 2])
        volumes= np.transpose(volumes, [1, 0, 2])
        todays= np.transpose(todays, [1, 0, 2])

        actions = np.transpose(actions, [1, 0])
        rewards = np.transpose(rewards, [1, 0])
        terminal = np.transpose(terminals, [1, 0])
        dates =np.transpose(dates, [1, 0])
        positions=np.transpose(positions, [1, 0 ])
        order_prices =np.transpose(order_prices, [1, 0 ])
        current_prices=np.transpose(current_prices, [1, 0 ])
        time_since=np.transpose(time_since, [1, 0 ])


        #states = np.reshape(states, [states.shape[0], states.shape[1], 1, states.shape[2], states.shape[3]])

        lstm_state_c, lstm_state_h = self.initial_zero_state_batch, self.initial_zero_state_batch
        lstm_state_target_c, lstm_state_target_h = self.sess.run(
            [self.state_output_target_c, self.state_output_target_h],
            {
                self.opens_: opens[0],
                self.highs_: highs[0],
                self.lows_: lows[0],
                self.closes_: closes[0],
                self.volumes_: volumes[0],
                self.todays_: todays[0],
                self.positions_: positions[0],
                self.order_prices_: order_prices[0],
                self.current_prices_: current_prices[0],
                self.time_since_:time_since[0],

                self.lengths: [self.series_length for i in range(self.batch_size)],

                self.c_state_target: self.initial_zero_state_batch,
                self.h_state_target: self.initial_zero_state_batch
            }
        )
        for i in range(self.min_history):
            j = i + 1
            lstm_state_c, lstm_state_h, lstm_state_target_c, lstm_state_target_h = self.sess.run(
                [self.state_output_c, self.state_output_h, self.state_output_target_c, self.state_output_target_h],
                {
                    self.opens: opens[i],
                    self.highs: highs[i],
                    self.lows: lows[i],
                    self.closes: closes[i],
                    self.volumes: volumes[i],
                    self.todays: todays[i],
                    self.positions: positions[i],
                    self.order_prices: order_prices[i],
                    self.current_prices: current_prices[i],
                    self.time_since: time_since[i],

                    self.opens_: opens[j],
                    self.highs_: highs[j],
                    self.lows_: lows[j],
                    self.closes_: closes[j],
                    self.volumes_: volumes[j],
                    self.todays_: todays[j],
                    self.positions_: positions[j],
                    self.order_prices_: order_prices[j],
                    self.current_prices_: current_prices[j],
                    self.time_since_: time_since[j],

                    self.lengths: [self.series_length for i in range(self.batch_size)],

                    self.c_state_target: lstm_state_target_c,
                    self.h_state_target: lstm_state_target_h,
                    self.c_state_train: lstm_state_c,
                    self.h_state_train: lstm_state_h
                }
            )
        for i in range(self.min_history, self.min_history + self.states_to_update):
            j = i + 1
            target_val, lstm_state_target_c, lstm_state_target_h = self.sess.run(
                [self.q_target_out, self.state_output_target_c, self.state_output_target_h],
                {
                    self.opens_: opens[j],
                    self.highs_: highs[j],
                    self.lows_: lows[j],
                    self.closes_: closes[j],
                    self.volumes_: volumes[j],
                    self.todays_: todays[j],
                    self.positions_: positions[j],
                    self.order_prices_: order_prices[j],
                    self.current_prices_: current_prices[j],
                    self.time_since_: time_since[j],
                    self.lengths: [self.series_length for i in range(self.batch_size)],


                    self.c_state_target: lstm_state_target_c,
                    self.h_state_target: lstm_state_target_h
                }
            )
            max_target = np.max(target_val, axis=1)
            target = (1. - terminal[i]) * self.gamma * max_target + rewards[i]

            _, q_, train_loss_, q_summary, lstm_state_c, lstm_state_h = self.sess.run(
                [self.train_op, self.q_out, self.loss,self.avg_q_summary, self.state_output_c, self.state_output_h],
                feed_dict={
                    self.opens: opens[i],
                    self.highs: highs[i],
                    self.lows: lows[i],
                    self.closes: closes[i],
                    self.volumes: volumes[i],
                    self.todays: todays[i],
                    self.positions: positions[i],
                    self.order_prices: order_prices[i],
                    self.current_prices: current_prices[i],
                    self.time_since: time_since[i],
                    self.lengths : [self.series_length for i in range(self.batch_size)],

                    self.c_state_train: lstm_state_c,
                    self.h_state_train: lstm_state_h,
                    self.action: actions[i],
                    self.target_val: target,
                    self.lr: self.learning_rate
                }
            )
            q += q_
            loss += train_loss_



        if self.train_steps % 5000 == 0:
            self.file_writer.add_summary(q_summary, steps)
            #self.file_writer.add_summary(merged_imgs, steps)

        if steps % 20000 == 0 and steps > 50000:
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum:
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1
        return q.mean(), loss / (self.states_to_update)



    def add_placeholders(self):
        self.w = {}
        self.w_target = {}

        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")
        # create placeholder to fill in lstm state
        self.c_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_c")
        self.h_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_h")
        self.lstm_state_train = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_train, self.h_state_train)

        self.c_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_c")
        self.h_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_h")
        self.lstm_state_target = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_target, self.h_state_target)

        # initial zero state to be used when starting episode
        self.initial_zero_state_batch = np.zeros((self.batch_size, self.lstm_size))
        self.initial_zero_state_single = np.zeros((1, self.lstm_size))

        self.initial_zero_complete = np.zeros((self.num_lstm_layers, 2, self.batch_size, self.lstm_size))

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.terminal = tf.placeholder(dtype=tf.float32, shape=[None], name="terminal")

        self.target_val = tf.placeholder(dtype=tf.float32, shape=[None], name="target_val")


        self.learning_rate_step = tf.placeholder("int64", None, name="learning_rate_step")

        self.opens = tf.placeholder(tf.float32, [None, self.series_length])
        self.highs = tf.placeholder(tf.float32, [None,self.series_length])
        self.lows = tf.placeholder(tf.float32, [None, self.series_length])
        self.closes = tf.placeholder(tf.float32, [None, self.series_length])
        self.volumes = tf.placeholder(tf.float32, [None, self.series_length])
        self.todays = tf.placeholder(tf.int32, [None, self.series_length]) 
        self.positions = tf.placeholder(tf.int32, [None])  # -1 short positions, 0 neutral, +1 long positions
        self.order_prices = tf.placeholder(tf.float32, [None])
        self.current_prices = tf.placeholder(tf.float32, [None])
        self.time_since = tf.placeholder(tf.int32, [None])

        self.opens_ = tf.placeholder(tf.float32, [None, self.series_length])
        self.highs_ = tf.placeholder(tf.float32, [None, self.series_length])
        self.lows_ = tf.placeholder(tf.float32, [None, self.series_length])
        self.closes_ = tf.placeholder(tf.float32, [None, self.series_length])
        self.volumes_ = tf.placeholder(tf.float32, [None, self.series_length])
        self.todays_ = tf.placeholder(tf.int32, [None, self.series_length])
        self.positions_ = tf.placeholder(tf.int32, [None])  # -1 short positions, 0 neutral, +1 long positions
        self.order_prices_ = tf.placeholder(tf.float32, [None])
        self.current_prices_ = tf.placeholder(tf.float32, [None])
        self.time_since_ = tf.placeholder(tf.int32, [None])

        self.lengths = tf.placeholder(tf.int32,[None])


        # we can not have a positions open overnight, so it can not be more than 390 minutes in theory.
        # But in reality, we won't be holding a positions too long, so maybe let's say one hour max, 60 max

        # self.symbol = tf.placeholder(tf.int32, [None])
        # self.day = tf.placeholder(tf.int32, [None])


    def get_inputs(self,opens,highs,lows,closes,volumes,positions,order_prices,current_prices,time_since,todays):

        log_x_mean = sequence_mean(tf.log((highs + lows) / 2. + 1), self.series_length)
        log_x = transform((highs + lows) / 2., log_x_mean)

        log_opens_mean = sequence_mean(tf.log(opens + 1), self.series_length)
        log_opens = transform(opens, log_opens_mean)

        log_highs_mean = sequence_mean(tf.log(highs + 1), self.series_length)
        log_highs = transform(highs, log_highs_mean)

        log_lows_mean = sequence_mean(tf.log(lows + 1), self.series_length)
        log_lows = transform(lows, log_lows_mean)

        log_closes_mean = sequence_mean(tf.log(closes + 1), self.series_length)
        log_closes = transform(closes, log_closes_mean)

        log_volumes_mean = sequence_mean(tf.log(volumes + 1), self.series_length)
        log_volumes = transform(volumes, log_volumes_mean)

        

        log_order_pricess = tf.log(order_prices + 1) - log_x_mean

        log_current_prices = tf.log(current_prices + 1) - log_x_mean

        x = tf.expand_dims(log_x, 2)

        features = tf.concat([
            tf.expand_dims(log_opens, 2),
            tf.expand_dims(log_highs, 2),
            tf.expand_dims(log_lows, 2),
            tf.expand_dims(log_closes, 2),
            tf.expand_dims(log_volumes, 2),

            tf.tile(tf.expand_dims(tf.one_hot(positions + 1, 3), 1), (1, tf.shape(opens)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(time_since, 60), 1), (1, tf.shape(opens)[1], 1)),

            tf.expand_dims(tf.cast(todays, tf.float32), 2),

            tf.tile(tf.reshape(log_opens_mean, (-1, 1, 1)), (1, tf.shape(opens)[1], 1)),
            tf.tile(tf.reshape(log_highs_mean, (-1, 1, 1)), (1, tf.shape(opens)[1], 1)),
            tf.tile(tf.reshape(log_lows_mean, (-1, 1, 1)), (1, tf.shape(opens)[1], 1)),
            tf.tile(tf.reshape(log_closes_mean, (-1, 1, 1)), (1, tf.shape(opens)[1], 1)),
            tf.tile(tf.reshape(log_volumes_mean, (-1, 1, 1)), (1, tf.shape(opens)[1], 1)),
            tf.tile(tf.reshape(log_x_mean, (-1, 1, 1)), (1, tf.shape(opens)[1], 1)),
            tf.tile(tf.reshape(log_order_pricess, (-1, 1, 1)), (1, tf.shape(opens)[1], 1)),
            tf.tile(tf.reshape(log_current_prices, (-1, 1, 1)), (1, tf.shape(opens)[1], 1)),

        ], axis=2)

        return tf.concat([x, features], axis=2)



    def wavenet_logits_train(self):

        x = self.get_inputs(self.opens, self.highs, self.lows, self.closes, self.volumes,
                            self.positions, self.order_prices, self.current_prices, self.time_since, self.todays)



        inputs, w, b = temporal_convolution_layer(
            inputs=x,
            output_units= 8,
            convolution_width=1,
            scope='train-CNN-1x1'
        )

        self.w["wcnn1"] = w
        self.w["bcnn1"] = b


        outputs = lstm_layer(inputs, self.lengths, self.lstm_size,scope="series-lstm-train")



        h,w, b = time_distributed_dense_layer(outputs, 128,
                                         scope='train-dense-encode-1',
                                         activation=tf.nn.relu )
        self.w["wtf1"] = w
        self.w["btf1"] = b

        out,w, b = time_distributed_dense_layer(h, 32,
                                             scope='train-dense-encode-2',
                                            activation=tf.nn.relu,
                                             reuse=tf.AUTO_REUSE)
        self.w["wtf2"] = w
        self.w["btf2"] = b

        shape = out.get_shape().as_list()
        out_flat = tf.reshape(out, [tf.shape(out)[0], 1, shape[1] * shape[2]])

        out, state = stateful_lstm(out_flat, self.num_lstm_layers, self.lstm_size, tuple([self.lstm_state_train]),
                                               scope_name="lstm_train")
        self.state_output_c = state[0][0]
        self.state_output_h = state[0][1]

        shape = out.get_shape().as_list()
        out = tf.reshape(out, [tf.shape(out)[0], shape[2]])

        out, w, b = fully_connected_layer(out, self.n_actions, scope_name='train-dense-encode-2', activation=None)

        self.w["wout"] = w
        self.w["bout"] = b


        self.q_out =  out
        self.q_action = tf.argmax(self.q_out, axis=1)

    def wavenet_logits_target(self):

        x=self.get_inputs(self.opens_,self.highs_,self.lows_,self.closes_,self.volumes_,
                               self.positions_,self.order_prices_,self.current_prices_,self.time_since_,self.todays_)

        inputs, w, b = temporal_convolution_layer(
            inputs=x,
            output_units=8,
            convolution_width=1,
            scope='target-CNN-1x1'
        )

        self.w_target["wcnn1"] = w
        self.w_target["bcnn1"] = b

        outputs = lstm_layer(inputs, self.lengths, self.lstm_size,
                             scope="series-lstm-target")

        h, w, b = time_distributed_dense_layer(outputs, 128,
                                               scope='target-dense-encode-1',
                                               activation=tf.nn.relu,
                                               reuse=tf.AUTO_REUSE)
        self.w_target["wtf1"] = w
        self.w_target["btf1"] = b

        out, w, b = time_distributed_dense_layer(h, 32,
                                               scope='target-dense-encode-2',
                                               activation=tf.nn.relu,
                                               reuse=tf.AUTO_REUSE)
        self.w_target["wtf2"] = w
        self.w_target["btf2"] = b


        shape = out.get_shape().as_list()
        out_flat = tf.reshape(out, [tf.shape(out)[0], 1, shape[1] * shape[2]])
        out, state = stateful_lstm(out_flat, self.num_lstm_layers, self.lstm_size,
                                                      tuple([self.lstm_state_target]), scope_name="lstm_target")
        self.state_output_target_c = state[0][0]
        self.state_output_target_h = state[0][1]

        shape = out.get_shape().as_list()

        out = tf.reshape(out, [tf.shape(out)[0], shape[2]])

        out, w, b = fully_connected_layer(out, self.n_actions, scope_name='target-dense-encode-2', activation=None)

        self.w_target["wout"] = w
        self.w_target["bout"] = b

        self.q_target_out = out
        self.q_target_action = tf.argmax(self.q_target_out, axis=1)

    def add_loss_op_target(self):
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1, name='q_acted')
        self.delta = train - self.target_val
        self.loss = tf.reduce_mean(huber_loss(self.delta))

        avg_q = tf.reduce_mean(self.q_out, 0)
        q_summary = []
        for i in range(self.n_actions):
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        #self.merged_image_sum = tf.summary.merge(self.image_summary, "images")
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)



    def build(self):
        self.add_placeholders()
        self.wavenet_logits_train()
        self.wavenet_logits_target()
        self.add_loss_op_target()

        self.add_train_op(self.lr_method, self.lr, self.loss, clip=10)
        self.initialize_session()
        self.init_update()

    def update_target(self):
        for name in self.w:
            self.target_w_assign[name].eval({self.target_w_in[name]: self.w[name].eval(session=self.sess)},
                                            session=self.sess)
        for var in self.lstm_vars:
            self.target_w_assign[var.name].eval({self.target_w_in[var.name]: var.eval(session=self.sess)},
                                                session=self.sess)
        for var in self.series_lstm_vars:
            self.target_w_assign[var.name].eval({self.target_w_in[var.name]: var.eval(session=self.sess)},
                                                session=self.sess)

    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}

        for name in self.w:
            self.target_w_in[name] = tf.placeholder(tf.float32, self.w_target[name].get_shape().as_list(), name=name)
            self.target_w_assign[name] = self.w_target[name].assign(self.target_w_in[name])

        self.lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm_train")
        lstm_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm_target")

        for i, var in enumerate(self.lstm_vars):
            self.target_w_in[var.name] = tf.placeholder(tf.float32, var.get_shape().as_list())
            self.target_w_assign[var.name] = lstm_target_vars[i].assign(self.target_w_in[var.name])


        self.series_lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="series-lstm-train")
        series_lstm_vars_taerget= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="series-lstm-target")
        for i, var in enumerate(self.series_lstm_vars):
            self.target_w_in[var.name] = tf.placeholder(tf.float32, var.get_shape().as_list())
            self.target_w_assign[var.name] = series_lstm_vars_taerget[i].assign(self.target_w_in[var.name])

