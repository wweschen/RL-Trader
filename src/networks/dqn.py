import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug
from networks.base import  BaseModel

from datetime import datetime, timedelta

from tf_utils import (time_distributed_dense_layer, temporal_convolution_layer,
    sequence_mean, fully_connected_layer, transform, sequence_smape, shape,huber_loss)



class DQN(BaseModel):

    def __init__(self, n_actions, config):
        super(DQN, self).__init__(config, "dqn")
        self.n_actions = n_actions
        #self.history_len = config.history_len
        self.cnn_format = config.cnn_format
        self.residual_channels =config.residual_channels
        self.skip_channels = config.skip_channels
        self.dilations = config.dilations
        self.filter_widths = config.filter_widths
        self.forecast_window=config.forecast_window
        self.price_data_size=config.price_data_size
        self.batch_size=config.batch_size
        self.series_length=self.price_data_size-self.forecast_window

    def get_feed_dict(self, data, action, reward, terminal,
                      positions, today, order_prices, current_prices, time_steps_since):

        self.stoday = datetime.strftime(today, '%m/%d/%Y')
        is_today = [date == self.stoday for date in data.dates]

        return {
            self.open_encode: [data.opens],
            self.high_encode: [data.highs],
            self.low_encode: [data.lows],
            self.close_encode: [data.closes],
            self.volume_encode: [data.volumes], 
            self.is_today: [is_today],
            self.action: [action],
            self.reward: [reward],
            self.terminal: [terminal],
            self.position: [positions],
            self.order_price: [order_prices],
            self.est_current_price: [current_prices],
            self.time_since_open: [time_steps_since],
            self.lr: self.learning_rate
        }
    def get_feed_dict_batch(self,opens,highs,lows,closes,volumes, todays,action, reward, terminal,
                              positions,dates,order_prices,current_prices,time_steps_since):

        stoday = [datetime.strftime(d.astype(datetime), '%m/%d/%Y') for d in dates]
        #print(type(stoday))
        is_today = [[date ==stoday for date in todays[i]] for i in range(len(dates)-1)]

        return {
            self.open_encode: opens,
            self.high_encode:highs,
            self.low_encode: lows,
            self.close_encode: closes,
            self.volume_encode: volumes, 
            self.is_today: is_today,
            self.action: action,
            self.reward:reward,
            self.terminal:terminal,
            self.position: positions,
            self.order_price: order_prices,
            self.est_current_price: current_prices,
            self.time_since_open: time_steps_since,
            self.lr: self.learning_rate
        }

    def train_on_batch_target(self, opens,highs,lows,closes,volumes, todays,action, reward,opens_,highs_,lows_,closes_,volumes_,
                              todays_,terminal, steps,positions,dates,order_prices,current_prices,time_steps_since):

        feed_dict = self.get_feed_dict_batch(opens,highs,lows,closes,volumes, todays,action, reward, terminal,
                              positions,dates,order_prices,current_prices,time_steps_since)

        feed_dict_ = self.get_feed_dict_batch(opens_,highs_,lows_,closes_,volumes_, todays_,action, reward, terminal,
                              positions,dates,order_prices,current_prices,time_steps_since)


        target_val = self.q_target_out.eval(feed_dict=feed_dict_, session=self.sess)
        max_target = np.max(target_val, axis=1)
        target = (1. - terminal) * self.gamma * max_target + reward
        feed_dict[self.target_val] = target

        _, q, train_loss, q_summary  = self.sess.run(
            [self.train_op, self.q_out, self.loss, self.avg_q_summary],
            feed_dict=feed_dict
        )

        if self.train_steps % 1000 == 0:
            self.file_writer.add_summary(q_summary, self.train_steps)
            #self.file_writer.add_summary(image_summary, self.train_steps)
        if steps % 20000 == 0 and steps > 50000:
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum:
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1
        return q.mean(), train_loss


    def add_placeholders(self):
        self.w = {}
        self.w_target = {}

        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.terminal = tf.placeholder(dtype=tf.float32, shape=[None], name="terminal")

        self.target_val = tf.placeholder(dtype=tf.float32, shape=[None], name="target_val")


        self.learning_rate_step = tf.placeholder("int64", None, name="learning_rate_step")

        self.open_encode = tf.placeholder(tf.float32, [None, self.series_length])
        self.high_encode = tf.placeholder(tf.float32, [None,self.series_length])
        self.low_encode = tf.placeholder(tf.float32, [None, self.series_length])
        self.close_encode = tf.placeholder(tf.float32, [None, self.series_length])
        self.volume_encode = tf.placeholder(tf.float32, [None, self.series_length])

        self.is_today = tf.placeholder(tf.int32, [None, self.series_length]) 
        self.position = tf.placeholder(tf.int32, [None])  # -1 short position, 0 neutral, +1 long position
        self.order_price = tf.placeholder(tf.float32, [None])
        self.est_current_price = tf.placeholder(tf.float32, [None])
        self.time_since_open = tf.placeholder(tf.int32, [None])
        # we can not have a position open overnight, so it can not be more than 390 minutes in theory.
        # But in reality, we won't be holding a position too long, so maybe let's say one hour max, 60 max

        # self.symbol = tf.placeholder(tf.int32, [None])
        # self.day = tf.placeholder(tf.int32, [None])


    def get_inputs(self):

        self.log_x_encode_mean = sequence_mean(tf.log((self.high_encode + self.low_encode) / 2. + 1), self.series_length)
        self.log_x_encode = transform((self.high_encode + self.low_encode) / 2., self.log_x_encode_mean)
        

        self.log_open_encode_mean = sequence_mean(tf.log(self.open_encode + 1), self.series_length)
        self.log_open_encode = transform(self.open_encode, self.log_open_encode_mean)

        self.log_high_encode_mean = sequence_mean(tf.log(self.high_encode + 1), self.series_length)
        self.log_high_encode = transform(self.high_encode, self.log_high_encode_mean)

        self.log_low_encode_mean = sequence_mean(tf.log(self.low_encode + 1), self.series_length)
        self.log_low_encode = transform(self.low_encode, self.log_low_encode_mean)

        self.log_close_encode_mean = sequence_mean(tf.log(self.close_encode + 1), self.series_length)
        self.log_close_encode = transform(self.close_encode, self.log_close_encode_mean)

        self.log_volume_encode_mean = sequence_mean(tf.log(self.volume_encode + 1), self.series_length)
        self.log_volume_encode = transform(self.volume_encode, self.log_volume_encode_mean)

        self.position = tf.placeholder(tf.int32, [None])

        self.log_order_price = tf.log(self.order_price + 1) - self.log_x_encode_mean

        self.log_est_current_price = tf.log(self.est_current_price + 1) - self.log_x_encode_mean

        self.x = tf.expand_dims(self.log_x_encode, 2)

        self.features = tf.concat([
            tf.expand_dims(self.log_open_encode, 2),
            tf.expand_dims(self.log_high_encode, 2),
            tf.expand_dims(self.log_low_encode, 2),
            tf.expand_dims(self.log_close_encode, 2),
            tf.expand_dims(self.log_volume_encode, 2),

            tf.tile(tf.expand_dims(tf.one_hot(self.position + 1, 3), 1), (1, tf.shape(self.open_encode)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.time_since_open, 60), 1), (1, tf.shape(self.open_encode)[1], 1)),

            #tf.expand_dims(tf.cast(self.is_today, tf.float32), 2),

            tf.tile(tf.reshape(self.log_open_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
            tf.tile(tf.reshape(self.log_high_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
            tf.tile(tf.reshape(self.log_low_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
            tf.tile(tf.reshape(self.log_close_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
            tf.tile(tf.reshape(self.log_volume_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
            tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
            tf.tile(tf.reshape(self.log_order_price, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),
            tf.tile(tf.reshape(self.log_est_current_price, (-1, 1, 1)), (1, tf.shape(self.open_encode)[1], 1)),

        ], axis=2)

        self.x = tf.concat([self.x, self.features], axis=2)

    def wavenet_logits_train(self):

        x= self.x
        inputs,w,b = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='train-x-proj-encode',
            reuse=tf.AUTO_REUSE
        )
        self.w["wf0"] = w
        self.w["bf0"] = b

        skip_outputs = []
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv,w,b = temporal_convolution_layer(
                inputs=inputs,
                output_units=2 * self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='train-dilated-conv-encode-{}'.format(i),
                reuse=tf.AUTO_REUSE
            )
            self.w["wc{}".format(i)] = w
            self.w["wb{}".format(i)] = b

            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

            outputs,w,b = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='train-dilated-conv-proj-encode-{}'.format(i),
                reuse=tf.AUTO_REUSE
            )
            self.w["wtf-{}".format(i)] = w
            self.w["btf-{}".format(i)] = b
            
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))

        h,w, b = time_distributed_dense_layer(skip_outputs, 128,
                                         scope='train-dense-encode-1',
                                         activation=tf.nn.relu,
                                         reuse=tf.AUTO_REUSE)
        self.w["wtf1"] = w
        self.w["btf1"] = b

        h,w, b = time_distributed_dense_layer(h, 3,
                                             scope='train-dense-encode-2',
                                            activation=tf.nn.relu,
                                             reuse=tf.AUTO_REUSE)
        self.w["wtf2"] = w
        self.w["btf2"] = b

        s = h.get_shape().as_list()
        out_flat = tf.reshape(h, [-1, reduce(lambda x, y: x * y, s[1:])])

        h, w, b = fully_connected_layer(out_flat, 128, scope_name='train-dense-encode-1', activation=tf.nn.relu)
        self.w["wf1"] = w
        self.w["bf1"] = b

        out, w, b = fully_connected_layer(h, self.n_actions, scope_name='train-dense-encode-2', activation=None)

        self.w["wout"] = w
        self.w["bout"] = b

        # h,w,b = time_distributed_dense_layer(skip_outputs, 128,
        #                                  scope='dense-encode-1',
        #                                  activation=tf.nn.relu,
        #                                  reuse=tf.AUTO_REUSE)
        # self.w["wf2"] = w
        # self.w["bf2"] = b
        #
        # out,w,b = time_distributed_dense_layer(h, 3,
        #                                      scope='dense-encode-2',
        #                                      reuse=tf.AUTO_REUSE)

        # self.w["wout"] = w
        # self.w["bout"] = b

        self.q_out =  out
        self.q_action = tf.argmax(self.q_out, axis=1)

    def wavenet_logits_target(self):

        x= self.x

        inputs,w,b = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='target-x-proj-encode',
            reuse=False
        )

        self.w_target["wf0"] = w
        self.w_target["bf0"] = b

        skip_outputs = []
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv, w, b = temporal_convolution_layer(
                inputs=inputs,
                output_units=2 * self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='target-dilated-conv-encode-{}'.format(i),
                reuse=tf.AUTO_REUSE
            )
            self.w_target["wc{}".format(i)] = w
            self.w_target["wb{}".format(i)] = b

            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

            outputs, w, b = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='target-dilated-conv-proj-encode-{}'.format(i),
                reuse=tf.AUTO_REUSE
            )
            self.w_target["wtf-{}".format(i)] = w
            self.w_target["btf-{}".format(i)] = b

            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))

        h, w, b = time_distributed_dense_layer(skip_outputs, 128,
                                               scope='target-dense-encode-1',
                                               activation=tf.nn.relu,
                                               reuse=tf.AUTO_REUSE)
        self.w_target["wtf1"] = w
        self.w_target["btf1"] = b

        h, w, b = time_distributed_dense_layer(h, 3,
                                               scope='target-dense-encode-2',
                                               activation=tf.nn.relu,
                                               reuse=tf.AUTO_REUSE)
        self.w_target["wtf2"] = w
        self.w_target["btf2"] = b

        s = h.get_shape().as_list()
        out_flat = tf.reshape(h, [-1, reduce(lambda x, y: x * y, s[1:])])

        h, w, b = fully_connected_layer(out_flat, 128, scope_name='target-dense-encode-1', activation=tf.nn.relu)
        self.w_target["wf1"] = w
        self.w_target["bf1"] = b

        out, w, b = fully_connected_layer(h, self.n_actions, scope_name='target-dense-encode-2', activation=None)

        self.w_target["wout"] = w
        self.w_target["bout"] = b

        self.q_target_out = out
        self.q_target_action = tf.argmax(self.q_target_out, axis=1)
        


    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}
        for name in self.w:
            self.target_w_in[name] = tf.placeholder(tf.float32, self.w_target[name].get_shape().as_list(), name=name)
            self.target_w_assign[name] = self.w_target[name].assign(self.target_w_in[name])

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
        self.get_inputs()
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

