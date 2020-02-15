import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug
from networks.base import  BaseModel

from datetime import datetime, timedelta

from tf_utils import (time_distributed_dense_layer, temporal_convolution_layer,
    sequence_mean, sequence_smape,transform,inverse_transform, shape)

from pykalman import KalmanFilter


class WaveNet(BaseModel):

    def __init__(self, config):
        super(WaveNet, self).__init__(config, "wavenet")

        #self.history_len = config.history_len
        self.cnn_format = config.cnn_format
        self.residual_channels =config.residual_channels
        self.skip_channels = config.skip_channels
        self.dilations = config.dilations
        self.filter_widths = config.filter_widths
        self.forecast_window=config.forecast_window
        self.price_data_size=config.price_data_size
        self.batch_size=config.batch_size
        self.encode_series_len=self.price_data_size-self.forecast_window
        self.decode_series_len = config.forecast_window

    def predict(self, data,  today ):

        feed_dict=self.get_feed_dict(data,  today )

        forecasts = self.sess.run(
            fetches=self.preds,
            feed_dict=feed_dict
        )
        return forecasts.flatten()

    def train(self, data, today ):

        feed_dict = self.get_train_feed_dict(data,today)


        forecasts, targets, loss = self.sess.run(
            fetches=[self.preds, self.labels, self.loss],
            feed_dict=feed_dict
        )
        return forecasts.flatten(),targets.flatten(), loss

    def get_train_feed_dict(self, data,  today):

        l = self.encode_series_len
        dl = self.decode_series_len
        d = np.array([data.get_t_data(-i)[2:6] for i in range(l)])

        mean = d.mean()

        kf = KalmanFilter(em_vars=['transition_covariance', 'observation_covariance'], initial_state_mean=mean,
                          n_dim_obs=4)
        v = kf.em(d)

        h = v.smooth(d)
        h = h[0].flatten()

        self.stoday = datetime.strftime(today, '%m/%d/%Y')

        is_today = [data.get_t_data(-(i+dl))[0]  == self.stoday  for i  in range(l-dl)]
        vol=[data.get_t_data(-(i+dl))[6] for i in range(l-dl)]

        return {
            self.y_encode:[h[dl:l].tolist()],
            self.volume_encode: [vol],
            self.is_today: [is_today],
            self.decode_len: [dl],
            self.encode_len: [l-dl],
            self.y_decode:[h[0:dl].tolist()],
            self.lr: self.learning_rate
        }
    def get_feed_dict(self, data,  today ):

        l = self.encode_series_len
        dl = self.decode_series_len
        d = np.array([data.get_t_data(-i)[2:6] for i in range(l)])

        mean = d.mean()

        kf = KalmanFilter(em_vars=['transition_covariance', 'observation_covariance'], initial_state_mean=mean,
                          n_dim_obs=4)
        v = kf.em(d)

        h = v.smooth(d)
        h=h[0].flatten()

        self.stoday = datetime.strftime(today, '%m/%d/%Y')
        is_today = [data.get_t_data(-i)[0] == self.stoday for i in range(l-dl)]
        vol = [data.get_t_data(-i)[6] for i in range(l-dl)]

        return {
            self.y_encode: [h[:-dl].tolist()],
            self.volume_encode: [vol],
            self.is_today: [is_today],
            self.decode_len: [dl],
            self.y_decode: [h[0:dl].tolist()],
            self.encode_len: [l-dl],
            self.lr: self.learning_rate
        }

    def add_placeholders(self):



        self.y_encode = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.encode_len = tf.placeholder(dtype=tf.int32,shape= [None])
        self.y_decode = tf.placeholder(dtype=tf.float32, shape=[None, self.decode_series_len])
        self.decode_len = tf.placeholder(dtype=tf.int32,shape= [None])

        self.volume_encode = tf.placeholder(dtype=tf.float32, shape=[None, None])

        self.is_today = tf.placeholder(dtype=tf.int32, shape=[None,None])

        self.encode_len = tf.placeholder(dtype=tf.int32, shape=[None])

        self.y_decode = tf.placeholder(dtype=tf.float32,shape=[None, self.decode_series_len])
        self.decode_len = tf.placeholder(dtype=tf.int32, shape=[None])

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

        self.learning_rate_step = tf.placeholder(dtype=tf.int64, shape=None, name="learning_rate_step")

        # we can not have a position open overnight, so it can not be more than 390 minutes in theory.
        # But in reality, we won't be holding a position too long, so maybe let's say one hour max, 60 max

        # self.symbol = tf.placeholder(tf.int32, [None])
        # self.day = tf.placeholder(tf.int32, [None])


    def get_inputs(self):

        self.log_x_encode_mean = sequence_mean(tf.log(self.y_encode + 1), self.encode_series_len-self.decode_series_len)
        self.log_x_encode = transform(self.y_encode, self.log_x_encode_mean)
        

        self.log_volume_encode_mean = sequence_mean(tf.log(self.volume_encode + 1), self.encode_series_len-self.decode_series_len)
        self.log_volume_encode = transform(self.volume_encode, self.log_volume_encode_mean)

        self.x = tf.expand_dims(self.log_x_encode, 2)


        self.encode_features = tf.concat([
            tf.expand_dims(self.log_volume_encode, 2),

            tf.expand_dims(tf.cast(self.is_today, tf.float32), 2),

            tf.tile(tf.reshape(self.log_volume_encode_mean, (-1, 1, 1)), (1, tf.shape(self.y_encode)[1], 1)),
            tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, tf.shape(self.y_encode)[1], 1)),

        ], axis=2)

        decode_idx = tf.tile(tf.expand_dims(tf.range(self.decode_series_len), 0), (tf.shape(self.y_decode)[0], 1))

        self.decode_features = tf.concat([
            tf.one_hot(decode_idx, self.decode_series_len),
            tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, self.decode_series_len, 1))
        ], axis=2)


    def add_loss_op (self):


        self.y_hat_decode = inverse_transform(tf.squeeze(self.y_hat_decode, 2), self.log_x_encode_mean)
        self.y_hat_decode = tf.nn.relu(self.y_hat_decode)

        self.labels = self.y_decode
        self.preds = self.y_hat_decode
        self.loss = sequence_smape(self.labels, self.preds, self.decode_series_len)


    def build(self):
        self.add_placeholders()
        self.get_inputs()

        self.y_hat_encode, conv_inputs = self.encode(self.x, features=self.encode_features)
        self.initialize_decode_params(self.x, features=self.decode_features)
        self.y_hat_decode = self.decode(self.y_hat_encode, conv_inputs, features=self.decode_features)

        self.add_loss_op()

        self.add_train_op(self.lr_method, self.lr, self.loss, clip=10)

        self.initialize_session()



    def encode(self, x, features):

        x = tf.concat([x, features], axis=2)

        inputs = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='x-proj-encode'
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
                scope='dilated-conv-encode-{}'.format(i)
            )
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='dilated-conv-proj-encode-{}'.format(i)
            )
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))

        h = time_distributed_dense_layer(skip_outputs, 128,
                                         scope='dense-encode-1',
                                         activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 1,
                                             scope='dense-encode-2' )

        return y_hat, conv_inputs[:-1]

    def initialize_decode_params(self, x, features):
        x = tf.concat([x, features], axis=2)

        inputs = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='x-proj-decode'
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
                scope='dilated-conv-decode-{}'.format(i)
            )
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='dilated-conv-proj-decode-{}'.format(i)
            )
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = time_distributed_dense_layer(skip_outputs, 128, scope='dense-decode-1', activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 1, scope='dense-decode-2')
        return y_hat

    def decode(self, x, conv_inputs1, features):
        batch_size = tf.shape(x)[0]

        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(conv_inputs1, self.dilations)):
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            batch_idx = tf.reshape(batch_idx, [-1])

            queue_begin_time = self.encode_len - dilation - 1
            temporal_idx = tf.expand_dims(queue_begin_time, 1) + tf.expand_dims(tf.range(dilation), 0)
            temporal_idx = tf.reshape(temporal_idx, [-1])

            idx = tf.stack([batch_idx, temporal_idx], axis=1)
            slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, shape(conv_input, 2)))

            layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation + self.decode_series_len)
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        features_ta = tf.TensorArray(dtype=tf.float32, size=self.decode_series_len)
        features_ta = features_ta.unstack(tf.transpose(features, (1, 0, 2)))

        # initialize output tensor array
        emit_ta = tf.TensorArray(size=self.decode_series_len, dtype=tf.float32)

        # initialize other loop vars
        elements_finished = 0 >= self.decode_len
        time = tf.constant(0, dtype=tf.int32)

        # get initial x input
        current_idx = tf.stack([tf.range(tf.shape(self.encode_len)[0]), self.encode_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx)

        def loop_fn(time1, current_input, queues):

            current_features = features_ta.read(time1)
            current_input = tf.concat([current_input, current_features], axis=1)

            with tf.variable_scope('x-proj-decode', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                x_proj = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            skip_outputs, updated_queues = [], []
            for i, (conv_input, queue, dilation) in enumerate(zip(conv_inputs1, queues, self.dilations)):
                state = queue.read(time1)
                with tf.variable_scope('dilated-conv-decode-{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights'.format(i))
                    b_conv = tf.get_variable('biases'.format(i))
                    dilated_conv = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(x_proj, w_conv[1, :, :]) + b_conv
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

                with tf.variable_scope('dilated-conv-proj-decode-{}'.format(i), reuse=True):
                    w_proj = tf.get_variable('weights'.format(i))
                    b_proj = tf.get_variable('biases'.format(i))
                    concat_outputs = tf.matmul(dilated_conv, w_proj) + b_proj
                skips, residuals = tf.split(concat_outputs, [self.skip_channels, self.residual_channels], axis=1)

                x_proj += residuals
                skip_outputs.append(skips)
                updated_queues.append(queue.write(time1 + dilation, x_proj))

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            with tf.variable_scope('dense-decode-1', reuse=True):
                w_h = tf.get_variable('weights')
                b_h = tf.get_variable('biases')
                h = tf.nn.relu(tf.matmul(skip_outputs, w_h) + b_h)

            with tf.variable_scope('dense-decode-2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                y_hat2 = tf.matmul(h, w_y) + b_y

            elements_finished2 = (time1 >= self.decode_len)
            finished = tf.reduce_all(elements_finished2)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 1], dtype=tf.float32),
                lambda: y_hat2
            )
            next_elements_finished = (time1 >= self.decode_len -1)

            return next_elements_finished, next_input, updated_queues

        def condition(unused_time, elements_finished1, *_):
            return tf.logical_not(tf.reduce_all(elements_finished1))

        def body(time1, elements_finished1, emit_ta1, *state_queues1):
            (next_finished, emit_output, state_queues2) = loop_fn(time1, initial_input, state_queues1)

            emit = tf.where(elements_finished1, tf.zeros_like(emit_output), emit_output)
            emit_ta2 = emit_ta1.write(time1, emit)

            #elements_finished2 = tf.logical_or(elements_finished1, next_finished)

            return [time1 + 1, next_finished, emit_ta2] + list(state_queues2)

        returned = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[time, elements_finished, emit_ta] + state_queues
        )

        outputs_ta = returned[2]
        y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))

        return y_hat

