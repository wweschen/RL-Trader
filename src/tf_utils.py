import tensorflow as tf
import time
import numpy as np
import os.path
import csv
import errno
import sys
if (sys.version_info[0]==2):
  import cPickle
elif (sys.version_info[0]==3):
  import _pickle as cPickle


def lstm_layer(inputs, lengths, state_size, keep_prob=1.0, scope='lstm-layer', reuse=False, return_final_state=False):
    """
    LSTM layer.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, max sequence length, state_size] containing the lstm
        outputs at each timestep.

    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, output_state = tf.nn.dynamic_rnn(
            inputs=inputs,
            cell=cell_fw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        if return_final_state:
            return outputs, output_state
        else:
            return outputs

def stateful_lstm(x, num_layers, lstm_size, state_input, scope_name="lstm"):
    with tf.variable_scope(scope_name):
        cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)

        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=x,
            initial_state=state_input)
        return outputs, state

def temporal_convolution_layer(inputs, output_units, convolution_width, causal=False, dilation_rate=[1] ,
                               activation=None, dropout=None, scope='temporal-convolution-layer', reuse=False,parameter_out=False):
    """
    Convolution over the temporal axis of sequence data.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of timesteps to use in convolution.
        causal: Output at timestep t is a function of inputs at or before timestep t.
        dilation_rate:  Dilation rate along temporal axis.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        shift=0

        if causal:
            shift = (convolution_width / 2) + (int(dilation_rate[0] - 1) / 2)
            pad = tf.zeros([tf.shape(inputs)[0], int(shift), inputs.shape.as_list()[2]])
            inputs = tf.concat ([pad, inputs], axis=1)

        w = tf.get_variable(
            name='weights',
            initializer=tf.random_normal_initializer(
                mean=0,
                stddev=1.0 / tf.sqrt(float(convolution_width)*float(shape(inputs, 2)))
            ),
            shape=[convolution_width, shape(inputs, 2), output_units]
        )

        z = tf.nn.convolution(inputs, w, padding='SAME', dilation_rate=dilation_rate)

        b = tf.get_variable(
            name='biases',
            initializer=tf.constant_initializer(),
            shape=[output_units]
        )
        z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        z = z[:, :int(-shift), :] if causal else z

        if parameter_out:
            return z, w, b
        else:
            return z

def fully_connected_layer(x, output_dim, scope_name="fully", initializer=tf.random_normal_initializer(stddev=0.02),
                          activation=tf.nn.relu):
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope_name):
        w = tf.get_variable("w", [shape[1], output_dim], dtype=tf.float32,
                            initializer=initializer)
        b = tf.get_variable("b", [output_dim], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        out = tf.nn.xw_plus_b(x, w, b)
        if activation is not None:
            out = activation(out)

        return out, w, b


def time_distributed_dense_layer(inputs, output_units,   activation=None, batch_norm=None,
                                 dropout=None, scope='time-distributed-dense-layer', reuse=False,parameter_out=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable(
            name='weights',
            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0 / float(shape(inputs, -1))),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, w)

        b = tf.get_variable(
            name='biases',
            initializer=tf.constant_initializer(),
            shape=[output_units]
        )
        z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        if parameter_out:
            return z , w, b
        else:
            return z

def huber_loss(x, delta=1.0):
    return tf.where(tf.abs(x) < delta, 0.5 * tf.square(x), delta * tf.abs(x) - 0.5* delta)



def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]


def sequence_smape(y, y_hat, sequence_lengths ):
    max_sequence_length = tf.shape(y)[1]
    y = tf.cast(y, tf.float32)
    smape = 2*(tf.abs(y_hat - y) / (tf.abs(y) + tf.abs(y_hat)))

    # ignore discontinuity
    zero_loss = 2.0*tf.ones_like(smape)
    nonzero_loss = smape
    smape = tf.where(tf.logical_or(tf.equal(y, 0.0), tf.equal(y_hat, 0.0)), zero_loss, nonzero_loss)

    sequence_mask = tf.cast(tf.sequence_mask(sequence_lengths, maxlen=max_sequence_length), tf.float32)
    #sequence_mask = sequence_mask*(1 - is_nan)
    avg_smape = tf.reduce_sum(smape*sequence_mask) / tf.reduce_sum(sequence_mask)
    return avg_smape


def sequence_mean(x, lengths):
    return tf.reduce_sum(x, axis=1) / tf.cast(lengths, tf.float32)



def transform( x,mean):
    return tf.log(x + 1) - tf.expand_dims(mean, 1)

def inverse_transform( x,mean):
        return tf.exp(x + tf.expand_dims(mean, 1)) - 1


def save_pkl(obj, path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path, 'wb') as f:
        cPickle.dump(obj, f)
        print("  [*] save %s" % path)

def load_pkl(path):
    if os.path.exists(path):
      with open(path,'rb') as f:
        obj = cPickle.load(f)
        print("  [*] load %s" % path)
        return obj

