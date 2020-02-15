from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

class Config(object):

    train_steps = 50000000
    batch_size = 64
    history_len = 4
    frame_skip = 4
    epsilon_start = 0.2
    epsilon_end = 0.02
    max_steps = 10000
    epsilon_decay_episodes = 100000
    train_freq = 8
    update_freq = 10000
    train_start = 20000
    dir_save = "saved_session/"
    restore = False
    epsilon_decay = float((epsilon_start - epsilon_end))/float(epsilon_decay_episodes)
    random_start = 10
    test_step = 5000
    network_type = "drqn"


    gamma = 0.99
    learning_rate_minimum = 0.00025
    lr_method = "rmsprop"
    learning_rate = 0.00025
    lr_decay = 0.97
    keep_prob = 0.8

    num_lstm_layers = 1
    lstm_size = 512

    mem_size = 100000
    min_history = 1
    states_to_update = 4


    residual_channels = 32
    skip_channels = 32
    dilations = [2 ** i for i in range(4)]
    filter_widths = [2 for i in range(4)]

    price_data_size=200
    forecast_window= 16
    observation_window = 16
    env_name='OIH'

    is_train=True
    play_episodes=100

    if get_available_gpus():
        cnn_format = "NCHW"
    else:
        cnn_format = "NHWC"

