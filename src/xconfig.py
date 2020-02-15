class AgentConfig(object):
  scale = 10000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.99
  target_q_update_step = 1 * scale
  learning_rate = 0.00025
  learning_rate_minimum = 0.000025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  optimizer = 'adam'
  grad_clip = 20
  regularization_constant = 0.0
  keep_prob = 1.0
  early_stopping_steps = 3000
  warm_start_init_step = 0
  num_restarts = None
  enable_parameter_averaging = False
  min_steps_to_checkpoint = 100
  log_interval = 20
  loss_averaging_window = 100
  num_validation_batches = 1
  log_dir = 'logs'


  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  history_length = 4
  train_frequency = 4
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  env_name="OIH"
  forecast_window = 15
  _test_step = 5 * scale
  _save_step = _test_step * 10

  residual_channels = 32
  skip_channels = 32
  dilations = [2 ** i for i in range(4)]
  filter_widths = [2 for i in range(4)]


def get_config(FLAGS):

  config=AgentConfig()

  for k  in FLAGS:
    v=FLAGS[k].value
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
