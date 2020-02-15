
import numpy as np
import errno
import os
import os
import datetime
class BaseAgent():

    def __init__(self, config, environment):
        self.config = config
        self.env=environment

        self.lens = 0
        self.epsilon = config.epsilon_start
        self.min_reward = -1.
        self.max_reward = 1.0
        self.replay_memory = None
        self.history = None
        self.net = None
        if self.config.restore:
            self.load()
        else:
            self.i = 0



    def save(self):
        if self.replay_memory is not None:
            self.replay_memory.save()
        self.net.save_session(self.i)
        np.save(self.config.dir_save+'step.npy', self.i)

    def load(self):
        if self.replay_memory is not None:
            self.replay_memory.load()
        self.net.restore_session()
        self.i = np.load(self.config.dir_save+'step.npy')

    def log_trade(self, action, date, time, unit, order_price, current_price, gain, cause ):
        sdate = date.strftime('%Y-%m-%d')
        stime=""
        if time is not None:
            stime = time.strftime('%H:%M:%S')

        path = os.path.join(self.config.env_name, 'trade_logs', sdate, "trade_log.csv")
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
                '{},{},{},{},{:0.2f},{:0.2f},{:0.2f},{}\n'.format(sdate, stime, action,
                                                               unit, order_price, current_price, gain, cause))


