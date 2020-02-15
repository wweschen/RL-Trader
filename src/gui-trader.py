

import sys
import os
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import functools
import numpy as np
import random as rd
from PyQt5.QtCore import (QFile, QFileInfo, QPoint, QRect, QSettings, QSize,
        Qt, QTextStream)
from PyQt5.QtCore import pyqtSlot
from market_chart_widget import  ChartWidget
from emulator import Market,PriceData
from numpy import *
import matplotlib
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading
from time import sleep

import random


import tensorflow as tf

from  agent_q import QAgent
from  agent_drqn import DRQNAgent

from  agent_kalman_SAX_wavelets_search import KSWsearchAgent
from  agent_kalman_sax_wavelets import KSWAgent

from emulator import Market

from config import Config
#
# flags = tf.app.flags
#
# flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not')
# flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
# flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
# flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
# flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
#
# FLAGS = flags.FLAGS
#
# # Set random seed
# tf.set_random_seed(FLAGS.random_seed)
# random.seed(FLAGS.random_seed)
#
# if FLAGS.gpu_fraction == '':
#   raise ValueError("--gpu_fraction should be defined")



def setCustomSize(x, width, height):
    sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(x.sizePolicy().hasHeightForWidth())
    x.setSizePolicy(sizePolicy)
    x.setMinimumSize(QtCore.QSize(width, height))
    x.setMaximumSize(QtCore.QSize(width, height))

''''''

class CustomMainWindow(QMainWindow):

    def __init__(self,config, app):

        super(CustomMainWindow, self).__init__()

        # Define the geometry of the main window
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle("GUI RL-Trader")

        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)


        # Place the matplotlib figure
        self.myFig = ChartWidget('OIH',18,12,config.forecast_window,parent=self,app=app)

        self.LAYOUT_A.addWidget(self.myFig )

        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'MarketDataFeed',
                                      target = play,
                                      daemon = True,
                                      args = (self.addData_callbackFunc,))
        myDataLoop.start()

        self.show()

    ''''''


    def addData_callbackFunc(self, data,position,reward,forecasts,forecast_history ):

        self.myFig.add_data(data,position,reward,forecasts,forecast_history )



''' End Class '''

# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QtCore.QObject):
    data_signal = QtCore.pyqtSignal(PriceData,float,float,ndarray,ndarray )

''' End Class '''


def play(addData_callbackFunc):

    config = Config()

    env = Market(config, 'OIH', '/Users/wweschen/rl-trader/data/oil.txt', 4, 1000)
    #env = Market(config, 'OIH', '/home/wes/rl-trader/data/oil.txt', 4, 1000)
    # if not FLAGS.use_gpu:
    #   config.cnn_format = 'NHWC'

    #agent = KSWAgent(config, env)
    agent =KSWsearchAgent(config, env)
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)


    if config.is_train:
        agent.train(config.train_steps, mySrc)
    else:
        agent.play(config.play_episodes,mySrc)




if __name__== '__main__':
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    config = Config()
    myGUI = CustomMainWindow(config, app)


    sys.exit(app.exec_())

''''''