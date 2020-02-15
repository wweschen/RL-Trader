

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

    def __init__(self,app):

        super(CustomMainWindow, self).__init__()

        # Define the geometry of the main window
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle("my first window")

        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)


        # Place the matplotlib figure
        self.myFig = ChartWidget('OIH',18,12,parent=self,app=app)

        self.LAYOUT_A.addWidget(self.myFig )

        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'myDataLoop',
                                      target = play,
                                      daemon = True,
                                      args = (self.addData_callbackFunc,))
        myDataLoop.start()

        self.show()

    ''''''


    def addData_callbackFunc(self, data,position,reward,forecasts,forecast_history):

        self.myFig.add_data(data,position,reward,forecasts,forecast_history)



''' End Class '''

# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QtCore.QObject):
    data_signal = QtCore.pyqtSignal(PriceData,float,float,ndarray,ndarray)

''' End Class '''


def play(addData_callbackFunc):


    env = Market('OIH','/Users/wweschen/rl-trader/data/oil.txt', 4, 1000)

    data, today, time, position, reward, terminal, actions = env.random_past_day()

    terminal = False
    forecasts = np.zeros(30,dtype=float32)
    forecast_history=np.zeros(30,dtype=float32)
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    mySrc.data_signal.emit(data,  position, reward, forecasts, forecast_history)  # <- Here you emit a signal!

    while not terminal:

        actions = env.get_valid_actions()
        rnd_action = actions[rd.randint(0, len(actions) - 1)]
        data,today,time, position, reward, terminal, actions = env.step(rnd_action)
        print(data.times[-1])

        forecasts = np.random.uniform(0, 100, 30)

        forecast_history = (data.highs[-30:] + data.lows[-30:]) / 2.0

        forecast_history =  forecast_history + forecast_history * (
                0.5 - np.random.random(30)) * .2

        mySrc.data_signal.emit(data,position,reward,forecasts,forecast_history)  # <- Here you emit a signal!
        sleep(1)



if __name__== '__main__':
    app = QApplication(sys.argv)
    QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow(app)


    sys.exit(app.exec_())

''''''