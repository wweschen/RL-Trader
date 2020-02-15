import pandas as pd
import os
from numpy import genfromtxt
from datetime import datetime, timedelta
import queue
import sys
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from random import randint
import matplotlib.animation as animation

# 畫圖形週邊東西的套件
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter

import bisect

from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors

from dateutil.parser import parse

class Chart():

    def  __init__(self,symbol, data = None,forecast_data=None,forecast_hisory=None,
                  position_data=None,rewards_data=None, colorup='g', colordown='r'):

        self.colorup=colorup
        self.colordown=colordown

        self.forecast_window=30

        self.chartData = data
        self.forecasts = forecast_data
        self.forecast_history = forecast_hisory
        self.position_data = position_data
        self.rewards_data = rewards_data
        #for i in range(0, 200):
        #    self.chartData.add(next(self.dataGen))



        self.fig = plt.figure(figsize=(12, 12))
        self.fig.set_tight_layout({"pad": 1.5})

        self.fig.suptitle(symbol)

        gs = gridspec.GridSpec(7, 7)
        gs.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.


        self.tickindex = [i for i in range(0,170)]
        self.tick_length=len(self.tickindex)

        self.axPrice = self.fig.add_subplot(gs[0:4, :-1])
        self.axPrice.tick_params(axis="y", direction="in", left=True, right=True)
        self.axPrice.get_xaxis().set_visible(False)
        # Specify tick label size
        self.axPrice.tick_params(axis='x', which='major', labelsize=4)
        self.axPrice.tick_params(axis='x', which='minor', labelsize=0)
        self.axPrice.set_xticks(np.arange(0, self.tick_length, 10))
        self.axPrice.set_xticks(np.arange(0, self.tick_length , 1), minor=True)
        self.axPrice.set_xticklabels(np.array([i - self.tick_length for i in np.arange(0, self.tick_length , 10)]))

        major_ticks = np.arange(-10,110, 10)
        minor_ticks = np.arange(-10,110, 1)

        self.axPrice.set_yticks(major_ticks)
        self.axPrice.set_yticks(minor_ticks, minor=True)

        self.axPrice.set_ylabel('Price(% of range)', fontsize=8)
        self.axPrice.set_xlim(-170,0)
        self.axPrice.grid(True)

        self.axLearning = self.fig.add_subplot(gs[0:4, 5:6])
        self.axLearning.set_facecolor('None')
        #self.axLearning.get_yaxis().set_visible(False)
        self.axLearning.set_xticks(np.arange(0, 30, 5))
        self.axLearning.set_xticks(np.arange(0, 30, 1), minor=True)
        self.axLearning.tick_params(axis="x", direction="in", top=True, bottom=False)
        self.axLearning.xaxis.set_ticks_position('top')
        self.axLearning.set_yticks(major_ticks)
        self.axLearning.tick_params(labelleft='off')
        self.axLearning.yaxis.grid(True)
        self.axLearning.set_xticklabels(np.array([i - 30 for i in np.arange(0, 30, 5)]))

        self.axForecast = self.fig.add_subplot(gs[0:4, 6:])

        #self.axForecast.get_yaxis().set_visible(False)
        self.axForecast.set_xticks(np.arange(0, 30, 5))
        self.axForecast.set_xticks(np.arange(0, 30, 1), minor=True)
        self.axForecast.tick_params(axis="x", direction="in", top=True, bottom=False)
        self.axForecast.set_yticks(major_ticks)
        self.axForecast.tick_params(labelleft='off')
        self.axForecast.xaxis.set_ticks_position('top')

        self.axForecast.yaxis.grid(True)

        self.axVolume = self.fig.add_subplot(gs[4, :-1], sharex=self.axPrice)

        self.axReward = self.fig.add_subplot(gs[5, :-1], sharex=self.axPrice)
        self.axReward.get_yaxis().set_visible(True)
        self.axReward.set_ylabel('Rewards', fontsize=8)
        self.axReward.yaxis.tick_right()
        self.axReward.yaxis.grid(True)

        self.ax5 = self.fig.add_subplot(gs[5, 6:], sharey=self.axPrice)
        self.ax5.get_yaxis().set_visible(False)
        self.ax5.get_xaxis().set_visible(False)
        self.ax5.set_facecolor('None')

        self.axPosition = self.fig.add_subplot(gs[6:, :-1], sharex=self.axPrice)
        self.axPosition.get_yaxis().set_visible(True)
        self.axPosition.set_ylabel('Position', fontsize=8)
        self.axPosition.yaxis.grid(True)

        self.ax7 = self.fig.add_subplot(gs[6:, 6:])
        self.ax7.set_facecolor('None')
        self.ax7.get_yaxis().set_visible(False)
        self.ax7.get_xaxis().set_visible(False)




        self.axVolume.get_xaxis().set_visible(False)

        self.axVolume.yaxis.tick_right()

        self.axVolume.set_yticks(np.arange(0,10,2))
        self.axVolume.set_yticks(np.arange(0,10,1),minor=True)
        self.axVolume.set_ylabel('Volume', fontsize=8)

        self.axVolume.yaxis.set_label_position('right')
        self.axVolume.grid(True)

        self.axVolume.set_ylim(0, 10)
        self.axVolume.get_xaxis().set_visible(False)



    def candlestick(self, ax, opens, highs, lows, closes, width=4, colorup='g', colordown='r', alpha=0.75, ):

        delta = width / 2.

        barVerts = [((i - delta, open),
                     (i - delta, close),
                     (i + delta, close),
                     (i + delta, open))
                    for i, open, close in zip(range(len(opens)), opens, closes)]


        downSegments = [((i, low), (i, min(open, close)))
                        for i, low, high, open, close in zip(range(len(lows)), lows, highs, opens, closes)]

        upSegments = [((i, max(open, close)), (i, high))
                      for i, low, high, open, close in zip(range(len(lows)), lows, highs, opens, closes)]

        rangeSegments = upSegments + downSegments

        r, g, b = colorConverter.to_rgb(colorup)
        colorup = r, g, b, alpha
        r, g, b = colorConverter.to_rgb(colordown)
        colordown = r, g, b, alpha
        colord = {True: colorup,
                  False: colordown,
                  }
        colors = [colord[open < close] for open, close in zip(opens, closes)]

        useAA = 0,  # use tuple here
        lw = 0.5,  # and here
        rangeCollection = LineCollection(rangeSegments,
                                         colors=((0, 0, 0, 1),),
                                         linewidths=lw,
                                         antialiaseds=useAA,
                                         )

        barCollection = PolyCollection(barVerts,
                                       facecolors=colors,
                                       edgecolors=((0, 0, 0, 1),),
                                       antialiaseds=useAA,
                                       linewidths=lw,
                                       )

        minx, maxx = 0, len(rangeSegments) / 2
        miny = min([low for low in lows])
        maxy = max([high for high in highs])

        corners = (minx, miny), (maxx, maxy)

        ax.update_datalim(corners)


        # add these last
        rangeCollection.set_alpha(0.4)
        barCollection.set_alpha(0.4)
        ax.collections.clear()

        ax.add_collection(rangeCollection)
        ax.add_collection(barCollection)
        return [rangeCollection, barCollection]

    def volume_overlay(self, ax, opens, closes, volumes, colorup='g', colordown='r', width=4, alpha=1.0):


        colorup = mcolors.to_rgba(colorup, alpha)
        colordown = mcolors.to_rgba(colordown, alpha)
        colord = {True: colorup, False: colordown}
        colors = [colord[open < close]
                  for open, close in zip(opens, closes)
                  if open != -1 and close != -1]

        delta = width / 2.
        bars = [((i - delta, 0), (i - delta, v), (i + delta, v), (i + delta, 0))
                for i, v in enumerate(volumes)
                if v != -1]

        barCollection = PolyCollection(bars,
                                       facecolors=colors,
                                       edgecolors=((0, 0, 0, 1),),
                                       antialiaseds=(0,),
                                       linewidths=(0.5,),
                                       )

        barCollection.set_alpha(0.4)
        corners = (0, 0), (len(bars), max(volumes))

        ax.collections.clear()

        ax.add_collection(barCollection)
        ax.update_datalim(corners)

        return [barCollection]

    def millions(self,x):
        'The two args are the value and tick position'
        return '%1.1fM' % (x * 1e-6)

    def thousands(self,x):
        'The two args are the value and tick position'
        return '%1.1fK' % (x * 1e-3)

    def update(self):
        import time
        tstart = time.time()

        yh = self.chartData.highs.max()
        yl = self.chartData.lows.min()
        r = 100.0 / (yh - yl)

        vl = self.chartData.volumes.min()
        vh = self.chartData.volumes.max()
        vr=10.0 / (vh - vl)

        self.df = pd.DataFrame({
                           'Date':self.chartData.dates,
                           'Time': self.chartData.times,
                           'Open': (self.chartData.opens-yl)*r,
                           'High': (self.chartData.highs-yl)*r,
                           'Low': (self.chartData.lows-yl)*r,
                           'Close': (self.chartData.closes-yl)*r,
                           'Volume': (self.chartData.volumes-vl)*vr})

        ##  Learning zone ####



        self.axLearning.lines.clear()
        self.axLearning.collections.clear()


        self.target_data1 = (self.df.High[-self.forecast_window:] + self.df.Low[-self.forecast_window:]) / 2.0

        self.targets1 = self.axLearning.scatter(range(30), self.target_data1, s=1, c="b", alpha=0.9, marker=".")

        self.predhistory = self.axLearning.scatter(range(30),  self.forecast_history, s=1, c="k",   marker=".")



        self.axLearning.set_ylim(-10.0, 110.0)
        self.axLearning.set_xlim(0.0, 30.0)



        #### forecast Targets ############


        self.axForecast.lines.clear()
        self.axForecast.collections.clear()

        self.targets = self.axForecast.scatter(range(30), self.forecasts,s=1,c="k",   marker=".")
        #self.axForecast.set_xticks(np.arange(0, 30, 5))
        #self.axForecast.set_xticks(np.arange(0, 30, 1), minor=True)
        self.axForecast.set_ylim(-10.0, 110.0)
        self.axForecast.set_xlim(0.0, 30.0)

        ##### axPrice #################
        self.axPrice.texts.clear()
        self.axPrice.lines.clear()
        self.price_candles = self.candlestick(self.axPrice, self.df.Open, self.df.High, self.df.Low, self.df.Close,
                                              width=1,
                                              colorup=self.colorup, colordown=self.colordown)

        self.axPrice.set_ylim(-10.0, 110.0)
        self.axPrice.set_xlim( 0.0, 170.0)
        self.price_range = [yh - i * (yh - yl) / 10 for i in range(1, 11)]
        self.price_range = [self.price_range[0] + self.price_range[0] - self.price_range[1]] + self.price_range + \
                           [ self.price_range[9] - self.price_range[0] + self.price_range[1]]

        self.ranges = []
        for i in range(0, 12):
            self.ranges.append(self.axPrice.text(self.tick_length-40, (10 - i) * 10, '${:1.2f}'.format(self.price_range[i])))

        self.last_price = "Date: {}, Time:{}, Open:{:2.2f}, High:{:2.2f}, Low:{:2.2f}, Close:{:2.2f}, Volume:{}".format(
            self.chartData.dates[-1], self.chartData.times[-1], self.chartData.opens[-1],
            self.chartData.highs[-1], self.chartData.lows[-1], self.chartData.closes[-1], self.chartData.volumes[-1])
        #print(self.last_price)


        self.text = self.axPrice.text(0.80, 0.95, self.last_price,
                                      horizontalalignment='right',
                                      verticalalignment='bottom',
                                      transform=self.axPrice.transAxes)
        ##### date change line ######
        s = self.df.Date
        s2 = s.ne(s.shift().bfill()).astype(int)
        xs = s2.diff()[s2.diff() != 0].index.values

        if len(xs) > 1:
            x = xs[1]
        else:
            x = 0

        self.line = self.axPrice.axvline(x=x, linewidth=.5, color='k', alpha=0.5, linestyle='--')

        ####################

        ##### axVolume #########

        self.axVolume.texts.clear()

        self.volume_bars = self.volume_overlay(self.axVolume , self.df.Open, self.df.Close,
                                          self.df.Volume, colorup=self.colorup,
                                          colordown=self.colordown, width=1)




        self.vol_range=[(vh - i * (vh - vl) / 5) for i in range(1, 5)]

        self.vol_ranges = []

        for i in range(0, 4):
            self.vol_ranges.append(self.axVolume.text(self.tick_length-10, (4-i)* 2,  '{}'.format(self.thousands(self.vol_range[i]))))

        #### position #####
        self.axPosition.lines.clear()
        self.axPosition.collections.clear()

        #self.positions = self.axPosition.scatter(range(170), self.position_data.positions, s=15, c="k", marker="hline")
        self.positions= self.axPosition.plot(self.position_data.positions,markersize=1, marker='.', color='blue', drawstyle='steps-post')
        #### position #####
        self.axReward.lines.clear()
        self.axReward.collections.clear()

        self.axReward.set_ylim(self.rewards_data.rewards.min()-100,self.rewards_data.rewards.max()+100)

        #self.rewards = self.axReward.scatter(range(170), self.rewards_data.rewards, s=15, c="k", marker=".")
        self.rewards = self.axReward.plot(self.rewards_data.rewards, markersize=1, marker='.', color='g',
                                              drawstyle='steps-post')

        import gc
        gc.collect()
        print('FPS:', 1 / (time.time() - tstart),'Artists:',  len(self.fig.findobj()))

        #return self.price_candles+self.volume_bars  +self.ranges +[self.line]+ self.vol_ranges+[self.targets] +[self.targets1]+[self.predhistory] +[self.text] + [self.positions] + [self.rewards]
        self.fig.canvas.draw()

    def show(self):
        self.update(0)

        self.fig.canvas.draw()


