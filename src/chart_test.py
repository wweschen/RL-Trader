import pandas as pd
import os
from numpy import genfromtxt
from datetime import datetime, timedelta
import queue
import sys
import numpy as np

# MatPlotLib 的主要模組
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import matplotlib.animation as animation

# 畫圖形週邊東西的套件
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter

import bisect

# 畫圖用的套件
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors

from dateutil.parser import parse


def candlestick( ax, opens, highs, lows, closes, width=4, colorup='g', colordown='r', alpha=0.75, ):
    "畫 K 線圖"

    delta = width / 2.

    # 中間的 Box
    barVerts = [((i - delta, open),
                 (i - delta, close),
                 (i + delta, close),
                 (i + delta, open))
                for i, open, close in zip(range(len(opens)), opens, closes)]

    # 下影線
    downSegments = [((i, low), (i, min(open, close)))
                    for i, low, high, open, close in zip(range(len(lows)), lows, highs, opens, closes)]

    # 上影線
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
    ax.autoscale_view()

    # add these last
    rangeCollection.set_alpha(0.4)
    barCollection.set_alpha(0.4)
    ax.add_collection(rangeCollection)
    ax.add_collection(barCollection)

    return [rangeCollection, barCollection]


def volume_overlay(ax,opens, closes, volumes, colorup='g', colordown='r', width=4, alpha=1.0):
    """Add a volume overlay to the current axes.  The opens and closes
    are used to determine the color of the bar.  -1 is missing.  If a
    value is missing on one it must be missing on all

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        a sequence of opens
    closes : sequence
        a sequence of closes
    volumes : sequence
        a sequence of volumes
    width : int
        the bar width in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency

    Returns
    -------
    ret : `barCollection`
        The `barrCollection` added to the axes

    """

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
    ax.add_collection(barCollection)
    ax.update_datalim(corners)

    ax.autoscale_view()
    # add these last
    return [barCollection]


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)


def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x * 1e-3)


def getHalfHours():
    halfhours = ['09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00',
                 '14:30', '15:00', '15:30', '16:00']
    return np.array(halfhours)


def getHalfHourIndex(time, tickhalfhours):
    "找出最接近 tickdate 的日期的 index"
    index = [bisect.bisect_left(time, tick) for tick in tickhalfhours]
    return np.array(index)


class Cursor(object):
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='lightgray')  # the horiz line
        self.ly = ax.axvline(color='lightgray')  # the vert line

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        plt.draw()


# def draw_price_ta(ax0, df):
# df['ma05'] = pd.Series.rolling(df['Close'], window=5).mean()
# df['ma20'] = pd.Series.rolling(df['Close'], window=20).mean()
# df['ma60'] = pd.Series.rolling(df['Close'], window=60).mean()
# ax0.plot(df['ma05'].values, color='m', lw=2, label='MA (5)')
# ax0.plot(df['ma20'].values, color='blue', lw=2, label='MA (20)')
# ax0.plot(df['ma60'].values, color='black', lw=2, label='MA (60)')


def draw_volume_ta(ax1, df):
    pass


def read_large_file(file_handler):
    block = []
    for line in file_handler:
        yield line


def dataGenerator(filepath):
    cur_dir = "/home/wes/rl-trader"
    path = cur_dir+"/"+filepath
    with open(path) as file_handler:
        for line in file_handler:
            yield line.rstrip('\n').split(",")

def minuteStockDataFeeder(filepath):
    one_minute = timedelta(minutes=1)
    end_time = datetime.strptime("16:00", '%H:%M')
    start_time = datetime.strptime("09:30", '%H:%M')
    current_date = ""
    current_time = ""
    gen = dataGenerator(filepath)
    s = ""

    while True:
        if s != "":
            sp = s
        s = next(gen)

        t = datetime.strptime(s[1], '%H:%M')
        d = datetime.strptime(s[0], '%m/%d/%Y')
        if current_date == "":
            current_date = d
        if current_time == "":
            current_time = start_time

        while True:
            if d == current_date:
                if (t == current_time):
                    yield (s[0], s[1], s[2], s[3], s[4], s[5], s[6])
                    break
                if (t > current_time):
                    if(t<end_time):
                        time = "{:02d}:{:02d}".format(current_time.hour, current_time.minute)
                        yield (s[0], time, s[2], s[3], s[4], s[5], s[6])
                        current_time = current_time + one_minute
                    else:
                        break
                if (t <= current_time):
                    break
            else:
                while current_time <= end_time:
                    time = "{:02d}:{:02d}".format(current_time.hour, current_time.minute)

                    yield (datetime.strftime(current_date, '%m/%d/%Y'), time, sp[2], sp[3], sp[4], sp[5], sp[6])
                    current_time = current_time + one_minute

                current_date = d
                current_time = start_time

class PriceChartData():
    def __init__(self, config):

        self.opens = np.zeros( [200 ], dtype=np.float32)
        self.highs = np.zeros([200], dtype=np.float32)
        self.lows = np.zeros([200], dtype=np.float32)
        self.closes = np.zeros([200], dtype=np.float32)
        self.volumes = np.zeros([200], dtype=np.int)
        self.dates = np.zeros([200], dtype=np.object)
        self.times = np.zeros([200], dtype=np.object)


    def add(self, data):
        #print(data[0],data[1],data[2],data[3],data[4],data[5],data[6])

        self.opens[:-1] = self.opens[1:]
        self.highs[:-1] = self.highs[1:]
        self.lows[:-1] = self.lows[1:]
        self.closes[:-1] = self.closes[1:]
        self.volumes[:-1] = self.volumes[1:]
        self.dates[:-1] = self.dates[1:]
        self.times[:-1] = self.times[1:]



        self.dates[-1] = data[0]
        self.times[-1] = data[1]
        self.opens[-1] = data[2]
        self.highs[-1] = data[3]
        self.lows[-1] = data[4]
        self.closes[-1] = data[5]
        self.volumes[-1] = data[6]

    def getData(self,i):
        return self.dates[i],self.times[i] ,self.opens[i] ,self.highs[i],self.lows[i] ,self.closes[i] ,self.volumes[i]


class Chart():


    def  __init__(self,symbol, filepath, colorup='g', colordown='r'):

        self.colorup=colorup
        self.colordown=colordown

        self.dataGen = minuteStockDataFeeder(filepath)

        self.chartData = PriceChartData('')

        #for i in range(0, 200):
        #    self.chartData.add(next(self.dataGen))

        df = pd.DataFrame({ 'Date':self.chartData.dates,
                            'Time': self.chartData.times,
                           'Open': self.chartData.opens,
                           'High': self.chartData.highs,
                           'Low': self.chartData.lows,
                           'Close': self.chartData.closes,
                           'Volume': self.chartData.volumes})

        self.fig = plt.figure(figsize=(16, 12))

        self.fig.suptitle(symbol)

        gs = gridspec.GridSpec(7, 8)
        gs.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.

        self.axPrice = self.fig.add_subplot(gs[0:4, :-1])
        self.axPrice.tick_params(axis="y", direction="in", left=True, right=True)
        self.axPrice.get_xaxis().set_visible(False)
        # Specify tick label size
        self.axPrice.tick_params(axis='x', which='major', labelsize=4)
        self.axPrice.tick_params(axis='x', which='minor', labelsize=0)
        major_ticks = np.arange(-211, 0, 30)
        minor_ticks = np.arange(-211, 0, 5)
        self.axPrice.set_xticks(major_ticks)
        self.axPrice.set_xticks(minor_ticks, minor=True)

        major_ticks = np.arange(-10,110, 10)
        minor_ticks = np.arange(-10,110, 1)

        self.axPrice.set_yticks(major_ticks)
        self.axPrice.set_yticks(minor_ticks, minor=True)

        self.axForecast = self.fig.add_subplot(gs[0:4, 7:])

        self.axForecast.get_yaxis().set_visible(False)
        self.axForecast.set_xticks(np.arange(0, 30, 5))
        self.axForecast.set_xticks(np.arange(0, 30, 1), minor=True)
        self.axForecast.tick_params(axis="x", direction="in", top=True, bottom=False)
        self.axForecast.xaxis.set_ticks_position('top')


        self.axVolume = self.fig.add_subplot(gs[4, :-1], sharex=self.axPrice)

        self.axCash = self.fig.add_subplot(gs[5, :-1])
        self.axCash.get_yaxis().set_visible(False)
        self.axCash.get_xaxis().set_visible(False)

        self.ax5 = self.fig.add_subplot(gs[5, 7:], sharey=self.axPrice)
        self.ax5.get_yaxis().set_visible(False)
        self.ax5.get_xaxis().set_visible(False)

        self.axAction = self.fig.add_subplot(gs[6:, :-1], sharex=self.axPrice)
        self.axAction.get_yaxis().set_visible(False)

        self.ax7 = self.fig.add_subplot(gs[6:, 7:])
        self.ax7.get_yaxis().set_visible(False)
        self.ax7.get_xaxis().set_visible(False)



        tickindex = df.index.values


        self.price_candles = candlestick(self.axPrice,df.Open, df.High, df.Low, df.Close, width=1,
                                                              colorup=colorup, colordown=colordown)




        self.axPrice.set_xticks(tickindex)
        self.axPrice.set_xticks(tickindex)

        self.axPrice.set_ylabel('Price(% of range)', fontsize=8)

        self.axPrice.grid(True)



        self.volume_bars  = volume_overlay(self.axVolume,df.Open, df.Close, df.Volume, colorup=colorup, colordown=colordown, width=1)



        # ax1.set_xticks(tickindex)
        self.axVolume.set_xticks(np.arange(0, len(df.Time), 10))
        self.axVolume.set_xticks(np.arange(0, len(df.Time), 1), minor=True)
        self.axVolume.set_xticklabels(np.array([i - 200 for i in np.arange(0, len(df.Time), 10)]))

        self.axVolume.get_xaxis().set_visible(False)

        self.axVolume.yaxis.tick_right()

        self.axVolume.set_yticks(np.arange(0,10,2))
        self.axVolume.set_yticks(np.arange(0,10,1),minor=True)
        self.axVolume.set_ylabel('Volume', fontsize=8)
        self.axVolume.yaxis.set_label_position('right')
        self.axVolume.grid(True)


    def thousands(self,x):
        'The two args are the value and tick position'
        return '%1.1fK' % (x * 1e-3)

    def millions(self, x):
        'The two args are the value and tick position'
        return '%1.1fM' % (x * 1e-6)



    def update(self,i):
        import time
        tstart = time.time()
        self.chartData.add(next(self.dataGen))

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
        #print(self.df.tail(1))
        ##### axPrice #################
        self.price_candles = candlestick(self.axPrice,  self.df.Open ,self.df.High ,
                                          self.df.Low ,self.df.Close , width=1,
                                         colorup=self.colorup, colordown=self.colordown)


        self.axPrice.set_ylim(-10.0, 110.0)

        self.price_range = [yh - i * (yh - yl) / 10 for i in range(1, 11)]
        self.price_range = [self.price_range[0] + self.price_range[0] - self.price_range[1]] + self.price_range + [
            self.price_range[9] - self.price_range[0] + self.price_range[1]]

        self.ranges = []
        for i in range(0, 12):
            self.ranges.append(self.axPrice.text(0.0, (10 - i) * 10, '${:1.2f}'.format(self.price_range[i])))

        self.last_price = "Date: {}, Time:{}, Open:{:2.2f}, High:{:2.2f}, Low:{:2.2f}, Close:{:2.2f}, Volume:{}".format(
            self.chartData.dates[-1], self.chartData.times[-1], self.chartData.opens[-1],
            self.chartData.highs[-1], self.chartData.lows[-1], self.chartData.closes[-1], self.chartData.volumes[-1])
        #print(self.last_price)
        self.text = self.axPrice.text(0.90, 0.95, self.last_price,
                                      horizontalalignment='right',
                                      verticalalignment='bottom',
                                      transform=self.axPrice.transAxes)


        ##### axVolume #########
        self.volume_bars = volume_overlay(self.axVolume, self.df.Open, self.df.Close,
                                          self.df.Volume, colorup=self.colorup,
                                          colordown=self.colordown, width=1)

        self.axVolume.set_ylim(0,10)


        self.vol_range=[(vh - i * (vh - vl) / 5) for i in range(1, 5)]

        self.vol_ranges = []

        for i in range(0, 4):
            self.vol_ranges.append(self.axVolume.text(190, (4-i)* 2,  '{}'.format(self.thousands(self.vol_range[i]))))

        ##### date change line ######
        s=self.df.Date
        s2 = s.ne(s.shift().bfill()).astype(int)
        xs=s2.diff()[s2.diff() != 0].index.values

        if len(xs)>1:
            x=xs[1]
        else:
            x=0

        self.line=self.axPrice.axvline(x=x,linewidth=.5, color='k',alpha=0.5,linestyle='--')

        ####################
        import gc
        gc.collect()
        print('FPS:', 1 / (time.time() - tstart))
        return self.price_candles+self.volume_bars+[self.text]+self.ranges+[self.line]+ self.vol_ranges

def main():

    symbol='OIH'


    chart=Chart(symbol,'data/oil.txt')


    ani = animation.FuncAnimation(chart.fig, chart.update, interval=100, blit=True)

    plt.show()

if __name__ == '__main__':
  main()