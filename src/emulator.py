
import numpy as np
from datetime import datetime, timedelta
from numpy import genfromtxt
import pandas as pd
import random
from market_chart import Chart
from market_chart_widget import ChartWidget,PriceData,PositionHistoryData,RewardHistoryData
from threading import Thread,Event
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def dataGenerator( filepath):

    with open(filepath) as file_handler:
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


def processOneDayData(df):
    one_minute = timedelta(minutes=1)
    end_time = datetime.strptime("16:00", '%H:%M')
    start_time = datetime.strptime("09:30", '%H:%M')
    current_date = ""
    current_time = ""
    gen = df.iterrows()
    s = pd.Series()
    sp = pd.Series()
    try:
        while True:
            if not s.empty:
                sp = s

            s = next(gen)[1]

            t = datetime.strptime(s[1].decode("utf-8"), '%H:%M')
            d = datetime.strptime(s[0].decode("utf-8"), '%m/%d/%Y')
            if current_date == "":
                current_date = d
            if current_time == "":
                current_time = start_time

            while True:
                if d == current_date:
                    if (t == current_time):
                        yield (s[0].decode("utf-8"), s[1].decode("utf-8"), s[2], s[3], s[4], s[5], s[6])
                        current_time = current_time + one_minute
                        break
                    if (t > current_time):
                        if (t < end_time):
                            time = "{:02d}:{:02d}".format(current_time.hour, current_time.minute)
                            yield (s[0].decode("utf-8"), time, s[2], s[3], s[4], s[5], s[6])
                            current_time = current_time + one_minute
                        else:
                            break
                    if (t <= current_time):
                        break

    except StopIteration:
        while current_time <= end_time:
            time = "{:02d}:{:02d}".format(current_time.hour, current_time.minute)

            yield (datetime.strftime(current_date, '%m/%d/%Y'), time, sp[2], sp[3], sp[4], sp[5], sp[6])
            current_time = current_time + one_minute

    finally:
        del gen


class Market:
    def __init__(self,config, symbol, filepath, open_cost,unit):
        super(Market, self).__init__()
        self.chart=None
        self.symbol= symbol
        self.config = config
        data = genfromtxt(filepath, dtype="S10,S5,f8,f8,f8,f8,int32",
                          names=['sdate', 'stime', 'open', 'high', 'low', 'close', 'volume'], delimiter=",")

        self.df = pd.DataFrame(data)

        def conv_time(x):
            return datetime.strptime(x.decode("utf-8"), '%H:%M')

        def conv_date(x):
            return datetime.strptime(x.decode("utf-8"), '%m/%d/%Y')

        self.df['time'] = self.df['stime'].apply(conv_time)
        self.df['date'] = self.df['sdate'].apply(conv_date)

        self.dates = self.df['date'].unique()

        self.open_cost = open_cost
        self.unit = unit

        self.n_action = 3
        self.t_max=390

        self.n_actions=[0,1,2,3,4,5,6]
        self.action_labels = ['stay_neutral', 'buy_open', 'sell_close','hold_long','sell_open','buy_close','hold_short']
        self.time = datetime.strptime("09:30", '%H:%M')
        self.position =0
        self.order_price = 0.
        self.current_price = 0.
        self.time_since=0
        self.data_gen = None
        self.data = PriceData(200-self.config.forecast_window)
        self.t=0
    def enable_chart(self,is_widget=False, parent=None, app = None):
        if (is_widget):
            self.chart= ChartWidget(self.symbol,18,12,parent=parent,app=app)
        else:
            self.chart=Chart(self.symbol)

    def reset(self):
        #self.position = 0. position can carry over a day in theory.
        #self.order_price =0.
        self.current_price =0.
        self.data_gen = None
        self.data.reset()
        self.t=0 
        self.time_since=0

    def a_given_past_day(self, aday):
        self.reset()
        dt = np.datetime64(datetime.strptime(aday, '%m/%d/%Y'))

        i = np.where(self.dates == dt)[0][0]

        self.today = datetime.utcfromtimestamp(self.dates[i].astype('O') / 1e9)

        if self.today.weekday() == 6:
            i = i + 1
        if self.today.weekday() == 5:
            i = i - 1

        self.today = datetime.utcfromtimestamp(self.dates[i].astype('O') / 1e9)

        self.yesterday = datetime.utcfromtimestamp(self.dates[i - 1].astype('O') / 1e9)

        if self.yesterday.weekday() == 6:
            self.yesterday = datetime.utcfromtimestamp(self.dates[i - 3].astype('O') / 1e9)

        df1 = self.df[self.df['date'] == self.yesterday]
        df2 = self.df[self.df['date'] == self.today]

        pgen = processOneDayData(df1)
        datagen = processOneDayData(df2)

        for i in range(391):
            r = next(pgen)
            self.data.add(r)

        self.data_gen = datagen

        time = datetime.strptime(self.data.times[-1], '%H:%M')
        return self.data, self.today, time, self.position, 0, 0, 0, False, self.get_valid_actions()

    def random_past_day(self):

        self.reset()
        random.seed()

        i = random.randint(1, len(self.dates) - 1)

        self.today=datetime.utcfromtimestamp(self.dates[i] .astype('O') / 1e9)

        if self.today.weekday()==6:
            i=i+1
        if self.today.weekday()==5:
            i=i-1

        self.today = datetime.utcfromtimestamp(self.dates[i].astype('O') / 1e9)

        self.yesterday = datetime.utcfromtimestamp(self.dates[i-1].astype('O') / 1e9)

        if self.yesterday.weekday()==6:
            self.yesterday=datetime.utcfromtimestamp(self.dates[i-3].astype('O') / 1e9)


        df1 = self.df[self.df['date'] ==  self.yesterday]
        df2 = self.df[self.df['date'] == self.today]

        pgen = processOneDayData(df1)
        datagen = processOneDayData(df2)

        for i in range(391):
            r = next(pgen)
            self.data.add(r)

        self.data_gen = datagen

        time = datetime.strptime(self.data.times[-1] ,'%H:%M')
        return self.data,  self.today, time, self.position, 0, 0, 0, False, self.get_valid_actions()


    def get_valid_actions(self):
        # actions:
        # # 0= stay side line
        # # 1 = open buy, 2 = sell to close, 3 = hold long position
        # # 4 = sell to open, 5 = buy to close, 6 = hold short position
        #
        if self.position==0: #neutral
            return [0, 1, 4]  # noop, Buy to Open,   Sell to Open
        if self.position >0: #long
            return [2, 3]  #   sell to close, hold long
        if self.position <0: #short
            return [5,6]   # buy to close, hold short
        #
        # if self.position == 0:  # neutral
        #     return [0, 1, 2]  # hold, Buy , Sell
        # if self.position > 0:  # long
        #     return [0, 2]  # hold, sell to close,
        # if self.position < 0:  # short
        #     return [0, 1]  # hold, buy to close


    def step(self, action):

        done = False
        reward =0.

        d=next(self.data_gen)
        self.data.add(d)

        self.current_price = (self.data.get_t_data(0)[3] + self.data.get_t_data(0)[5]) / 2.
        #self.time_since=0
        unitcost = self.open_cost / self.unit

        # let's use Kalman filter to find a true price (state) with all prices, volumes can be used as observations
        #




        # if action == 0:  # noop
        #     reward = 0.
        #     if self.position > 0:  # consider paper gain and lost
        #
        #         self.time_since+=1
        #
        #     if self.position < 0.:  # consider paper gain or loss
        #
        #         self.time_since+=1
        #
        # elif action == 1:  # Buy
        #     if self.position > 0:
        #         raise ValueError('can not take this action:' + str(action))
        #
        #     if self.position == 0:
        #         self.position = 1
        #         self.order_price = self.current_price
        #
        #         reward = -unitcost / self.order_price
        #
        #     if self.position < 0:
        #
        #         self.time_since =0
        #
        #         reward = -1 * ((self.current_price - self.order_price) - unitcost) / self.order_price
        #         self.position = 0
        #
        # elif action == 2:  # Sell
        #     if self.position < 0:
        #         raise ValueError('can not take this action:' + str(action))
        #
        #     if self.position == 0:
        #         self.position = -1
        #         self.order_price = self.current_price
        #         reward = -self.open_cost / self.order_price
        #
        #     if self.position > 0:
        #
        #         self.time_since =0
        #
        #         reward = ((self.current_price - self.order_price) - unitcost) / self.order_price
        #         self.position = 0

        if action == 0:  # noop
            reward = 0.
            if self.position !=0:
                raise ValueError('can not take this action:' + str(action))
            self.time_since=0

        elif action == 1:  # Buy Open

            if self.position != 0:
                raise ValueError('can not take this action:' + str(action))

            if self.position == 0:  # open a long position
                self.position = 1
                self.order_price = self.current_price
                reward = -unitcost
                self.time_since=1

        elif action == 2:  # Sell Close
            if self.position <= 0:
                raise ValueError('can not take this action:' + str(action))

            if self.position > 0:  # close long position
                reward = self.position * (self.current_price - self.order_price) - unitcost
                self.position = 0
                self.time_since+=1


        elif action == 3:  # hold long
            if self.position <= 0:
                raise ValueError('can not take this action (double down): ' + str(action))
            self.time_since += 1
        elif action == 4:  # sell to Open
            if self.position != 0:
                raise ValueError('can not take this action (double down): ' + str(action))

            if self.position == 0:  # open a short position
                self.position = -1
                self.order_price = self.current_price
                reward = -unitcost
                self.time_since = 1

        elif action == 5:  # Sell Close
            if self.position >= 0:
                raise ValueError('can not take this action:' + str(action))

            if self.position < 0:  # close a short position
                reward = self.position * (self.current_price - self.order_price) - unitcost
                self.position = 0
                self.time_since += 1

        elif action == 6:  # hold short
            if self.position >= 0:
                raise ValueError('can not take this action:' + str(action))
            if self.position < 0:
                self.time_since += 1

        else:
            raise ValueError('no such action: ' + str(action))


        #new reward function, using lambda decay on price differences
        # r = dirction*position* ( p0+lambda**1*p1+lambda**2*p2+ ... + lambda**n*pn)
        # where n= 30, lambda = 0.73 ,
        # direction = 1 when buy -1 when sell
        # position -1 when short, +1 when long, 0 when neutral
        #
        # p = np.add(self.data.highs[-self.config.observation_window:], self.data.lows[-self.config.observation_window:]) / 2
        # d = np.diff(p)
        #
        # l = 0.6
        # e = [d[-i - 1] * l ** i for i in range(self.config.observation_window)]
        # # print(e)
        # r=np.sum(e)
        #
        #
        # if action == 0:  # noop
        #     reward = 0.
        #     if self.position != 0:
        #         raise ValueError('can not take this action:' + str(action))
        #     self.time_since=0
        #
        # elif action == 1:  # Buy Open
        #
        #     if self.position != 0:
        #         raise ValueError('can not take this action:' + str(action))
        #
        #     if self.position == 0:  # open a long position
        #         self.position = 1
        #         self.order_price = self.current_price
        #         reward = r
        #         self.time_since=1
        #
        # elif action == 2:  # Sell Close
        #     if self.position <= 0:
        #         raise ValueError('can not take this action:' + str(action))
        #
        #     if self.position > 0:  # close long position
        #         reward = -1*r
        #         self.position = 0
        #         self.time_since+=1
        #
        #
        # elif action == 3:  # hold long
        #     if self.position <= 0:
        #         raise ValueError('can not take this action (double down): ' + str(action))
        #     self.time_since += 1
        #     reward=r
        #
        # elif action == 4:  # sell to Open
        #     if self.position != 0:
        #         raise ValueError('can not take this action (double down): ' + str(action))
        #
        #     if self.position == 0:  # open a short position
        #         self.position = -1
        #         self.order_price = self.current_price
        #         reward = -1*r
        #         self.time_since = 1
        #
        # elif action == 5:  # Sell Close
        #     if self.position >= 0:
        #         raise ValueError('can not take this action:' + str(action))
        #
        #     if self.position < 0.:  # close a short position
        #         reward = -1 * -1 * r
        #         self.position = 0.
        #         self.time_since += 1
        #
        # elif action == 6:  # hold short
        #     if self.position >= 0:
        #         raise ValueError('can not take this action:' + str(action))
        #     if self.position < 0.:
        #         self.time_since += 1
        #     reward = -1 * r
        #
        # else:
        #     raise ValueError('no such action: ' + str(action))

        self.t += 1
        self.action=action
        self.time = datetime.strptime(self.data.times[-1] , '%H:%M')
        self.terminal = self.t==self.t_max
        self.reward = reward

        return self.data,   self.today, self.time, self.position, self.order_price,self.current_price, reward, self.t==self.t_max, self.get_valid_actions()

    def play(self):

        if self.chart == None:
            return

        self.random_past_day()
        terminal = False
        self.chart.chartData = self.data
        self.chart.position_data.reset()
        self.chart.rewards_data.reset()

        while not terminal:
            actions = self.get_valid_actions()
            rnd_action = actions[random.randint(0, len(actions) - 1)]
            data, today, time, position, order_price,current_price, reward, terminal, actions = self.step(rnd_action)

            self.chart.position_data.add(position)
            self.chart.rewards_data.add(reward)

            self.chart.chartData = data

            self.chart.forecasts = np.random.uniform(0, 100, self.config.forecast_window)

            self.chart.forecast_history = (data.highs[-self.config.forecast_window:] + data.lows[-self.config.forecast_window:]) / 2.0

            self.chart.forecast_history = self.chart.forecast_history + self.chart.forecast_history * (
                    0.5 - np.random.random(self.config.forecast_window)) * .2
            self.chart.update()

            print(rnd_action,today,time, position, reward, terminal, actions)

    def monitor_start(self):
        if self.chart == None:
            self.enable_chart()


        if self.chart.position_data == None:
            self.chart.position_data = PositionHistoryData(200-self.config.forecast_window)
        if self.chart.rewards_data ==None:
            self.chart.rewards_data = RewardHistoryData(200-self.config.forecast_window)

        plt.show()

    def monitor_refresh(self,data,position,reward,forecasts,forecast_history):


        self.chart.chartData=data
        self.chart.position_data.add(position)
        self.chart.rewards_data.add(reward)
        self.chart.forecasts = forecasts
        self.chart.forecast_history = forecast_history

        self.chart.update()

def test():
    env = Market('OIH','/Users/wweschen/rl-trader/data/oil.txt', 4, 1000)

    env.enable_chart( )

    env.chart.position_data = PositionHistoryData(200-16)
    env.chart.rewards_data = RewardHistoryData(200-16)




    class MyThread(Thread):
        def __init__(self, event):
            Thread.__init__(self)
            self.stopped = event

        def run(self):
            while not self.stopped.wait(0):
                env.play()

    stopFlag = Event()
    thread = MyThread(stopFlag)
    thread.start()

    plt.show()



if __name__ == '__main__':
   test()
