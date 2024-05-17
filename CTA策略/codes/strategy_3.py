# -*- coding: utf-8 -*-

"""
@Author : fuyurong
@Create : 2023/8/25
@Package: strategy_3.py
@software: PyCharm

策略描述
    菲阿里四价策略
        上轨：昨日最高价
        下轨: 昨日最低价

        交易逻辑:
            未持仓:
                突破上轨买入，突破下轨卖出
            持仓:
                持仓数大于0, 收盘价低于开盘价平仓
                持仓数小于0, 收盘价高于开盘价平仓

        改进：
            设置时间间隔
            ATR止盈止损

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import datetime
import time

class SIJIAStrategy(bt.Strategy):
    params = (
        ('printlog', True),
        ('atr_period', 5),                  # ATR的周期，默认设置为14
        ('atr_mult_stoploss', 3),           # ATR的倍数，用于计算止损
        ('atr_mult_takeprofit', 3),         # ATR的倍数，用于计算止盈
        ('use_stop', True)                  # 是否使用止盈止损
    )

    def __init__(self):

        # 原始的分钟数据
        self.minute_data = self.datas[0]

        # 重采样的日数据
        self.daily_data = self.datas[1]

        # 重采样的30分数据
        self.minute_data_30 = self.datas[2]

        # 上下轨
        # self.upper = self.daily_data.high[-1]
        # self.lower = self.daily_data.low[-1]
        self.upper = bt.indicators.Highest(self.daily_data.high, period=1)
        self.lower = bt.indicators.Lowest(self.daily_data.low, period=1)
        self.atr = bt.indicators.ATR(self.minute_data_30, period=self.p.atr_period)

        # 跟踪信息
        self.order = None                           # 交易指令
        self.buy_price = 0                          # 买入价
        self.buy_comm = 0                           # 买入手续费
        self.sell_price = 0                         # 卖出价
        self.sell_comm = 0                          # 卖出手续费
        self.buy_size = 0                           # 买入仓位
        self.trade_count = 0                        # 交易总次数
        self.trades_today = 0                       # 当天交易总次数， 限制开仓次数
        self.prev_day = None                        # 用于跟踪上一天的日期
        self.start_value = self.broker.get_value()  # 初始资金，用以计算年度收益率
        self.stop_loss_price = None                 # 止损价格
        self.take_profit_price = None               # 止盈价格
        self.last_next_run = None                   # 时间间隔追踪
        self.trade_days = 0

        # 输出日志

    def log(self, txt, dt=None, droprint=False):
        """
            日志函数， 用于统一输出日志

        """
        if self.params.printlog or droprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s %s' % (dt.isoformat(), txt))

    def next(self):

        # 记录前一天的日期，以便重置当天的开仓次数
        current_day = self.datas[0].datetime.date(0)
        if self.prev_day is not None and current_day != self.prev_day:
            self.trades_today = 0
        self.prev_day = current_day

        current_time = self.datas[0].datetime.time(0)
        # 如果这是策略的首次运行或者自上次运行已经过去了15分钟，则更新时间并继续执行策略逻辑
        if self.last_next_run is None or (current_time.minute - self.last_next_run.minute) % 60 >= 30:
            self.last_next_run = current_time
        else:
            return  # 如果还没有到达下一个15分钟区间，直接返回

        if self.trades_today >= 10:
            # 限制当天的交易次数
            return

        # 每天9点半以后开始交易
        if self.datas[0].datetime.time() < datetime.time(9, 30):
            return

        if self.position:  # 如果有持仓, 检查是否需要平仓
            if self.datas[0].datetime.time() == datetime.time(14, 45) \
                    or self.datas[0].datetime.time() == datetime.time(
                    22, 45):
                # if self.data.datetime.time() == datetime.time(22, 45):
                self.close()
                self.log('当天平仓')

        # 防止午后趋势不确定，不进行交易
        if self.datas[0].datetime.time() > datetime.time(13, 0) \
                and self.datas[0].datetime.time() < datetime.time(23,0):
            return

        if self.order:  # 有挂单，不执行任何操作
            return

        # 使用ATR来设置止损和止盈价格
        if not self.position:
            if self.minute_data.close > self.upper:
                self.buy()
                self.stop_loss_price = self.minute_data.close - self.params.atr_mult_stoploss * self.atr[0]
                self.take_profit_price = self.minute_data.close + self.params.atr_mult_takeprofit * self.atr[0]


            elif self.minute_data.close < self.lower:
                self.sell(size=100)

                self.stop_loss_price = self.minute_data.close + self.params.atr_mult_stoploss * self.atr[0]
                self.take_profit_price = self.minute_data.close - self.params.atr_mult_takeprofit * self.atr[0]


        elif self.position.size > 0 and self.minute_data.close < self.daily_data.open:
            self.close()

        elif self.position.size < 0 and self.minute_data.close > self.daily_data.open:
            self.close()

        # 如果价格触及止损或止盈价格，平仓

        if self.p.use_stop:

            if self.position.size > 0:
                if self.minute_data.close <= self.stop_loss_price or self.minute_data.close >= self.take_profit_price:
                    self.close()

            elif self.position.size < 0:
                if self.minute_data.close >= self.stop_loss_price or self.minute_data.close <= self.take_profit_price:
                    self.close()


        self.trade_days += 1  # 记录交易天数

        # 记录交易执行记录

    def notify_order(self, order):
        # 如果order为submitted / accepted, 返回空
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:

            if order.isbuy():
                self.log(f"买入: \n价格: {order.executed.price},\
                成本: {order.executed.value},\
                手续费: {order.executed.comm}")


            else:
                self.log(f"卖出: \n价格: {order.executed.price},\
                成本: {order.executed.value},\
                手续费: {order.executed.comm}")

            self.trades_today += 1


        # 交易失败 / 指令取消
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("交易失败")

        self.order = None

    # 记录交易收益情况
    def notify_trade(self, trade):

        if trade.isclosed:
            self.log(f"策略收益: \n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}")

    def stop(self):

        # 从TradeAnalyzer获取数据

        # 总交易次数
        total_trades = self.analyzers.trade_analyzer.get_analysis()['total']['total']

        # 做多次数
        long_trades = self.analyzers.trade_analyzer.get_analysis()['long']['total']
        # 做多盈利次数
        long_wins = self.analyzers.trade_analyzer.get_analysis()['long']['won']

        # 做空次数
        short_trades = self.analyzers.trade_analyzer.get_analysis()['short']['total']
        # 做空盈利次数
        short_wins = self.analyzers.trade_analyzer.get_analysis()['short']['won']

        # 盈利次数
        win_trades = self.analyzers.trade_analyzer.get_analysis()['won']['total']

        # 盈利总数 / 亏损总数
        profit = self.analyzers.trade_analyzer.get_analysis()['won']['pnl']['total']
        loss = self.analyzers.trade_analyzer.get_analysis()['lost']['pnl']['total']

        if long_trades > 0:
            # 做多胜率
            long_win_rate = long_wins / long_trades
        else:
            long_win_rate = 0

        if short_trades > 0:
            # 做空胜率
            short_win_rate = short_wins / short_trades
        else:
            short_win_rate = 0

        # 总胜率
        total_win_rate = win_trades / total_trades

        # 总收益
        end_value = self.broker.get_value()
        total_return = (end_value / self.start_value) - 1

        # 盈亏比
        profit_loss_ratio = profit / abs(loss)

        # 我们使用总交易数除以数据长度来模拟换手率，但请注意这不是真正的换手率定义
        turnover_rate = total_trades / len(self.datas[0])

        self.log(f"期末总资金: {self.broker.getvalue():.2f}", droprint=True)

        self.log(f"交易次数: {total_trades},\n"
                 f"总胜率: {total_win_rate},\n"
                 f"总收益: {total_return},\n"
                 f"做多次数: {long_trades},\n"
                 f"做空次数: {short_trades},\n"
                 f"盈亏比: {profit_loss_ratio}\n", )

        self.results = (
            self.analyzers.cumulative_return.get_analysis()['cumulative_return'],
            turnover_rate,
            self.analyzers.mysharpe.get_analysis(),
            self.analyzers.drawdown.get_analysis()['max']['drawdown'],
            total_trades,
            total_win_rate,
            total_return,
            long_trades,
            short_trades,
            profit_loss_ratio,
            long_win_rate,
            short_win_rate,
            self.analyzers.cumulative_return.get_analysis()['dates'],
        )