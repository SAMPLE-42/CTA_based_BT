# -*- coding: utf-8 -*-

"""
@Author : fuyurong
@Create : 2023/8/21
@Package: strategy_2.py
@software: PyCharm

策略描述

    ATR策略
        H: 当天的最高价
        L: 当天的最低价
        C_1: 昨天的收盘价
        TR = MAX(H - L, ABS(H - C_1), ABS(L - C_1))
        APR = TR / N
        上轨 = open + k*ATR
        下轨 = open - k*ATR
        open = 前一天的收盘价

        交易逻辑:
            价格突破上轨， 买入开仓
            价格跌穿下轨， 卖出开仓

        改进:
            重采样，调参
            APR计算是某一段时间内价格的真实波动
            可将其视为止盈止损的标准
            但是真实波动幅度均值不能直接预测价格走势及其趋势的稳定性，只能代表交易的频繁性
            一段长时间的低APR可能开始下一个趋势(延续之前或者反转)
            但是非常高的APR通常是由于短时间内价格的大幅上涨 / 下跌造成，不可能长期维持高水平

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import datetime
import time


class ATRStrategy(bt.Strategy):
    params = (
        ('period_atr', 14),
        ('period_ma', 14),
        ('atr_mult', 0.3),
        ('printlog', True),
    )

    def __init__(self):

        # 原始的分钟数据
        self.minute_data = self.datas[0]

        # 重采样的日数据
        self.daily_data = self.datas[1]

        # 重采样的30分数据
        self.minute_data_30 = self.datas[2]

        # 使用日数据计算ATR和移动均线
        self.atr = bt.indicators.ATR(self.minute_data_30, period=self.p.period_atr)
        self.ma = bt.indicators.SimpleMovingAverage(self.daily_data.close, period=self.p.period_ma)

        # 上下轨
        self.upper = self.minute_data_30.close + self.p.atr_mult * self.atr
        self.lower = self.minute_data_30.close - self.p.atr_mult * self.atr

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
            dt = dt or self.data.datetime.date(0)
            print('%s %s' % (dt.isoformat(), txt))

    def next(self):

        current_time = self.datas[0].datetime.time(0)
        # 如果这是策略的首次运行或者自上次运行已经过去了15分钟，则更新时间并继续执行策略逻辑
        if self.last_next_run is None or (current_time.minute - self.last_next_run.minute) % 60 >= 30:
            self.last_next_run = current_time
        else:
            return  # 如果还没有到达下一个15分钟区间，直接返回

        # 记录前一天的日期，以便重置当天的开仓次数
        current_day = self.datas[0].datetime.date(0)
        if self.prev_day is not None and current_day != self.prev_day:
            self.trades_today = 0
        self.prev_day = current_day

        if self.trades_today >= 5:
            # 限制当天的交易次数
            return

        # 每天9点半以后开始交易
        if self.datas[0].datetime.time() < datetime.time(9, 30):
            return

        if self.position:  # 如果有持仓, 检查是否需要平仓
            if self.datas[0].datetime.time() == datetime.time(14, 45) \
                    or self.datas[0].datetime.time() == datetime.time(22, 45):
                # if self.data.datetime.time() == datetime.time(22, 45):
                self.close()
                self.log('当天平仓')

        # 防止午后趋势不确定，不进行交易
        if self.datas[0].datetime.time() > datetime.time(13, 0) \
                and self.datas[0].datetime.time() < datetime.time(23,0):
            return

        if self.order:  # 有挂单，不执行任何操作
            return

        if not self.position:
            # if self.minute_data.close[-1] > self.upper and self.minute_data.close > self.ma:  # 正交叉表示价格突破了上轨
            if self.minute_data.close[-1] > self.upper:
                self.buy()
            # elif self.minute_data.close[-1] < self.lower and self.minute_data.close < self.ma:  # 负交叉表示价格突破了下轨
            elif self.minute_data.close[-1] < self.lower:
                self.sell(size=100)

        # elif self.position.size > 0 and self.minute_data.close[-1] < self.lower and self.minute_data.close < self.ma:
        elif self.position.size > 0 and self.minute_data.close[-1] < self.lower:
            self.close()

        # elif self.position.size < 0 and self.minute_data.close[-1] > self.upper and self.minute_data.close > self.ma:
        elif self.position.size < 0 and self.minute_data.close[-1] > self.upper:
            self.close()

        # 记录交易执行记录

    def notify_order(self, order):
        # 如果order为submitted / accepted, 返回空
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            # 如果order为buy / sell， 则记录价格
            # self.trade_count += 1

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




