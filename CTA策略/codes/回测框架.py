import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import backtrader.analyzers as btanalyzers
import seaborn as sns
import datetime
import time


class teststrtegy(bt.Strategy):
    # 参数设置
    params = (
        ('N', 30),
        ('K1', 0.5),
        ('K2', -0.5),
        ('printlog', True),
        ('ATR_period', 30),
        ('ATR_mult', 1.5),
        ('stop_loss', 0.01),  # 止损2%
        ('take_profit', 0.04),  # 止盈5%
        ('use_stop', True),  # 是否使用止盈止损, 默认为True
        ('slippage', 0.0002),  # 滑点

    )

    # 初始化
    def __init__(self):

        # 期货基本信息
        self.close = self.data.close  # 收盘价
        self.open = self.data.open  # 开盘价
        self.high = self.data.high  # 最高价
        self.low = self.data.low  # 最低价

        # 所需指标
        self.HH = bt.indicators.Highest(self.data.high, period=self.p.N)
        self.LL = bt.indicators.Lowest(self.data.low, period=self.p.N)
        self.LC = bt.indicators.Lowest(self.data.close, period=self.p.N)
        self.HC = bt.indicators.Highest(self.data.close, period=self.p.N)

        # 计算ATR，用以过滤行情
        self.atr = bt.indicators.ATR(self.data, period=self.p.ATR_period)

        # 定义Range / BuyLine / SellLine
        self.range = bt.indicators.Max(
            (self.HH - self.LC), (self.HC - self.LL)
        )

        self.buy_line = self.open + self.p.K1 * self.range
        self.sell_line = self.open + self.p.K2 * self.range

        # 价格与上下轨的交叉
        self.buy_signal = bt.ind.CrossOver(self.close, self.buy_line)
        self.sell_signal = bt.ind.CrossOver(self.close, self.sell_line)

        # 市场趋势信号
        # self.market_status = bt.ind.crossover(self.short_ma, self.long_ma)

        # 跟踪信息
        self.order = None  # 交易指令
        self.buy_price = 0  # 买入价
        self.buy_comm = 0  # 买入手续费
        self.sell_price = 0  # 卖出价
        self.sell_comm = 0  # 卖出手续费
        self.buy_size = 0  # 买入仓位
        self.buy_count = 0  # 执行买入次数
        self.daily_opens = 0  # 当天开仓次数
        self.trade_count = 0  # 交易总次数
        self.trades_today = 0  # 当天交易总次数， 限制开仓次数
        self.long_count = 0  # 买入次数
        self.short_count = 0  # 卖空次数
        self.trade_days = 0  # 交易总天数
        self.winning_count = 0  # 胜率
        self.winning_amount = 0  # 盈利金额
        self.losing_amount = 0  # 亏损金额
        self.prev_day = None  # 用于跟踪上一天的日期
        self.start_value = self.broker.get_value()  # 初始资金，用以计算年度收益率
        self.stop_loss_price = None  # 止损价格
        self.take_profit_price = None  # 止盈价格

    # 输出日志
    def log(self, txt, dt=None, droprint=False):
        """
            日志函数， 用于统一输出日志

        """
        if self.params.printlog or droprint:
            dt = dt or self.data.datetime.date(0)
            print('%s %s' % (dt.isoformat(), txt))

    # 策略主体
    def next(self):

        atr_limit = self.atr[0] * self.p.ATR_mult

        # 从交易逻辑来看, 应该使用前一时刻的收盘价作为交易价格
        # pre_close = self.close[-1]

        current_day = self.data.datetime.date(0)
        if self.prev_day is not None and current_day != self.prev_day:
            self.trades_today = 0
        self.prev_day = current_day

        if self.trades_today >= 5:
            # 限制当天的交易次数
            return

        if self.params.use_stop and self.position:
            # 如果有多仓
            if self.position.size > 0:
                # 设置止盈止损价格
                if not self.stop_loss_price or not self.take_profit_price:
                    self.stop_loss_price = self.buy_price * (1 - self.params.stop_loss)
                    self.take_profit_price = self.buy_price * (1 + self.params.take_profit)

                # 检查是否触发止盈或止损
                if self.data.close[0] <= self.stop_loss_price or self.data.close[0] >= self.take_profit_price:
                    self.close()
                    self.stop_loss_price = None
                    self.take_profit_price = None
                    self.log('止盈止损平仓')
            # 如果有空仓
            elif self.position.size < 0:
                # 设置止盈止损价格
                if not self.stop_loss_price or not self.take_profit_price:
                    self.stop_loss_price = self.sell_price * (1 + self.params.stop_loss)
                    self.take_profit_price = self.sell_price * (1 - self.params.take_profit)

                # 检查是否触发止盈或止损
                if self.data.close[0] >= self.stop_loss_price or self.data.close[0] <= self.take_profit_price:
                    self.close()
                    self.stop_loss_price = None
                    self.take_profit_price = None
                    self.log('止盈止损平仓')

        if self.position:
            if self.data.datetime.time() == datetime.time(14, 45) or self.data.datetime.time() == datetime.time(22, 45):
                # if self.data.datetime.time() == datetime.time(22, 0):
                # 如果有持仓, 检查是否需要平仓， 在收盘前1小时执行,避免尾盘的极端行情

                self.close()
                self.log('当天平仓')

        # 每天9点半以后开始交易
        if self.data.datetime.time() < datetime.time(9, 30):
            return

        # 早盘以后不交易
        if self.data.datetime.time() > datetime.time(13, 0) and self.data.datetime.time() < datetime.time(23, 0):
            return

        if self.order:
            # 对于未完成订单，不执行任何操作
            return

        # 价格向上突破， 若持有空仓, 先平仓, 再开多仓(否则直接开多仓)
        if self.buy_signal > 0:

            if self.position.size < 0:
                # 有空单, 平仓再开多
                self.close()
                self.log('卖出所有空单')

            # if self.close < atr_limit:
            self.order = self.buy()

            self.long_count += 1  # 记录开多仓次数
            self.trades_today += 1

            # else:
            # 价格波动过大，不开仓
            # self.log('ATR波动过大,本周期不买入')

        # 价格向下突破, 若持有多仓, 先平仓, 再开空仓(否则直接开空仓)
        if self.sell_signal < 0:

            if self.position.size > 0:
                # 有多单， 平仓再空
                self.close()
                self.log('卖出所有多单')

            # 添加ATR波动性过滤
            # if self.close < atr_limit:
            self.order = self.sell()
            self.short_count += 1  # 记录开空仓次数
            self.trades_today += 1

            # else:
            # self.log('ATR波动过大,本周期不卖出')

        self.trade_days += 1  # 记录交易天数

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

                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm

            else:
                self.log(f"卖出: \n价格: {order.executed.price},\
                成本: {order.executed.value},\
                手续费: {order.executed.comm}")

                self.sell_price = order.executed.price
                self.sell_price = order.executed.comm

        # 交易失败 / 指令取消
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("交易失败")

        # 设置self.order = None, 避免代码在一次买入或卖出时被未决订单锁住
        self.order = None

    # 记录交易收益情况
    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"策略收益: \n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}")

            if trade.pnlcomm > 0:
                # 记录获利次数
                self.winning_count += 1

                # 记录获利收益
                self.winning_amount += trade.pnlcomm

            else:
                # 记录亏损数额
                self.losing_amount += trade.pnlcomm

    def stop(self):
        self.trade_count = self.long_count + self.short_count

        # 胜率 = 获利次数 / 总交易次数
        self.win_rate = self.winning_count / self.trade_count if self.trade_count else 0

        # 年化收益率
        end_value = self.broker.get_value()
        total_return = (end_value / self.start_value) - 1

        # 假设一天交易8小时，每小时交易60分钟
        minutes_per_year = 252 * 8 * 60
        years = self.trade_days * 8 * 60 / minutes_per_year
        self.annual_return = (1 + total_return) ** (1 / years) - 1

        # 计算annual_volatility
        self.daily_returns = np.array(self.analyzers.cumulative_return.get_analysis()['daily_return'])
        self.annual_volatility = np.std(self.daily_returns) * np.sqrt(252)

        # 计算return_risk_ratio
        if self.annual_volatility != 0:  # 防止除以零
            self.return_risk_ratio = self.annual_return / self.annual_volatility
        else:
            self.return_risk_ratio = np.nan

        # 盈亏比 = 盈利总金额 / 亏损总金额
        self.profit_loss_ratio = self.winning_amount / abs(self.losing_amount) if self.losing_amount else float('inf')

        # 卡玛比率
        self.calma = self.annual_return / self.analyzers.drawdown.get_analysis()['max']['drawdown']

        # 可添加组合收益
        self.log(f"期末总资金: {self.broker.getvalue():.2f}", droprint=True)
        self.log(f"交易次数: {self.trade_count},\n"
                 f"胜率: {self.win_rate},\n"
                 f"年化波动率: {self.annual_volatility},\n"
                 f"收益风险比: {self.return_risk_ratio},\n"
                 f"做多次数: {self.long_count},\n"
                 f"做空次数: {self.short_count},\n"
                 f"盈亏比: {self.profit_loss_ratio}")

        self.results = (self.params.K1, self.params.K2,
                        self.calma,
                        self.analyzers.cumulative_return.get_analysis()['cumulative_return'],
                        self.analyzers.turnover.get_analysis()['turnover'],
                        self.analyzers.mysharpe.get_analysis(),
                        self.analyzers.drawdown.get_analysis()['max']['drawdown'],
                        self.trade_count,
                        self.win_rate,
                        self.annual_return,
                        self.annual_volatility,
                        self.return_risk_ratio,
                        self.long_count,
                        self.short_count,
                        self.profit_loss_ratio)


class TradeSizer(bt.Sizer):
    params = (('stake', 5),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return self.p.stake
        position = self.broker.getposition(data)
        if not position.size:
            return 0
        else:
            return position.size
        return self.p.stake


# 累计收益分析器
class CumulativeReturn(bt.Analyzer):
    def __init__(self):
        self.values = []  # 保存每天的组合价值
        self.daily_returns = []  # 日度收益率
        self.cumulative_returns = []  # 累计收益率
        self.prev_date = None

    def next(self):
        current_date = self.strategy.datetime.date(0)
        current_value = self.strategy.broker.get_value()

        # 如果日期发生变化或是策略的第一天
        if self.prev_date is None or current_date != self.prev_date:

            if self.prev_date is not None:  # 不是策略的第一天
                daily_return = (current_value / self.values[-1]) - 1
                self.daily_returns.append(daily_return)

                cum_return = (1 + daily_return) * (
                            1 + self.cumulative_returns[-1]) - 1 if self.cumulative_returns else daily_return
                self.cumulative_returns.append(cum_return)

            self.values.append(current_value)
            self.prev_date = current_date

    def get_analysis(self):
        return {
            "daily_return": self.daily_returns,
            "cumulative_return": self.cumulative_returns,
            "values": self.values
        }


# 换手率分析器
class Turnover(bt.Analyzer):
    def __init__(self):
        self.turnover = 0.
        self.total_vol = 0
        self.daily_portfolio_values = []  # 新增列表记录每日组合价值

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.total_vol += abs(order.executed.size)

    def next(self):
        # 在每个时间步保存投资组合的价值
        self.daily_portfolio_values.append(self.strategy.broker.get_value())

    def get_analysis(self):
        average_portfolio_value = sum(self.daily_portfolio_values) / len(self.daily_portfolio_values)

        # 重新定义换手率
        if average_portfolio_value == 0:
            self.turnover = 0
        else:
            self.turnover = self.total_vol / average_portfolio_value

        return {"turnover": self.turnover}


# 定价交易费率
class CustomCommScheme(bt.CommInfoBase):
    params = (
        ('commission', 0.0003),  # 默认百分比手续费
        ('open_cost', 0),  # 固定费用
        ('open_rate', 1.00e-04),  # 比率费用
        ('close_cost', 0),
        ('close_rate', 1.00e-04),
        ('mult', 10),  # 乘数
        ('leverage', 10)
    )

    def _getcommission(self, size, price, pseudoexec):
        """
            计算交易手续费
        """

        size_abs = abs(size) / self.p.leverage
        # 基于交易大小和价格计算基本手续费
        comm = self.p.commission * size_abs * price * self.p.mult

        # 固定开仓费用为0时，使用比率费用
        if self.p.close_cost != 0:
            if size > 0:
                comm += self.p.open_cost
            else:
                comm += self.p.close_cost
        else:
            if size > 0:
                comm += size_abs * price * self.p.mult * self.p.open_rate
            else:
                comm += size_abs * price * self.p.mult * self.p.close_rate

        return comm


if __name__ == '__main__':

    start_time = time.time()
    cerebro = bt.Cerebro(stdstats=False)

    # 添加数据

    datas = bt.feeds.GenericCSVData(
        dataname='D:\python_projects\pythonProject1\RB_sort.csv',
        fromdate=datetime.datetime(2017, 5, 16),
        todate=datetime.datetime(2023, 8, 17),
        datetime=11,
        open=0,
        high=2,
        low=3,
        close=1,
        volume=4,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        timeframe=bt.TimeFrame.Minutes,
    )

    # 添加数据
    cerebro.adddata(datas)

    cerebro.addstrategy(teststrtegy, printlog=True, slippage=0.0002, use_stop=False)
    # 调参设置， 注释不调参
    # cerebro.optstrategy(teststrtegy, K1=np.arange(0.0, 3, 0.5), K2=np.arange(0.0, 3, 0.5))

    cerebro.addanalyzer(CumulativeReturn, _name='cumulative_return')  # 累计收益率
    cerebro.addanalyzer(Turnover, _name='turnover')  # 换手率
    # bt自带的夏普比率分析器默认日度数据
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                        _name='mysharpe',
                        timeframe=bt.TimeFrame.Minutes,
                        compression=1,
                        # 期货交易时间:上午9点——下午3点， 晚上9点——晚上11点
                        factor=480 * 252,  # for 1-minute data,
                        annualize=True,
                        riskfreerate=0.00000207,
                        )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')  # 最大回撤
    cerebro.addsizer(TradeSizer)

    # 设置滑点
    # cerebro.broker.set_slippage_perc(0.02, 0.1)

    # 设置初始金额，手续费
    # 参数CustomCommScheme(commission, cost, rate, mult)
    # commission: 测试万3，万5，默认万5
    # cost：固定开仓费用， rate：比率费用
    # mult:合约乘数
    comminfo = CustomCommScheme(open_cost=3, close_cost=3, open_rate=0, close_rate=0, mult=5)
    cerebro.broker.addcommissioninfo(comminfo=comminfo)
    cerebro.broker.set_cash(200000.0)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    results = cerebro.run(printlog=True, maxcpus=None)
    end_time = time.time()  # 结束时间
    print(f"单次运行时间：{end_time - start_time} 秒")

    # 收集结果

    params = []
    for res in results:
        strat = res
        params.append(strat.results)

    df = pd.DataFrame(params, columns=['K1', 'K2', 'calma', 'cumulative_return', 'turnover',
                                       'sharpe_ratio', 'max_drawdown', 'trade_count', 'win_rate',
                                       'annual_return', 'annual_volatility', 'return_risk_ratio',
                                       'long_count', 'short_count', 'profit_loss_ratio'])
    df['sharpe_ratio'] = df['sharpe_ratio'][0]['sharperatio']