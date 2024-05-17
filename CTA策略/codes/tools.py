# -*- coding: utf-8 -*-

"""
@Author : fuyurong
@Create : 2023/8/22
@Package: tools.py
@software: PyCharm


"""

import backtrader as bt
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体为微软雅黑，你也可以选择其他的中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


"""卡玛比率分析器"""
class KamaRatio(bt.Analyzer):

    def start(self):
        self.start_cash = self.strategy.broker.get_cash()

    def stop(self):
        rets = np.array(self.strategy._returns)
        self.rets = rets
        self.kama_ratio = np.mean(rets) / np.std(rets)

    def get_analysis(self):
        return {'kama_ratio': self.kama_ratio}


"""累计收益率分析器"""
class CumulativeReturn(bt.Analyzer):
    def __init__(self):
        self.values = []  # 保存每天的组合价值
        self.daily_returns = []  # 日度收益率
        self.cumulative_returns = []  # 累计收益率
        self.dates = []  # 保存每天的日期
        self.prev_date = None

    def next(self):

        current_date = self.strategy.datetime.date(0)
        self.dates.append(current_date)
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
            "values": self.values,
            'dates': self.dates
        }




"""定价交易费率"""
class CustomCommScheme(bt.CommInfoBase):

    # 默认参数

    params = (
        ('commission', 0.0003),         # 默认百分比手续费
        ('open_cost', 0),               # 固定费用
        ('open_rate', 1.00e-04),        # 比率费用
        ('close_cost', 0),
        ('close_rate', 1.00e-04),
        ('mult', 10),                   # 乘数
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

"""定义交易规模"""
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


"""主力合约筛选合约"""
def calculate_main_contract(df):

    df['date'] = pd.to_datetime(df['date'])
    df['date1'] = df['date'].dt.date

    # 计算每天每个合约的 open_interest
    idx = pd.DataFrame(df.groupby(['date1'])['open_interest'].idxmax()).dropna()['open_interest'].astype(int)
    main_contract = df.loc[idx]

    # 根据规则3，确保连续5天的持仓量最大的合约保持相同，否则主力合约不发生改变
    main_contract['instrument_shift'] = main_contract['instrument'].shift()
    main_contract['instrument_same'] = main_contract['instrument'] == main_contract['instrument_shift']
    main_contract['instrument_same'] = main_contract.groupby(
        (main_contract['instrument_same'] == False).cumsum()).cumcount()
    #     main_contract = main_contract[main_contract['instrument_same'] < 5]

    list_same = list(main_contract['instrument_same'])
    list_ = list(main_contract['instrument'])

    df_temp = pd.DataFrame([list_, list_same]).T
    df_temp['adjust'] = np.where(df_temp[1] >= 5, df_temp[0], df_temp[0].shift(1))
    df_temp['adjust'][0] = 0
    list_new = list(df_temp['adjust'])
    list_df = list(df.groupby(['date1']))
    return_df = pd.DataFrame()
    #     print(list_df[-1][1][list_df[-1][1]['instrument']==list_[-1]])
    for i in range(len(list_)):
        #         print(list_df[i][1][list_df[i][1]['instrument']==list_[i]])
        temp = list_df[i][1]

        temp2 = temp[temp['instrument'] == list_new[i]]
        return_df = pd.concat([return_df, temp2])

    return return_df


def find_switch_dates(main_contract):
    """
    返回合约切换的日期列表。
    """
    switch_dates = main_contract[main_contract['instrument'] != main_contract['instrument'].shift()]['date'].tolist()
    return switch_dates

def get_main_cont(df):
    """
    在合约切换日对主力合约价格平滑化
    """
    main_contract = calculate_main_contract(df)
    sw_date = find_switch_dates(main_contract)
    for date in sw_date:
        main_contract['close_after'] = np.where(main_contract['date'] == date, main_contract['close'] - (main_contract['close'].shift(1) - main_contract['open']), main_contract['close'])

    return main_contract


def get_date_list(data):
    dates_list = eval(data['日期'][0])
    unique_dates = []
    for date in dates_list:
        if date not in unique_dates:
            unique_dates.append(date)
    return unique_dates



"""可视化策略输出结果"""

def concatenate_dfs(*dataframes, column_name='累计收益率', date_list):
    """
    :param dataframes:可输入多个策略的dataframe
    :param column_name:对应列名
    :param date_list: 对应交易日期
    :return: 净值曲线
    """
    frames = []

    # 遍历提供的所有DataFrames
    for df in dataframes:
        # 从DataFrame中提取并转换列表
        list_ = ast.literal_eval(df[column_name][0])

        # 将列表转换为DataFrame并添加到frames列表
        frames.append(pd.DataFrame(list_, columns=[column_name]))
    frames.append(pd.DataFrame(date_list, columns=['日期']))

    # 使用pd.concat连接所有的DataFrame

    result_df = pd.concat(frames, axis=1, ignore_index=True)

    # 对不同的回测结果输入不同的columns
    # result_df.columns = ['菲阿里四价策略(未优化)', 'ATR止盈止损(经验参数)', '菲阿里四价策略(限制时间间隔_5min)', '菲阿里四价策略(限制时间间隔_优化)', '日期']

    result_df.index = result_df['日期']
    result_df.drop(['日期'], axis=1, inplace=True)

    return result_df


# 可视化
def get_plots(data, name=None):
    plt.figure(figsize=(9, 4))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)

    plt.title("{}回测结果".format(name))
    plt.xlabel("日期")
    plt.ylabel("累计收益")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('{}.pdf'.format(name))
    plt.show()

"""可视化热力图"""
def get_h_plot(df):
    for metric in ['最大回撤', '夏普比率', '年化收益']:
        heatmap_data = df.pivot('k1', 'k2', metric)
        heatmap_data = heatmap_data.iloc[::-1]
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=False, fmt=".2f")
        plt.title(f'{metric} 不同的 K1 和 K2 参数')
        plt.savefig('{}.pdf'.format(metric))
        plt.show()

