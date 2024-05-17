# -*- coding: utf-8 -*-

"""
@Author : fuyurong
@Create : 2023/8/24
@Package: main.py
@software: PyCharm

回测主函数
"""

import backtrader as bt
import pandas as pd
from instrument_config import contracts
from tools import CustomCommScheme, CumulativeReturn, TradeSizer, get_plots, get_h_plot, get_main_cont, concatenate_dfs
from strategy_1 import R_BreakerStrategy
from strategy_2 import ATRStrategy
from strategy_3 import SIJIAStrategy



def run_backtest(data_path, from_date, to_date, open_cost, close_cost, open_rate, close_rate, mult, cash, Strategy):

    cerebro = bt.Cerebro()

    # 加载数据...
    data = bt.feeds.GenericCSVData(
        dataname=data_path,
        fromdate=from_date,
        todate=to_date,
        datetime=11,
        open=0,
        high=2,
        low=3,
        close=1,
        volume=4,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        timeframe=bt.TimeFrame.Minutes,
    )
    cerebro.adddata(data)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=30)


    comminfo = CustomCommScheme(open_cost=open_cost,
                                close_cost=close_cost,
                                open_rate=open_rate,
                                close_rate=close_rate,
                                mult=mult)

    cerebro.broker.addcommissioninfo(comminfo=comminfo)

    cerebro.addsizer(TradeSizer)
    cerebro.addanalyzer(CumulativeReturn, _name='cumulative_return')

    cerebro.addanalyzer(bt.analyzers.SharpeRatio,
                        _name='mysharpe',
                        timeframe=bt.TimeFrame.Minutes,
                        compression=1,
                        factor=480 * 252,
                        annualize=True,
                        riskfreerate=0.00000207,
                        )

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    cerebro.broker.set_cash(cash)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.addstrategy(Strategy)
    results = cerebro.run(printlog=True, maxcpus=None, use_stop=True)

    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    params = []
    for res in results:
        strat = res
        params.append(strat.results)

    df = pd.DataFrame(params, columns=['累计收益率', '换手率', '夏普比率',
                                       '最大回撤', '交易总次数', '总胜率', '总收益',
                                       '做多次数', '做空次数', '盈亏比', '做多胜率', '做空胜率', '日期'])
    df['夏普比率'] = df['夏普比率'][0]['sharperatio']

    return df




# 对每个合约进行回测
# 以SIJIAStrategy为例
for contract in contracts:

    # mian_cons = get_main_cont(contract['csv_path'])

    results_df = run_backtest(
        data_path=contract['csv_path'],
        from_date=contract['from_dt'],
        to_date=contract['to_dt'],
        open_cost=contract['openFee'],
        close_cost=contract['closeFee'],
        open_rate=contract['openRate'],
        close_rate=contract['closeRate'],
        mult=contract['Multiple'],
        cash=contract['cash'],
        Strategy=SIJIAStrategy,
    )
    results_df.to_csv('{}.csv'.format(contract))
    print(f"Results for {contract['name']}")
    print(results_df)
    print("---------------------------------------------------")


# get_plots(concatenate_dfs(results_df))