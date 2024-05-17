# -*- coding: utf-8 -*-

"""
@Author : fuyurong
@Create : 2023/8/21
@Package: instrument_config.py
@software: PyCharm

每个期货品种的参数以及文件路径， 筛选后的合约
"""

import pandas as pd
import datetime

"""PTA期货合约"""

TA = {
    'from_dt': datetime.datetime(2017, 5, 16),
    'to_dt': datetime.datetime(2023, 8, 17),
    'name': 'TA',
    'Multiple': 5,
    'PriceTick': 2.0,
    'openRate': 0.0,
    'openFee': 3.0,
    'closeRate': 0.0,
    'closeFee': 3.0,
    'csv_path': "D:\python_projects\pythonProject1\新建文件夹\\test\data\TA.csv",
    'cash': 200000,
}


"""铜期货合约"""
CU = {
    'from_dt': datetime.datetime(2017, 3, 16),
    'to_dt': datetime.datetime(2023, 8, 17),
    'name': 'TA',
    'Multiple': 5,
    'PriceTick': 10.0,
    'openRate': 5.00e-05,
    'openFee': 0.0,
    'closeRate': 5.00e-05,
    'closeFee': 0.0,
    'csv_path': "D:\python_projects\pythonProject1\新建文件夹\\test\data\CU.csv",
    'cash': 2000000
}

"""棕榈油期货合约"""
P = {
    'from_dt': datetime.datetime(2017, 5, 16),
    'to_dt': datetime.datetime(2023, 8, 17),
    'name': 'P',
    'Multiple': 10,
    'PriceTick': 2.0,
    'openRate': 0.0,
    'openFee': 2.75,
    'closeRate': 0.0,
    'closeFee': 2.75,
    'csv_path': "D:\python_projects\pythonProject1\新建文件夹\\test\data\P.csv",
    'cash': 200000
}

"""豆粕期货合约"""
M = {
    'from_dt': datetime.datetime(2017, 5, 16),
    'to_dt': datetime.datetime(2023, 8, 17),
    'name': 'M',
    'Multiple': 10,
    'PriceTick': 1.0,
    'openRate': 0.0,
    'openFee': 1.65,
    'closeRate': 0.0,
    'closeFee': 1.65,
    'csv_path': "D:\python_projects\pythonProject1\新建文件夹\\test\data\M.csv",
    'cash': 200000
}

"""铁矿石期货合约"""
I = {
    'from_dt': datetime.datetime(2018, 3, 26),
    'to_dt': datetime.datetime(2023, 8, 17),
    'name': 'I',
    'Multiple': 100,
    'PriceTick': 0.5,
    'openRate': 0.00033,
    'openFee': 0.0,
    'closeRate': 0.000132,
    'closeFee': 0.0,
    'csv_path': "D:\python_projects\pythonProject1\新建文件夹\\test\data\I.csv",
    'cash': 200000
}

"""螺纹钢期货合约"""
RB = {
    'from_dt': datetime.datetime(2017, 5, 16),
    'to_dt': datetime.datetime(2023, 8, 17),
    'name': 'RB',
    'Multiple': 10,
    'PriceTick': 1.0,
    'openRate': 1.00e-04,
    'openFee': 0.0,
    'closeRate': 1.00e-04,
    'closeFee': 0.0,
    'csv_path': "D:\python_projects\pythonProject1\新建文件夹\\test\data\RB.csv",
    'cash': 200000
}

"""原油期货合约"""
SC = {
    'from_dt': datetime.datetime(2018, 3, 24),
    'to_dt': datetime.datetime(2023, 8, 17),
    'name': 'SC',
    'Multiple': 1000,
    'PriceTick': 0.1,
    'openRate': 0.0,
    'openFee': 20.0,
    'closeRate': 0.0,
    'closeFee': 20.0,
    'csv_path': "D:\python_projects\pythonProject1\新建文件夹\\test\data\SC.csv",
    'cash': 200000000
}

# 将合约数据放入一个列表
contracts = [TA, CU, P, M, I, RB, SC]


