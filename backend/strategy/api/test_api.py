import pandas as pd
import numpy as np


path = "test_data/"


def get_all_stock_codes_test():
    data = pd.read_csv(path + "codes.csv", dtype={'CODE': str})
    data = data.sort_values("CODE")
    return data


# print("------------------------------------------------")
# 获取全部股票代码，用于前端的展示
# 前端无需传入，直接调用即可
# 后端返回dataframe，存放全部CODE
# codes = get_all_stock_codes()
# codes.to_csv('codes.csv', index=False)
# codes = get_all_stock_codes_test()
# print(codes)


def fetch_stock_data_test(code):
    data = pd.read_csv(path + '000001_stock.csv', dtype={'CODE': str})
    return data


# print("------------------------------------------------")
# 获取单支股票信息，前端点击某只股票时显示相应的信息
# 前端传入相应code
# 后端返回dataframe，包括对应code所有日期的OHLC（开盘价，最低价，最高价，收盘价），交易量，转手率共计6项数据
# 前端需要画图。交易量转手率可以直接使用折线图，OHLC可以考虑使用https://charts.ag-grid.com/vue/ohlc-series/
# code = "000001"
# info = fetch_stock_data(code)
# info.to_csv('000001_stock.csv', index=False)
# info = fetch_stock_data_test(code)
# print(info)


def select_by_date_in_strategy_test(date):
    data = pd.read_csv(path + 'strategies.csv', dtype={'CODE': str})
    return data


# print("------------------------------------------------")
# 获取某天股票的策略信息，前端通过选择日期后端进行查询
# 前端传入相应date
# 后端返回dataframe，包括对应日期，所有触发策略的股票，其CODE，触发的策略，触发策略的数量
# date = "2024-04-19"
# strategies = select_by_date_in_strategy(date)
# strategies.to_csv('strategies.csv', index=False)
# strategies = select_by_date_in_strategy_test(date)
# print(strategies)


def select_by_date_in_data_test(date, order):
    name = order + ".csv"
    data = pd.read_csv(path + name, dtype={'CODE': str})
    return data


# print("------------------------------------------------")
# 获取某天全部股票的交易信息，前端通过选择日期后端进行查询
# 前端传入日期和排序的方式
# 后端输出排序后的股票信息
# date = "2024-04-19"
# order = 'O'
# data_by_O = select_by_date_in_data(date, order)
# data_by_O.to_csv('O.csv', index=False)
# data_by_O = select_by_date_in_data_test(date, order)
# print(data_by_O)
# print("------------------------------------------------")
# order = 'H'
# data_by_H = select_by_date_in_data(date, order)
# data_by_H.to_csv('H.csv', index=False)
# data_by_H = select_by_date_in_data_test(date, order)
# print(data_by_H)
# print("------------------------------------------------")
# order = 'L'
# data_by_L = select_by_date_in_data(date, order)
# data_by_L.to_csv('L.csv', index=False)
# data_by_L = select_by_date_in_data_test(date, order)
# print(data_by_L)
# print("------------------------------------------------")
# order = 'C'
# data_by_C = select_by_date_in_data(date, order)
# data_by_C.to_csv('C.csv', index=False)
# data_by_C = select_by_date_in_data_test(date, order)
# print(data_by_C)
# print("------------------------------------------------")
# order = 'VOL'
# data_by_VOL = select_by_date_in_data(date, order)
# data_by_VOL.to_csv('VOL.csv', index=False)
# data_by_VOL = select_by_date_in_data_test(date, order)
# print(data_by_VOL)
# print("------------------------------------------------")
# order = 'turnover'
# data_by_turnover = select_by_date_in_data(date, order)
# data_by_turnover.to_csv('turnover.csv', index=False)
# data_by_turnover = select_by_date_in_data_test(date, order)
# print(data_by_turnover)


def fetch_and_process_stock_data_test(codes):
    n = len(codes)
    # 生成n个随机数,并归一化
    random_nums = np.random.rand(n)
    weights = random_nums / np.sum(random_nums)

    weights_dict = {}
    for i, code in enumerate(codes):
        weights_dict[code] = weights[i]

    return weights_dict, [0.1, 1.1, 2.1]

# print("------------------------------------------------")
# 根据选择的股票给出投资建议
# 前端传入股票列表
# 后端返回为每个股票分配的权重以及预期的收益
# codes = ["873833", "873806", "999999"]
# weights, performance = fetch_and_process_stock_data(codes)
# weights, performance = fetch_and_process_stock_data_test(codes)
# print(weights, performance)
