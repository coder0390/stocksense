import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier

from utils.database import fetch_stock_data


def calculate_mean_historical_return(returns, frequency=252):
    """
    计算每只股票的年化历史平均回报率。

    参数:
    - returns: DataFrame，索引为日期，每列为一只股票的日回报率。
    - frequency: 年化因子，默认为252，假设一年有252个交易日。

    返回:
    - 年化历史平均回报率：Series，索引为股票代码，值为对应的年化平均回报率。
    """
    # 计算每只股票的日回报率平均值
    mean_daily_returns = returns.mean()

    # 将日回报率的平均值转换为年化回报率
    annualized_returns = (1 + mean_daily_returns) ** frequency - 1

    return annualized_returns


def custom_sample_cov(returns):
    """
    自定义计算样本协方差矩阵的函数。

    参数:
    - returns: DataFrame，索引为日期，每列为一只股票的日回报率。

    返回:
    - 协方差矩阵：DataFrame，索引和列都是股票代码，值为对应的股票回报率之间的协方差。
    """
    # 计算协方差矩阵
    cov_matrix = returns.cov()

    # 转换为年化协方差矩阵
    # 一年有252个交易日
    annualized_cov_matrix = cov_matrix * 252

    return annualized_cov_matrix


def calculate_daily_returns(df):
    df['returns'] = df['C'].pct_change()
    return df.dropna()


def calculate_beta(stock_df, market_df):

    # 对齐两个DataFrame以确保它们在相同的日期上
    stock_df, market_df = stock_df.align(market_df, join='inner', axis=0)

    # 计算日收益率
    stock_returns = stock_df['C'].pct_change().dropna()
    market_returns = market_df['C'].pct_change().dropna()

    # 计算收益率的协方差和市场收益率的方差
    covariance = np.cov(stock_returns, market_returns)[0][1]
    variance = np.var(market_returns)

    # 计算贝塔系数
    beta = covariance / variance
    return beta


def fetch_and_process_stock_data(stock_codes, database_path):
    """
    从给定的股票代码列表中获取股票数据,计算每日收益率,并将结果整合到一个 DataFrame 中。

    参数:
    stock_codes (list): 需要获取数据的股票代码列表

    返回:
    pandas.DataFrame: 包含所有股票的每日收益率的 DataFrame
    """
    betas = []

    returns = pd.DataFrame()

    market_code = '999999'  # 市场基准股票代码
    market_data = fetch_stock_data(market_code, database_path)
    market_data = calculate_daily_returns(market_data)

    for code in stock_codes:
        df = fetch_stock_data(code, database_path)
        stock_data = calculate_daily_returns(df)

        beta = calculate_beta(stock_data, market_data)
        betas.append(beta)

        df['daily_return'] = df['C'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        returns[code] = df.set_index('date')['daily_return']

    returns.dropna(inplace=True)

    # 再次检查确保没有NaN或无穷大值
    if returns.isnull().values.any() or np.isinf(returns.values).any():
        raise ValueError("回报率数据中存在NaN或无穷大值，请检查数据源。")

    mu = calculate_mean_historical_return(returns)
    S = custom_sample_cov(returns)
    # print(S)
    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w[0] + w[1] == 1)

    # 设置求解器
    ef.solver = 'ECOS'

    weights = ef.max_sharpe(risk_free_rate=-1)

    # print(weights)

    cleaned_weights = ef.clean_weights()
    print("优化后的投资组合权重：", cleaned_weights)
    portfolio_performance = ef.portfolio_performance(verbose=True, risk_free_rate=0.02)
    print(portfolio_performance)

    return cleaned_weights, portfolio_performance, betas


# codes = ["000001", "999999", "873833"]
# fetch_and_process_stock_data(codes)
