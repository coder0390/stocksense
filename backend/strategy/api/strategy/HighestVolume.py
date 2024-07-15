import pandas as pd


def HighestVolume(data: pd.DataFrame) -> pd.DataFrame:
    """
    高成交量突破策略的Python实现，选股条件是：

    1. 股票达到过去30天内的最高收盘价。
    2. 当天的成交量是过去30天内的最高成交量。

    这种情况可能暗示股票基本面有积极的发展。
    """
    # 计算30天最高收盘价和最高成交量
    rolling_max_close = data['C'].rolling(window=30).max()
    rolling_max_volume = data['VOL'].rolling(window=30).max()

    # 当天收盘价和成交量达到30天内的最高点
    conditions = (data['C'] == rolling_max_close) & (data['VOL'] == rolling_max_volume)

    # 返回符合条件的股票
    selected_stocks = data[conditions]
    return selected_stocks


def run_strategy(data: pd.DataFrame):
    return HighestVolume(data)
