import pandas as pd


def run_basic_fundamental_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    价格行为基本面推断策略的实现，选股条件是：
    1. 股票的收盘价在过去30天内持续上升。
    2. 成交量在上升日子里持续增长，且在价格上升的最后一天达到30天内的高点。
    这可能表明市场对股票的基本面持正面看法。
    """
    # 检查收盘价是否持续上升
    increasing_days = data['C'].rolling(window=30).apply(lambda x: pd.Series(x).is_monotonic_increasing, raw=True)

    # 成交量在价格上升的最后一天是否为30天内最高
    volume_peak = data['VOL'] == data['VOL'].rolling(window=30).max()

    # 合并条件，确保increasing_days的数据类型是bool，用于条件判断
    conditions = (increasing_days.astype(bool)) & (volume_peak)

    # 返回符合条件的股票
    selected_stocks = data[conditions]
    return selected_stocks


def run_strategy(data: pd.DataFrame):
    return run_basic_fundamental_strategy(data)
