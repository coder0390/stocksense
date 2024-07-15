import pandas as pd


def AA25(data: pd.DataFrame) -> pd.DataFrame:
    """
    AA30选股策略的Python实现版本，该策略的选股条件是：
    1. 计算当前股票收盘价与10天前收盘价的百分比变化，如果百分比变化大于30%，并且该变化量大于前一日的变化量，且前一日的变化量大于1%，则满足条件。
    """
    C = data['C']

    AA = (C - C.shift(10)) / C.shift(10) * 100

    conditions = (AA > 25) & (AA.shift(1) < AA) & (AA.shift(1) > 1)

    # 返回符合条件的股票
    selected_stocks = data[conditions]
    return selected_stocks


def run_strategy(data: pd.DataFrame):
    return AA25(data)
