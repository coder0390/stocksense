import pandas as pd
import talib


def ASCEND(data: pd.DataFrame) -> pd.DataFrame:
    """
    MUWRG Stock Selection Strategy implemented using Python and TA-Lib.
    """
    # Ensure all data columns are in the correct float format
    for col in ['O', 'H', 'L', 'C', 'VOL']:
        data[col] = data[col].astype(float)

    C = data['C']  # Close price
    H = data['H']  # High price
    L = data['L']  # Low price
    O = data['O']  # Open price
    VOL = data['VOL']  # Volume

    # Moving averages of the close price
    A1 = talib.MA(C, timeperiod=5)
    A2 = talib.MA(C, timeperiod=10)
    A3 = talib.MA(C, timeperiod=20)
    A4 = talib.MA(C, timeperiod=60)
    A5 = talib.MA(C, timeperiod=245)

    # Moving averages of the volume
    B1 = talib.MA(VOL, timeperiod=10)
    B2 = talib.MA(VOL, timeperiod=60)

    # MACD calculation using talib
    DIF, DEA, MACD = talib.MACD(C, fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = MACD

    # High and low functions
    highest_high_21 = talib.MAX(H, timeperiod=21)
    lowest_low_21 = talib.MIN(L, timeperiod=21)
    highest_high_42 = talib.MAX(H, timeperiod=42)
    lowest_low_42 = talib.MIN(L, timeperiod=42)

    # Calculating the "快" and "慢" indicators
    fast = 100 * (highest_high_21 - C) / (highest_high_21 - lowest_low_21)
    slow = 100 * (highest_high_42 - C) / (highest_high_42 - lowest_low_42)

    # Defining the conditions for the strategy
    conditions = (
            (C > O) &
            (C > data['C'].shift(1)) &
            (C > A2) &
            (MACD > data['MACD'].shift(1)) &
            (MACD > 0) &
            (data['MACD'].shift(1) < data['MACD'].shift(2)) &
            (data['MACD'].shift(2) < data['MACD'].shift(3)) &
            (fast < 30) & (slow < 30) & (fast < slow)
    )

    # Selecting stocks that meet the criteria
    selected_stocks = data[conditions]
    return selected_stocks


def run_strategy(data: pd.DataFrame):
    return ASCEND(data)
