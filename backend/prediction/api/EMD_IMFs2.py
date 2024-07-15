import numpy as np
import pandas as pd
from PyEMD import EMD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from skfuzzy import control as ctrl
import skfuzzy as fuzz

from prediction.api.visualization import visualization


def emd_imfs2(df, visualization_flag):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    price_data = df['C'].values.reshape(-1, 1)  # 使用收盘价数据

    # 计算移动平均线和相对强弱指数
    df['SMA'] = df['C'].rolling(window=5).mean()
    df['RSI'] = 100 - (100 / (1 + df['C'].diff(1).apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                              df['C'].diff(1).apply(lambda x: abs(min(x, 0))).rolling(window=14).mean()))

    # 标准化数据
    scaler = MinMaxScaler()
    price_data_scaled = scaler.fit_transform(price_data.reshape(-1, 1)).flatten()

    # 使用EMD进行分解
    emd = EMD()
    imfs = emd(price_data_scaled)

    # 创建输入和输出变量
    input1 = ctrl.Antecedent(np.arange(0, 1, 0.01), 'input1')
    input2 = ctrl.Antecedent(np.arange(0, 1, 0.01), 'input2')
    output = ctrl.Consequent(np.arange(0, 1, 0.01), 'output')

    # 定义隶属函数
    input1['low'] = fuzz.trimf(input1.universe, [0, 0, 0.5])
    input1['medium'] = fuzz.trimf(input1.universe, [0, 0.5, 1])
    input1['high'] = fuzz.trimf(input1.universe, [0.5, 1, 1])

    input2['low'] = fuzz.trimf(input2.universe, [0, 0, 0.5])
    input2['medium'] = fuzz.trimf(input2.universe, [0, 0.5, 1])
    input2['high'] = fuzz.trimf(input2.universe, [0.5, 1, 1])

    output['low'] = fuzz.trimf(output.universe, [0, 0, 0.5])
    output['medium'] = fuzz.trimf(output.universe, [0, 0.5, 1])
    output['high'] = fuzz.trimf(output.universe, [0.5, 1, 1])

    # 定义更多的模糊规则
    rule1 = ctrl.Rule(input1['low'] & input2['low'], output['low'])
    rule2 = ctrl.Rule(input1['low'] & input2['medium'], output['low'])
    rule3 = ctrl.Rule(input1['low'] & input2['high'], output['medium'])
    rule4 = ctrl.Rule(input1['medium'] & input2['low'], output['low'])
    rule5 = ctrl.Rule(input1['medium'] & input2['medium'], output['medium'])
    rule6 = ctrl.Rule(input1['medium'] & input2['high'], output['high'])
    rule7 = ctrl.Rule(input1['high'] & input2['low'], output['medium'])
    rule8 = ctrl.Rule(input1['high'] & input2['medium'], output['high'])
    rule9 = ctrl.Rule(input1['high'] & input2['high'], output['high'])

    # 创建控制系统
    stock_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    stock_sim = ctrl.ControlSystemSimulation(stock_ctrl)

    # 确保所有IMFs的长度与原始信号相同
    min_length = min(len(imf) for imf in imfs)
    imfs = [imf[:min_length] for imf in imfs]
    price_data_scaled = price_data_scaled[:min_length]

    # 使用IMFs进行预测
    predicted_prices_scaled = []
    for i in range(len(imfs[0])):
        stock_sim.input['input1'] = imfs[0][i]
        stock_sim.input['input2'] = imfs[1][i]
        stock_sim.compute()
        predicted_prices_scaled.append(stock_sim.output['output'])

    # 反标准化预测值
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1)).flatten()

    # 计算均方误差（MSE）、均方根误差（RMSE）和平均绝对误差（MAE）
    mse = mean_squared_error(price_data[:min_length], predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(price_data[:min_length], predicted_prices)

    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')

    # 输出后一天的预测值
    next_day_prediction = predicted_prices[-1]
    print(f'Next Day Predicted Price: {next_day_prediction}')

    if visualization_flag:
        # 获取最后三天的实际日期
        last_date = df.index[-1]
        date_range = pd.date_range(last_date - pd.Timedelta(days=2), last_date + pd.Timedelta(days=3))

        # 创建实际股价和预测股价的数据集
        actual_prices = price_data[-6:].flatten()  # 将实际股价展平为一维数组
        predicted_prices = np.concatenate([price_data[-3:].flatten(), predicted_prices[:3]], axis=0)
        visualization(
            'Stock Price Prediction for Last 3 Days and Next 3 Days', 'Date', 'Price',
            date_range[:3], actual_prices[-3:], 'Actual Prices',
            date_range[2:], predicted_prices[2:], 'Predicted Prices',
            marker2='x')

    return next_day_prediction, pd.to_datetime(df.index[-1] + pd.Timedelta(days=1)), mse, rmse, mae
