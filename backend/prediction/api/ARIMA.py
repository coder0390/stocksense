import math
from datetime import timedelta

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

from prediction.api.visualization import visualization


def arima(df, visualization_flag=False):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 选择需要的列，包含 'H' 和 'L'
    df = df[['H', 'L', 'C']]

    # 分离训练集和测试集
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    # 创建并训练ARIMA模型
    model = ARIMA(train['C'], order=(5, 1, 0))  # 这里的 (5, 1, 0) 是ARIMA模型的(p, d, q)参数，需要根据数据调整
    model_fit = model.fit()

    # 在测试集上进行预测
    history = list(train['C'])
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test['C'].iloc[t])

    # 计算误差
    mse = mean_squared_error(test['C'], predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(test['C'], predictions)
    print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

    # 预测未来一天的收盘价
    forecast = model_fit.forecast(steps=1)
    print("未来一天的预测收盘价：", forecast)

    # 获取最近一周的实际收盘价
    recent_week_data = df[-7:]

    # 计算未来一天的日期
    last_date = recent_week_data.index[-1]
    future_dates = [last_date + timedelta(days=1)]

    if visualization_flag:
        # 将实际收盘价和预测收盘价合并到一个DataFrame
        all_dates = list(recent_week_data.index) + future_dates
        all_closing_prices = list(recent_week_data['C']) + list(forecast)

        result_df = pd.DataFrame({
            '日期': all_dates,
            '收盘价': all_closing_prices
        })

        visualization('测试集上的预测值与实际值对比', '日期', '收盘价',
                      test.index, test['C'], '实际值',
                      test.index, predictions, '预测值')
        visualization('股价涨幅', '日期', '收盘价',
                      recent_week_data.index, recent_week_data['C'], '实际值',
                      future_dates, forecast, '预测值')

    return forecast[0], future_dates[0], mse, rmse, mae
