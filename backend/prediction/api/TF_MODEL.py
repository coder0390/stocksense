from datetime import timedelta

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, GRU, Bidirectional, Dense

from prediction.api.visualization import visualization


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 3])  # 3代表收盘价列
    return np.array(X), np.array(Y)


def evaluate_model(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse, rmse, mae


def predict_future(look_back, data, model, scaler, days=1):
    predictions = []
    last_sequence = data[-look_back:]  # 最后一个输入序列
    for _ in range(days):
        # 预测下一个值
        prediction = model.predict(last_sequence[np.newaxis, :, :])[0][0]
        # 反归一化预测的收盘价
        prediction_unscaled = scaler.inverse_transform([[0, 0, 0, prediction, 0, 0]])[0][3]
        predictions.append(prediction_unscaled)
        # 更新输入序列
        new_sequence = np.append(last_sequence[1:], [[0, 0, 0, prediction, 0, 0]], axis=0)
        last_sequence = np.append(last_sequence, [[0, prediction, 0, 0, 0, 0]], axis=0)[1:]
    return predictions


def tf_model(df, visualization_flag, model_name):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 选择需要的列，包含 'H' 和 'L'
    df = df[['O', 'H', 'L', 'C', 'VOL', 'turnover']]

    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    # 使用过去60天的数据来预测未来的收盘价
    look_back = 60
    X, Y = create_dataset(scaled_data, look_back)

    # 检查数据形状
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # 将数据划分为训练集和测试集
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    # 检查训练和测试数据的形状
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_test shape:", Y_test.shape)

    model = Sequential()
    if model_name == "BILSTM":
    # 构建BiLSTM模型
        model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(look_back, X_train.shape[2])))
        model.add(Bidirectional(LSTM(50)))
    elif model_name == "GRU":
        model.add(GRU(50, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
        model.add(GRU(50))
    elif model_name == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, X.shape[2])))
        model.add(LSTM(50))
    elif model_name == "CNN":
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))

    recent_data = scaled_data[-look_back:]

    # 预测未来一天
    future_predictions = predict_future(look_back, recent_data, model, scaler, days=1)

    print("未来一天的预测收盘价：", future_predictions)

    # 获取最近一周的实际收盘价
    recent_week_data = df[-7:]

    # 计算未来一天的日期
    last_date = recent_week_data.index[-1]
    future_dates = [last_date + timedelta(days=1)]

    # 将实际收盘价和预测收盘价合并到一个DataFrame
    all_dates = list(recent_week_data.index) + future_dates
    all_closing_prices = list(recent_week_data['C']) + list(future_predictions)

    result_df = pd.DataFrame({
        '日期': all_dates,
        '收盘价': all_closing_prices
    })

    # 使用测试集进行预测
    Y_test_predictions = model.predict(X_test)

    # 反归一化预测值
    Y_test_predictions_rescaled = scaler.inverse_transform(
        [[0, 0, 0, pred, 0, 0] for pred in Y_test_predictions.flatten()])[:, 3]
    Y_test_rescaled = scaler.inverse_transform([[0, 0, 0, true_val, 0, 0] for true_val in Y_test])[:, 3]

    # 评估模型
    mse, rmse, mae = evaluate_model(Y_test_rescaled, Y_test_predictions_rescaled)

    print(f"模型评估 - MSE: {mse}, RMSE: {rmse}, MAE: {mae}")

    if visualization_flag:
        visualization('STOCK PRICE FOR ' + model_name, '日期', '收盘价',
                      recent_week_data.index, recent_week_data['C'], '实际值',
                      future_dates, future_predictions, '预测值')
        n = len(Y_test_rescaled)
        x = np.arange(n)
        visualization('CONTRAST FOR ' + model_name, '时间步长', '收盘价',
                      x, Y_test_rescaled, '实际值',
                      x, Y_test_predictions_rescaled, '预测值')

    return future_predictions[0], future_dates[0], mse, rmse, mae
