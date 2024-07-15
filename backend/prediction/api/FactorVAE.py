import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers, models
import tensorflow as tf

from prediction.api.visualization import visualization, visualization_for_3


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstructed, z_mean, z_log_var = self(data)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(data, reconstructed))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def factorvae(df, visualization_flag):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    price_data = df['C'].values.reshape(-1, 1)  # 使用收盘价数据

    # 标准化数据
    scaler = MinMaxScaler()
    price_data_scaled = scaler.fit_transform(price_data)

    # 定义VAE模型
    latent_dim = 10  # 潜在空间维度

    # 编码器
    inputs = layers.Input(shape=(price_data_scaled.shape[1],))
    h = layers.Dense(128, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)

    z = Sampling()([z_mean, z_log_var])

    latent_dim = 10
    encoder_inputs = layers.Input(shape=(price_data_scaled.shape[1],))
    h = layers.Dense(128, activation='relu')(encoder_inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(latent_inputs)
    decoder_outputs = layers.Dense(price_data_scaled.shape[1], activation='sigmoid')(x)
    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')

    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')

    vae.fit(price_data_scaled, epochs=50, batch_size=32, shuffle=True)

    encoded_data = encoder.predict(price_data_scaled)[2]  # 获取z

    look_back = 60
    train_size = int(len(encoded_data) * 0.8)
    test_size = len(encoded_data) - train_size
    train, test = encoded_data[:train_size], encoded_data[train_size:]

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    model = models.Sequential()
    model.add(layers.LSTM(50, input_shape=(look_back, latent_dim)))
    model.add(layers.Dense(latent_dim))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = scaler.inverse_transform(decoder.predict(trainPredict))
    trainY = scaler.inverse_transform(decoder.predict(trainY))
    testPredict = scaler.inverse_transform(decoder.predict(testPredict))
    testY = scaler.inverse_transform(decoder.predict(testY))

    trainScore = mean_squared_error(trainY, trainPredict)
    testScore = mean_squared_error(testY, testPredict)
    trainRMSE = np.sqrt(trainScore)
    testRMSE = np.sqrt(testScore)
    trainMAE = mean_absolute_error(trainY, trainPredict)
    testMAE = mean_absolute_error(testY, testPredict)

    print(f'Train Score: {trainScore:.2f} MSE')
    print(f'Test Score: {testScore:.2f} MSE')
    print(f'Train RMSE: {trainRMSE:.2f}')
    print(f'Test RMSE: {testRMSE:.2f}')
    print(f'Train MAE: {trainMAE:.2f}')
    print(f'Test MAE: {testMAE:.2f}')

    # 预测后一天的值
    last_encoded_data = encoded_data[-look_back:]
    last_encoded_data = last_encoded_data.reshape((1, look_back, latent_dim))
    next_day_prediction = model.predict(last_encoded_data)
    next_day_prediction = scaler.inverse_transform(decoder.predict(next_day_prediction))

    print(f'Next Day Predicted Price: {next_day_prediction[0][0]:.2f}')

    if visualization_flag:
        last_date = df.index[-1]
        date_range = pd.date_range(last_date - pd.Timedelta(days=2), last_date + pd.Timedelta(days=3))

        # 创建实际股价和预测股价的数据集
        actual_prices = price_data[-6:]
        predicted_prices = np.concatenate([price_data[-3:], testPredict[:3]], axis=0)

        visualization(
            'Stock Price Prediction for Last 3 Days and Next 3 Days', 'Date', 'Price',
            date_range[:3], actual_prices[-3:], 'Actual Prices',
            date_range[2:], predicted_prices[2:], 'Predicted Prices',
            marker2='x')

        visualization_for_3(
            'Stock Price Prediction using VAE and LSTM', 'Time', 'Price',
            np.arange(len(price_data)), price_data, 'Actual Prices',
            np.arange(look_back, len(trainPredict) + look_back), trainPredict, 'Train Predict',
            np.arange(len(trainPredict) + (look_back * 2), len(price_data)), testPredict, 'Test Predict')

    return next_day_prediction[0][0], pd.to_datetime(df.index[-1] + pd.Timedelta(days=1)), testScore, testRMSE, testMAE
