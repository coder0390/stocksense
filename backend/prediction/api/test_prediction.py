from prediction.api.ARIMA import arima
from prediction.api.TF_MODEL import tf_model
from prediction.api.EMD_IMFs import emd_imfs
from prediction.api.EMD_IMFs2 import emd_imfs2
from prediction.api.FactorVAE import factorvae
from utils.database import fetch_stock_data
from copy import deepcopy


database_path = "../../stocks421.db"
code = "999999"
data = fetch_stock_data(code, database_path, 2750)
print("------------emd_imfs------------")
df_deep = deepcopy(data)
pre_ei, time_ei, mse_ei, rmse_ei, mae_ei = emd_imfs(df_deep, True)
print(pre_ei, time_ei, mse_ei, rmse_ei, mae_ei)
print("------------arima------------")
df_deep = deepcopy(data)
pre_a, time_a, mse_a, rmse_a, mae_a = arima(df_deep, True)
print(pre_a, time_a, mse_a, rmse_a, mae_a)
print("------------bilstm------------")
df_deep = deepcopy(data)
pre_b, time_b, mse_b, rmse_b, mae_b = tf_model(df_deep, True, "BILSTM")
print(pre_b, time_b, mse_b, rmse_b, mae_b)
print("------------emd_imfs2------------")
df_deep = deepcopy(data)
pre_ei2, time_ei2, mse_ei2, rmse_ei2, mae_ei2 = emd_imfs2(df_deep, True)
print(pre_ei2, time_ei2, mse_ei2, rmse_ei2, mae_ei2)
print("------------factorvae------------")
df_deep = deepcopy(data)
pre_f, time_f, mse_f, rmse_f, mae_f = factorvae(df_deep, True)
print(pre_f, time_f, mse_f, rmse_f, mae_f)
print("------------gru------------")
df_deep = deepcopy(data)
pre_g, time_g, mse_g, rmse_g, mae_g = tf_model(df_deep, True, 'GRU')
print(pre_g, time_g, mse_g, rmse_g, mae_g)
print("------------lstm------------")
df_deep = deepcopy(data)
pre_l, time_l, mse_l, rmse_l, mae_l = tf_model(df_deep, True, 'LSTM')
print(pre_l, time_l, mse_l, rmse_l, mae_l)
print("------------lstm_cnn------------")
df_deep = deepcopy(data)
pre_c, time_c, mse_c, rmse_c, mae_c = tf_model(df_deep, True, 'CNN')
print(pre_c, time_c, mse_c, rmse_c, mae_c)
