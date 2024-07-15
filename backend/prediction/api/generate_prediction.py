from copy import deepcopy

from prediction.api.ARIMA import arima
from prediction.api.EMD_IMFs import emd_imfs
from prediction.api.EMD_IMFs2 import emd_imfs2
from prediction.api.FactorVAE import factorvae
from prediction.api.TF_MODEL import tf_model
from utils.database import fetch_stock_data, insert_stock_prediction


def compare_models(database_path, code):
    """
    比较不同模型在给定数据上的预测性能。

    参数:
    database_path (str): 数据库文件路径
    code (str): 股票代码
    num_data (int): 获取的数据量

    返回:
    dict: 包含各模型预测结果的字典
    """

    # 获取股票数据
    data = fetch_stock_data(code, database_path, 275)

    # 初始化结果字典
    results = {}

    # ARIMA
    df_deep = deepcopy(data)
    pre_a, time_a, mse_a, rmse_a, mae_a = arima(df_deep, False)
    results['arima'] = (pre_a, time_a, mse_a, rmse_a, mae_a)

    data = fetch_stock_data(code, database_path, 2750)

    # EMD-IMFS
    # df_deep = deepcopy(data)
    # pre_ei, time_ei, mse_ei, rmse_ei, mae_ei = emd_imfs(df_deep, False)
    # results['emd_imfs'] = (pre_ei, time_ei, mse_ei, rmse_ei, mae_ei)

    # BiLSTM
    # df_deep = deepcopy(data)
    # pre_b, time_b, mse_b, rmse_b, mae_b = tf_model(df_deep, False, "BILSTM")
    # results['bilstm'] = (pre_b, time_b, mse_b, rmse_b, mae_b)

    # EMD-IMFS2
    # df_deep = deepcopy(data)
    # pre_ei2, time_ei2, mse_ei2, rmse_ei2, mae_ei2 = emd_imfs2(df_deep, False)
    # results['emd_imfs2'] = (pre_ei2, time_ei2, mse_ei2, rmse_ei2, mae_ei2)

    # FactorVAE
    # df_deep = deepcopy(data)
    # pre_f, time_f, mse_f, rmse_f, mae_f = factorvae(df_deep, False)
    # results['factorvae'] = (pre_f, time_f, mse_f, rmse_f, mae_f)

    # GRU
    df_deep = deepcopy(data)
    pre_g, time_g, mse_g, rmse_g, mae_g = tf_model(df_deep, False, 'GRU')
    results['gru'] = (pre_g, time_g, mse_g, rmse_g, mae_g)

    # LSTM
    # df_deep = deepcopy(data)
    # pre_l, time_l, mse_l, rmse_l, mae_l = tf_model(df_deep, False, 'LSTM')
    # results['lstm'] = (pre_l, time_l, mse_l, rmse_l, mae_l)

    # LSTM-CNN
    # df_deep = deepcopy(data)
    # pre_c, time_c, mse_c, rmse_c, mae_c = tf_model(df_deep, False, 'CNN')
    # results['lstm_cnn'] = (pre_c, time_c, mse_c, rmse_c, mae_c)

    # 找到 MSE 最小的结果
    min_mse_model = min(results, key=lambda x: results[x][2])
    min_mse_result = results[min_mse_model]

    results['best_model'] = min_mse_model
    results['best_result'] = min_mse_result

    # print(min_mse_model)
    # print(min_mse_result)

    insert_stock_prediction(database_path=database_path, code=code, date=results['best_result'][1],
                            model=min_mse_model, predicted_value=results['best_result'][0],
                            mse=results['best_result'][2], rmse=results['best_result'][3], mae=results['best_result'][4])

    print(min_mse_model)
    print("Done")

    return results


# test_path = "../../stocks421.db"
# code = "000001"
# print(compare_models(test_path, code))
