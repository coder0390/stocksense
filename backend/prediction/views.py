import json
import datetime

from django.http import JsonResponse
from django.shortcuts import render

from prediction.api.generate_prediction import compare_models
from utils.database import get_prediction_records, fetch_stock_data

# Create your views here.


database_path = "stocks421.db"


def predict_test():
    import pandas as pd

    # 创建 DataFrame
    data = {
        "CODE": ["999999"],
        "date": ["2024-04-20"],
        "model": ["gru"],
        "predicted_value": [3081.7626206332448],
        "mse": [804.6077339594585],
        "rmse": [28.36560829524829],
        "mae": [22.01668455661792]
    }

    df = pd.DataFrame(data)

    # 查看 DataFrame
    return df


def get_predict(request):
    data = json.loads(request.body)
    code = data.get('code')
    date = data.get('date')
    date = "2024-04-20"

    actual = fetch_stock_data(code, database_path, 7)

    dates = actual['date'].tolist()
    dates_datetime = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
    latest_date = dates_datetime[-1] + datetime.timedelta(days=1)
    dates_datetime.append(latest_date)
    dates = [d.strftime('%Y-%m-%d') for d in dates_datetime]

    # result = predict_test()
    result = get_prediction_records(database_path, code, date)
    if result.empty:
        compare_models(database_path, code)
        result = get_prediction_records(database_path, code, date)

    prices = actual['C'].tolist()
    prices.append(result['predicted_value'][0])

    if prices[-1] > prices[-2]:
        state = 'up'
    else:
        state = 'down'

    response_data = {
        'x': dates,
        'C': prices,
        "state": state
    }

    return JsonResponse(response_data, safe=False)
