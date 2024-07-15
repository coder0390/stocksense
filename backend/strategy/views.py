import json

from django.http import JsonResponse

from utils.database import get_all_stock_codes, fetch_stock_data, select_stocks_by_date_ordered, \
    select_strategy_by_date
from strategy.api.weight import fetch_and_process_stock_data


database_path = "stocks421.db"


def hello_world(request):
    """
    简单的 Hello World 接口
    """
    data = {
        'message': 'Hello, World!'
    }
    return JsonResponse(data)


def stock_codes(request):
    """
    获取所有股票代码的接口
    """
    # stock_codes_ = get_all_stock_codes_test()
    stock_codes_ = get_all_stock_codes(database_path)
    return JsonResponse(stock_codes_.to_dict('list'), safe=False)


def stock_data(request):
    data = json.loads(request.body)
    code = data.get('code')

    # stock_data_ = fetch_stock_data_test(code)
    stock_data_ = fetch_stock_data(code, database_path, 30)

    dates = stock_data_['date'].tolist()
    # Extract other columns as a list of lists
    ohlc = stock_data_[['O', 'C', 'L', 'H']].values.tolist()
    VOL = stock_data_['VOL'].values.tolist()
    turnover = stock_data_['turnover'].values.tolist()

    response_data = {
        'x': dates,
        'ohlc': ohlc,
        'VOL': VOL,
        'turnover': turnover,
    }

    return JsonResponse(response_data, safe=False)


def stocks_by_date(request):
    data = json.loads(request.body)
    date = data.get('date')
    order = data.get('order')
    page_size = data.get('page_size')
    page_no = data.get('page_no')

    # data_by_order = select_by_date_in_data_test(date, order)
    data_by_order, total_pages = select_stocks_by_date_ordered(database_path, date, order, page_size, page_no)

    response_data = {
        'data': data_by_order.to_dict('records'),
        'total_pages': total_pages
    }

    return JsonResponse(response_data, safe=False)


def strategy_data(request):
    data = json.loads(request.body)
    date = data.get('date')
    page_size = data.get('page_size')
    page_no = data.get('page_no')

    # strategy_data_ = select_by_date_in_strategy_test(date)
    strategy_data_, total_pages = select_strategy_by_date(database_path, date, page_size, page_no)
    response_data = {
        'data': strategy_data_.to_dict('records'),
        'total_pages': total_pages
    }

    return JsonResponse(response_data, safe=False)


def generate_weights(request):
    data = json.loads(request.body)
    codes = data.get("codes")

    # weights, performance = fetch_and_process_stock_data_test(codes)
    weights, performance, beta = fetch_and_process_stock_data(codes, database_path)

    response_data = {
        "weights": weights,
        "performance": performance,
        "beta": beta
    }

    return JsonResponse(response_data)
