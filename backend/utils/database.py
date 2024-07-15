import math
import sqlite3
import pandas as pd
import json


local_database_path = "../stocks421.db"


def get_all_stock_codes(database_path):
    conn = sqlite3.connect(database_path)
    stock_codes = pd.read_sql_query("SELECT DISTINCT CODE FROM historical_data", conn)
    conn.close()
    stock_codes = stock_codes.sort_values('CODE')
    return stock_codes


def get_all_stock_codes_list(database_path):
    stock_codes = get_all_stock_codes(database_path)
    return stock_codes['CODE'].tolist()


# print(get_all_stock_codes_list(database_path))


# 获取单支股票信息
# Retrieve single stock information
def fetch_stock_data(stock_code, database_path, limit=275):
    # connect database, plz use absolute path
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query(f"SELECT * FROM historical_data WHERE CODE='{stock_code}' ORDER BY date DESC LIMIT {limit}",
                           conn)
    conn.close()

    if df.empty:
        return None

    df = df.sort_values('date')
    return df


# print(fetch_stock_data("000001", database_path))


def select_stocks_by_date_ordered(database_path, date, order, page_size, page_no):
    conn = sqlite3.connect(database_path)

    total_count = pd.read_sql_query(f"SELECT COUNT(*) FROM historical_data WHERE date = '{date}'", conn).iloc[0, 0]
    total_pages = math.ceil(total_count / page_size)

    # 计算偏移量
    offset = (page_no - 1) * page_size
    # 查询指定日期的数据,并按照指定顺序排序,分页返回
    query = f"SELECT * FROM historical_data WHERE date = '{date}' ORDER BY {order} DESC LIMIT {page_size} OFFSET {offset}"
    result = pd.read_sql_query(query, conn)

    # 关闭数据库连接
    conn.close()

    return result, total_pages


def new_strategy_table(database_path):
    # 连接 SQLite 数据库
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    # c.execute("DROP TABLE strategy")
    # 创建 s 表
    c.execute("""
        CREATE TABLE IF NOT EXISTS strategy (
            CODE VARCHAR(255) NOT NULL,
            date DATE NOT NULL,
            strategy TEXT,
            count INTEGER,
            PRIMARY KEY (CODE, date),
            FOREIGN KEY (CODE) REFERENCES stock(CODE)
        )
    """)

    conn.commit()
    c.close()
    conn.close()
    print("Done")


# new_strategy_table(database_path)


def insert_strategy_table(data, database_path):
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    for index, row in data.iterrows():
        code = row['CODE']
        date = row['date']
        strategy = json.dumps(row['strategy'])
        count = row['strategy_count']

        c.execute("INSERT INTO strategy (CODE, date, strategy, count) VALUES (?, ?, ?, ?)",
                  (code, date, strategy, count))

    conn.commit()
    conn.close()


def select_strategy_by_date(database_path, date, page_size, page_no):
    conn = sqlite3.connect(database_path)

    total_count = pd.read_sql_query(f"SELECT COUNT(*) FROM strategy WHERE date = '{date}'", conn).iloc[0, 0]
    total_pages = math.ceil(total_count / page_size)

    offset = (page_no - 1) * page_size
    query = f"SELECT * FROM strategy WHERE date = '{date}' ORDER BY count DESC LIMIT {page_size} OFFSET {offset}"
    result = pd.read_sql_query(query, conn)
    # 关闭数据库连接
    conn.close()

    return result, total_pages


def new_prediction_table(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # 创建 stock_predictions 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction (
            CODE VARCHAR(255) NOT NULL,
            date DATE NOT NULL,
            model VARCHAR(255),
            predicted_value FLOAT,
            mse FLOAT,
            rmse FLOAT,
            mae FLOAT,
            PRIMARY KEY (CODE, date),
            FOREIGN KEY (CODE) REFERENCES stock(CODE)
        )
    """)

    # 提交更改并关闭连接
    conn.commit()
    cursor.close()
    conn.close()
    print("Done")


# new_prediction_table(database_path)


def insert_stock_prediction(database_path, code, date, model, predicted_value, mse, rmse, mae):
    """
    插入股票预测数据到 stock_predictions 表

    参数:
    db_file (str): SQLite 数据库文件路径
    code (str): 股票代码
    date (str): 预测日期, 格式为 'YYYY-MM-DD'
    predicted_value (float): 预测值
    mse (float): 均方误差
    rmse (float): 均方根误差
    mae (float): 平均绝对误差
    model (str): 使用的预测模型
    """

    # 连接到 SQLite 数据库
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # 插入数据到 stock_predictions 表
    cursor.execute("""
        INSERT INTO prediction
        (code, date, model, predicted_value, mse, rmse, mae)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (code, date.strftime("%Y-%m-%d"), model, predicted_value, mse, rmse, mae))

    # 提交更改并关闭连接
    conn.commit()
    cursor.close()
    conn.close()


def get_prediction_records(database_path, code, date):
    """
    Retrieves the prediction records for the given stock code and date.

    Args:
        cursor (cursor): The database cursor object.
        code (str): The stock code.
        date (datetime.date): The date to search for.

    Returns:
        A pandas DataFrame containing the prediction records, or None if no records are found.
    """
    conn = sqlite3.connect(database_path)
    result = pd.read_sql_query(f"SELECT * FROM prediction WHERE CODE = '{code}' AND date = '{date}'", conn)
    conn.close()

    return result


def new_news_table(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # cursor.execute("DROP TABLE news")
    # 创建 stock_predictions 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url VARCHAR(255) NOT NULL,
            overview TEXT,
            title VARCHAR(255),
            writer VARCHAR(255),
            date DATETIME,
            content TEXT,
            predicted_value VARCHAR(255)
        )
    """)

    # 提交更改并关闭连接
    conn.commit()
    cursor.close()
    conn.close()
    print("Done")


# new_news_table(database_path)


def insert_news_data(database_path, url, overview, title, writer, date, content, predicted_value):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # 插入数据到 news 表
    cursor.execute("""
        INSERT INTO news (url, overview, title, writer, date, content, predicted_value)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (url, overview, title, writer, date, content, predicted_value))

    # 提交更改并关闭连接
    conn.commit()
    cursor.close()
    conn.close()

    # print("News data inserted successfully.")


def select_all_news(database_path, page_size, page_no):
    conn = sqlite3.connect(database_path)

    total_count_query = "SELECT COUNT(*) FROM news"
    total_count = pd.read_sql_query(total_count_query, conn).iloc[0, 0]
    total_pages = math.ceil(total_count / page_size)

    # 查询所有新闻记录
    offset = (page_no - 1) * page_size
    # 查询所有新闻记录并按照日期降序排序
    result = pd.read_sql_query(f"SELECT * FROM news ORDER BY date DESC LIMIT {page_size} OFFSET {offset}", conn)

    # 关闭数据库连接
    conn.close()

    return result, total_pages


def select_news_by_keyword(database_path, keyword, page_size, page_no):
    conn = sqlite3.connect(database_path)

    # 查询符合关键词的新闻总数
    total_count_query = "SELECT COUNT(*) FROM news WHERE overview LIKE '%{}%'".format(keyword)
    total_count = pd.read_sql_query(total_count_query, conn).iloc[0, 0]
    # 计算总页数
    total_pages = math.ceil(total_count / page_size)

    offset = (page_no - 1) * page_size
    # 使用 LIKE 操作符进行模糊查询
    sql = "SELECT * FROM news WHERE overview LIKE '%{}%' ORDER BY date DESC LIMIT {} OFFSET {}".format(
        keyword, page_size, offset)
    result = pd.read_sql_query(sql, conn)

    # 关闭数据库连接
    conn.close()

    return result, total_pages


def select_news_by_id(database_path, id):
    conn = sqlite3.connect(database_path)

    # 使用 LIKE 操作符进行模糊查询
    result = pd.read_sql_query(f"SELECT * FROM news WHERE id = '{id}'", conn)

    # 关闭数据库连接
    conn.close()

    return result
