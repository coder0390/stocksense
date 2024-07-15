import pandas as pd
import os
import importlib.util
from multiprocessing import Pool
import multiprocessing as mp
import tkinter as tk
from tqdm import tqdm

from utils.database import fetch_stock_data, get_all_stock_codes_list


def update_progress_bar(*a):
    pbar.update()


def run_strategies(stock_code, strategies_path, database_path):
    df = fetch_stock_data(stock_code, database_path)

    if df is None:
        return None

    result = pd.DataFrame()

    for filename in os.listdir(strategies_path):
        if filename.endswith(".py"):
            try:
                spec = importlib.util.spec_from_file_location(
                    name=filename[:-3],
                    location=os.path.join(strategies_path, filename)
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                strategy_result = mod.run_strategy(df).copy()

                if strategy_result is not None and not strategy_result.empty:
                    strategy_result["strategy"] = filename[:-3]
                    result = pd.concat([result, strategy_result], ignore_index=True)

            except Exception as e:
                print(f"An error occurred while running the strategy '{filename[:-3]}': {e}")

    if not result.empty:
        result = result.groupby(['date', 'CODE']).agg(
            {'strategy': lambda x: list(x), 'O': 'first', 'L': 'first', 'H': 'first', 'C': 'first',
             'VOL': 'first'}).reset_index()
        result['strategy_count'] = result['strategy'].apply(len)
    return result


def run_all_strategies(stock_codes, strategies_path, database_path):
    results = []
    all_results = pd.DataFrame()

    with Pool(processes=mp.cpu_count()) as pool:
        for stock_code in stock_codes:
            result = pool.apply_async(run_strategies, args=(stock_code, strategies_path, database_path), callback=update_progress_bar)
            results.append(result)
        pool.close()
        pool.join()

    for result in results:
        res = result.get()
        if res is not None:
            all_results = pd.concat([all_results, res], ignore_index=True)

    pbar.close()

    # 建立数据库
    # sorted_results = all_results.sort_values(['date', 'strategy_count'], ascending=[True, False])
    # print(sorted_results)
    print("insert to database")
    # insert_strategy_table(all_results)

    # 对所有数据先按日期降序排序，再按strategy_count降序排序
    # all_results.sort_values(by=['date', 'strategy_count'], ascending=[False, False], inplace=True)

    # current_date_folder = datetime.datetime.now().strftime('%Y%m%d')
    # current_date_file = datetime.datetime.now().strftime('%Y%m%d%H%M')

    # 创建 "每日选股" 文件夹和以当日日期命名的子文件夹
    # directory = os.path.join("每日选股", current_date_folder)
    # os.makedirs(directory, exist_ok=True)

    # 在子文件夹中保存文件
    # filename = os.path.join(directory, f'{current_date_file}_选股结果.csv')
    # all_results.to_csv(filename, index=False)

    # 打开包含生成文件的文件夹
    # os.system(f'explorer {directory}')


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    # strategies_path = filedialog.askdirectory()
    database_path = "../../stocks421.db"
    strategies_path = "strategy"

    if strategies_path:
        global pbar
        stock_codes = get_all_stock_codes_list(database_path)
        pbar = tqdm(total=len(stock_codes))
        run_all_strategies(stock_codes, strategies_path, database_path)
