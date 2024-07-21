import pandas as pd

# CSVファイルの読み込み
data = pd.read_csv('../MalbehavD-V1-main/MalBehavD-V1-dataset.csv')

# API呼出し部分のデータを抽出
api_calls = data.iloc[:, 2:]

# 全てのAPI呼出しを一つのリストにまとめる
all_api_calls = api_calls.values.flatten()

# ユニークなAPI呼出しを取得
unique_api_calls = pd.unique(all_api_calls)

# ユニークなAPI呼出しの数を表示
num_unique_api_calls = len(unique_api_calls)
print(f'ユニークなAPI呼出しの数: {num_unique_api_calls}')
