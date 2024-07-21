import pandas as pd

# データの読み込み
data = pd.read_csv('C:/Users/hyu2_/OneDrive/デスクトップ/cnnlstm/MalbehavD-V1-main/MalBehavD-V1-dataset.csv')

# 全サンプル数の表示
n_samples = data.shape[0]
print(f'Total number of samples: {n_samples}')

# テスト用とトレーニング用のデータ数の計算
n_test = int(n_samples * 0.2)
n_train = n_samples - n_test

print(f'Number of training samples: {n_train}')
print(f'Number of testing samples: {n_test}')
