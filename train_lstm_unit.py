import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# データの読み込みと前処理
data = pd.read_csv('../MalbehavD-V1-main/MalBehavD-V1-dataset.csv')
labels = data.iloc[:, 1].values
api_calls = data.iloc[:, 2:].astype(str).values

# API呼出しのトークナイズとパディング
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
api_calls_tokenized = [le.fit_transform(call) for call in api_calls]
padded_api_calls = pad_sequences(api_calls_tokenized, maxlen=150, padding='post')

# ラベルのone-hotエンコーディング
labels = to_categorical(labels)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(padded_api_calls, labels, test_size=0.2, random_state=42)

# ユニット数のリスト
unit_sizes = [50, 100, 150, 200]

# 結果を保存するリスト
results = []

for units in unit_sizes:
    print(f"Training model with LSTM units: {units}")
    
    # モデルの構築
    model = Sequential()
    model.add(Embedding(input_dim=len(le.classes_), output_dim=128, input_length=150))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=units, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))

    # モデルのコンパイル
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 早期停止を設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # モデルのトレーニング
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

    # モデルの評価
    loss, accuracy = model.evaluate(X_test, y_test)
    results.append((units, accuracy))

    print(f"Test Accuracy with LSTM units {units}: {accuracy}")

# 結果の表示
for units, accuracy in results:
    print(f"LSTM Units: {units}, Test Accuracy: {accuracy}")
