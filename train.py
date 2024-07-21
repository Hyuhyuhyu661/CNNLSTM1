import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# データの読み込み
data = pd.read_csv('C:/Users/hyu2_/OneDrive/デスクトップ/cnnlstm/MalbehavD-V1-main/MalBehavD-V1-dataset.csv')

# ラベルとAPI呼出しの抽出
labels = data.iloc[:, 1].values
api_calls = data.iloc[:, 2:].values

# API呼出しの数値化（文字列を整数インデックスにマッピングする）
le = LabelEncoder()
# すべてのAPI呼出しをフラットリストに変換してLabelEncoderでフィット
le.fit([api for sublist in api_calls for api in sublist if pd.notna(api)])
# それぞれのサンプルに対して変換を適用
integer_encoded_calls = [le.transform(sublist[pd.notna(sublist)]).tolist() for sublist in api_calls]

# パディング
padded_api_calls = pad_sequences(integer_encoded_calls, maxlen=151, padding='post')

# ラベルのone-hotエンコーディング
labels = to_categorical(labels)

# データをトレーニング用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(padded_api_calls, labels, test_size=0.2, random_state=42)

# モデルの構築
model = Sequential()
model.add(Embedding(input_dim=len(le.classes_), output_dim=100, input_length=151))
#model.add(Dropout(0.5))  # ドロップアウト層を追加
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(150))
model.add(Dropout(0.1))  # LSTM層の後にもドロップアウト層を追加
model.add(Dense(2, activation='softmax'))

# モデルのコンパイル
optimizer = Adam(learning_rate=0.01)  # 学習率を変更
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 早期停止の設定
#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# モデルのトレーニング（エポック数を増やしてみる）
#model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
# モデルの評価
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# 学習曲線の可視化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# スコアを表形式で表示
import pandas as pd
df_scores = pd.DataFrame({
    "Metric": ["Loss", "Accuracy"],
    "Test Score": [loss, accuracy]
})
print(df_scores)