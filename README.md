# CNN-LSTM Malware Detection

## 概要
このプロジェクトは、マルウェア検出のためのCNN-LSTMモデルを開発することを目的としています。モデルは、API呼び出しのシーケンスデータを使用して、マルウェアと無害なソフトウェアを識別します。

## データセット
このプロジェクトでは、マルウェア検出のためにMalBehavD-V1データセットを使用しました。MalBehavD-V1は、様々なマルウェアと無害なソフトウェアのAPI呼び出しログを含むデータセットです。

### 引用
データセットは以下から入手しました：

- MalBehavD-V1 GitHubリポジトリ: [MalBehavD-V1](https://github.com/mpasco/MalbehavD-V1.git)

## インストール
このプロジェクトを実行するには、以下の手順に従ってください：

```bash
git clone https://github.com/Hyuhyuhyu661/CNNLSTM1.git
cd CNNLSTM1
python train.py




- train.py             #メインのプログラムファイル
- dC.py                #サンプル数を数えるプログラム
- SC.py                #ユニークなAPI呼び出しを数えるプログラム
- train_lstm_unit.py   #lstm層のパラメータを変化させて精度を検証するプログラム
- train_output_dim.py  #output_dimを変化させて精度を検証するプログラム

