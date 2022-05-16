import pandas as pd
import os


def Separate_data(path):
    # 入力データを格納するためのディレクトリ作成(作成済であればスキップ)
    os.makedirs('../input/', exist_ok=True)

    # オリジナル訓練データを読み込み
    ori_train_data = pd.read_csv(path)

    # データの最初の80%を訓練データ，残りの20%を検証データ
    train_data = ori_train_data.sample(frac=0.8)
    val_data = ori_train_data.drop(train_data.index)

    # 訓練データと検証データをcsvファイルに保存
    train_data.to_csv('../input/train2.csv')
    val_data.to_csv('../input/val.csv')
