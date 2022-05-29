# paddy_doctor

参加：
https://www.kaggle.com/competitions/paddy-disease-classification/overview

稲の病気を画像から検出するというコンペティション

（概要書く）

（画像はる）

# 結果
## 一回目の提出（v12）
EfficientNet-b4を利用し，20エポックで行った．

正答率97.270％程度を達成し，5月27日時点で100チーム中25位という結果を得ました．

(画像はる)


## 二回目の提出（v15）
EfficientNet-b4を利用し，21エポックで行った．
損失関数を、AdamからAdamWに変更した

正答率 95.040％程度で、チームベストを更新することができなかった．
（結果）

（学習曲線を貼る）

## 三回目の提出

# スケジュール
- 5/14 チーム開発環境の構築
    - trainデータをtrainとvaridの２つに分ける関数作成
    - データの4種類の情報を読み取るように変更。テスト結果をcsvファイルに記録する関数作成
    - Kaggle Notebook　でmainプログラム動くように移植する

- 5/20 Kaggle Notebook でmain.pyを動かしてみる（エラーあっていい）
    - 皆：エラー修正を行う(テストデータ用ローダー(TestDataset.py)の作成＋def test()の修正)
    - エラー修正終わったら、改善のアイデア出し

- 5/27 mainプログラムでのエラーがゼロ。改善のアイデアだし。

- 6/3 アイデアを反映したプログラム作成

- 6/10 アイデアを反映したプログラム実装

- 6/17 spyral3

- 6/24(最終日) spyral 3

# 中間反省会
良かった点
- 取り掛かりが早く，連絡をこまめに行いながらのエラー対応を行った．
    - スムーズに進行できた．
    - 後半余裕できた．
- 要求仕様書を作成し，共有したことが円滑なコミュニケーションがうまくいった．
    - 各自の作業の組み合わせがうまくいった．
    - 分担・作業の明確化
- エラー修正など共有しながら行った．
    - 初学者からしたらMLの理解が深まって良かった
    - エラーの共有めっちゃ重要
        - ここ大きかった．

悪かった点
- リーダーの負担が大きかった．
    - リーダーの作業量を減らす．
- リーダーが対応できなかったときに問題を放置してしまった（3日ほど）．
    - サブリーダーを決めておく．



# データ
    train.csv
        image_id：各画像を識別するためのID．train_imagesディレクトリに.jpg形式で入っている．
        label：水稲の病気の種類．9種類の病気ラベル＋1種類の正常ラベル．
            bacterial_leaf_blight：葉枯病(苗がしおれ、葉が黄変・乾燥)
            bacterial_leaf_streak：細菌性葉巻病(葉の褐変と乾燥)
            bacterial_panicle_blight：穂の枯れ(稲全体が白っぽく変色)
            blast：稲熱(いもち)病(稲の葉、襟、節、首、穂軸の一部、時には葉鞘など、地上部のあらゆる部分を侵す)
            brown_spot：葉鞘褐変病？(多数の大きな斑点で、葉全体が枯れる)
            dead_heart：稲の枯れ？
            downy_mildew：うどんこ病(葉の表面に白いカビ)
            hispa：イネヒバ？(虫による病害，葉の上面を削り，下部の表皮を残す)
            normal：正常
            tungro：稲ツングロ病(ヨコバイという虫による病害，葉の変色・生育阻害・蘖（ひこばえ）数の減少・不稔粒や一部充填粒の発生などの症状)
        variety：水稲の品種名．
        age：栽培年数．
    sample_submission.csv
        提出ファイル例
    train_images
        10,407枚の訓練画像が各ラベルディレクトリに入っている．ファイルネームはimage_id.jpg
    test_images
        3,469枚のテスト画像が入っている．
# ルール
    - ソースコードを組む際に実装できない場合や，エラーが出た場合は，各自30分は考える．考えて分からなかった場合はすぐにラインで相談し共有するようにする．
    - Githubで管理する．
    - 週一の報告を設ける．（進捗確認）
    - 要求仕様書を作成し，それに従う．
    - 一人の負担が大きくならないように，チーム内で協力して取り組む．．


# 問題点
    
