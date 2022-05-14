# paddy_doctor

参加：
https://www.kaggle.com/competitions/paddy-disease-classification/overview

ソースの共有：こっち

実行環境：Kaggle Notebook

# スケジュール
5/14 チーム開発環境の構築

5/20 Kaggle Notebook でmain.pyを動かす

5/27 mainプログラムでのエラーがゼロ。改善のアイデアだし。

6/3 アイデアを反映したプログラム作成

6/10 アイデアを反映したプログラム実装

6/17 spyral3

6/24(最終日) spyral 3

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

# 問題点
    画像の入力