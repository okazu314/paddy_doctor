## 結果
Score: 0.95040

- 学習の設定
  - epoch = 30  //実際は、21エポック目でエラー停止
- モデル
  - model = EfficientNet.from_pretrained('efficientnet-b4')
- 損失関数
  - optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
  - https://github.com/pytorch/pytorch/pull/50620
  - https://pytorch.org/vision/stable/models.html

## Error
- message
    - Version 22 was canceled after 43200.2s (timeout exceeded)
    - Your notebook was stopped because it exceeded the max allowed execution duration. Exit code: 137

- https://www.kaggle.com/product-feedback/168892　

137は基本的にキャンセルを意味します。これは多くの理由で発生する可能性があります。実行情報の理由は正確である必要があるため、これはタイムアウトでした。TPUのタイムアウト制限は3時間です。これは、CPU / GPUの9時間のタイムアウトよりもはるかに短い時間です。これは、TPUがより短い時間でより多くのことを実行でき、すべての人と共有するために、より早く解放する必要があるためです。

（関係あるかわからない。一応こんなのありました）

