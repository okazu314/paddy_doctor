## 結果
Score: 0.97231
- モデル
  - model = EfficientNet.from_pretrained('efficientnet-b4')
- 損失関数
  - optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
- 学習の設定
  - 20 epoch. 最終エポックの重みでテストを行う。 
