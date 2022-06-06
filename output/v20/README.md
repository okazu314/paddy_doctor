## 結果
  - ベストの重みを使用した正答率は97.424%
  - 10エポック目の重みを使用した正答率は97.424%

## 実験設定
- model = EfficientNet.from_pretrained('efficientnet-b4')
- optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)
- 
