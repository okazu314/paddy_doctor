## 結果
- Score: 0.97270  (best.pth)
- Score: 0.97770  (20epoch.pth)

## 設定
- model = EfficientNet.from_pretrained('efficientnet-b4')
- optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001)
- 20 epoch
