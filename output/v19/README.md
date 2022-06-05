## 結果
Score: 0.94463　（best.pth)
Score: 0.91234 (10epoch.pth)

## 設定
  - model = EfficientNet.from_pretrained('efficientnet-b4')
  - optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
  - 10 epoch
