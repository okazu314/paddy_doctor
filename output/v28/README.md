## 結果
- Score: 0.96078  (best.pth)
- Score: 0.91964  (10epoch)

## 設定
- model = timm.create_model('coat_lite_small',
- optimizer = torch.optim.RAdam(model.parameters(),lr=0.0001)
- 10epoch, 32batch size
