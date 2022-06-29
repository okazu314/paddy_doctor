model.load_state_dict(torch.load("../input/epoch10/10epoch.pth"))
for i in range(epoch):
    train(i)
    accuracy = val()
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.to('cpu').state_dict(),'best.pth')
        model.to(device)
