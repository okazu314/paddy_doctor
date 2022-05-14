import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.autograd import Variable
import torchvision.models
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import recall_score
#from torchvision import transforms
#from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.Dataloader import PaddyDataset
# from models.nnmodels import nnmodels モデルの呼び出しに使っていたが，今回は使わない

loss_list = []
val_loss_list = []
val_acc_list = []
test_loss_list = []
test_acc_list = []

def train(epoch):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        image=image.permute(0,3,1,2)
        #image=torch.from_numpy(image).float()
        output = model(image.float())
        loss = criterion(output, label)
        print(loss.data)
        loss.backward()
        optimizer.step()
        
        print(loss)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data))
    loss_list.append(loss)

def val():
    model.eval()
    val_loss = 0
    correct = 0
    for (image, label) in val_loader:
        image, label = Variable(image.float(), requires_grad=True), Variable(label)
        image=image.permute(0,3,1,2)
        output = model(image)
        val_loss += criterion(output, label) # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()
        
    acc=100. * correct / len(val_loader.dataset)
    # 後であってるかどうか確認
    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    val_loss_list.append(val_loss)
    val_acc_list.append(acc)
    return acc

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for (image, label) in test_loader:
        image, label = Variable(image.float(), volatile=True), Variable(label)
        image=image.permute(0,3,1,2)
        output = model(image)
        test_loss += criterion(output, label) # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()
        
    # recall = recall_score(y_true=label.cpu(), y_pred=pred.cpu(), pos_label=1) # recallの計算
    acc=100. * correct / len(test_loader.dataset) # 正解率の計算
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Recall: {}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc, recall))
    #test_loss_list.append(test_loss)
    #test_acc_list.append(acc)

# config
#INPUT_DIR = './Dataset/'
OUTPUT_DIR = './output/'
#Image_DIR=INPUT_DIR + 'Data/'

# Dataset
# 今回のデータ用に書き換える
Image_DIR=INPUT_DIR + 'Data2/' #test(抽出なし)の時はこれ
#train_Dir = INPUT_DIR+'train.csv' # train水増し付き（古）
train_Dir = INPUT_DIR+'train_not_aug.csv' # train水増しなし（古）

# trainとvalの分割用コードを追加


test_Dir = INPUT_DIR+'test.csv' # テストデータ(抽出なし)
test_Dir = INPUT_DIR+'test_scrutiny.csv' # テストデータ（抽出あり）

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPUがあるなら使用する
epoch = 30 # エポック数の設定

# Dataloaderの呼び出し
train_data = PaddyDataset(train_Dir,Image_DIR)
test_data = PaddyDataset(test_Dir,Image_DIR)
val_data = PaddyDataset(val_Dir,Image_DIR)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data,batch_size=32, shuffle=True)

# 自作モデルの設定
# 今回は使わない
'''
model = nnmodels()
model.to(device)
'''

# 事前学習済みresnet50を利用
model = torchvision.models.resnet50(pretrained = True)
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)

# 事前学習済みEfficientNetを利用
'''
model = EfficientNet.from_pretrained('efficientnet-b7')
model._fc = nn.Linear(model._fc.in_features, 10)
model.to(device)
'''

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
'''
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
'''
criterion = nn.CrossEntropyLoss()
best_acc = -1

for i in range(epoch):
    train(i)
    accuracy = val()
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.to('cpu').state_dict(), OUTPUT_DIR + 'best.pth')


# 学習の可視化
# 学習時の損失の可視化
plt.figure()
plt.plot(range(epoch), loss_list, 'r-', label = 'train_loss')
plt.plot(range(epoch), val_loss_list, 'b-', label = 'val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.savefig('./output/loss_train.png')
# 正答率の可視化
plt.figure()
plt.plot(range(epoch), acc_val_list, 'g-', label = 'val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.grid()
plt.savefig('./output/acc_val.png')

# テストの実行
model.load_state_dict(torch.load(OUTPUT_DIR + 'best.pth'))
test()
