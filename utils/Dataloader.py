import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms
#from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

class PaddyDataset(Dataset):
    def __init__(self, csv_file_path, root_dir, transform=None):
        #pandasでcsvデータの読み出し
        self.image_dataframe = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        #画像データへの処理
        self.transform = transform

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx): #idx=画像データセットのパス
        #dataframeから画像へのパスとラベルを読み出す
        #画像名
        name = self.image_dataframe.iat[idx, 0]
        img_name = os.path.join(self.root_dir, name)
        #画像のクラスラベル
        label = self.image_dataframe.iat[idx, 1]
        #稲の品種
        variety = self.image_dataframe.iat[idx, 2]
        #年齢
        age = self.image_dataframe.iat[idx, 3]
       
        #画像の読み込み
        image = io.imread(img_name)
        image = resize(image,(600,600),anti_aliasing=True)

        return image, name, label, variety, age



