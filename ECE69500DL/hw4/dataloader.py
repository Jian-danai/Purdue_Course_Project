import torch
import glob
import os
from PIL import Image
# import numpy as np
class Hw04_Coco_Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, datapath, transform):
        'Initialization'
        self.transform = transform
        path_list = sorted(glob.glob(datapath + '/*'))
        self.list_IDs = []
        self.labels = []
        n_class = len(path_list)
        for p in range(len(path_list)):
            # class_list.append(os.path.splitext
            # (os.path.basename(p))[0])
            img_path_list = sorted(glob.glob
                                   (path_list[p] + '/*'))
            self.list_IDs = self.list_IDs + img_path_list
            # base = np.zeros(n_class)
            # base[p] = 1
            self.labels = self.labels + [p] * \
                          len(img_path_list)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_path = self.list_IDs[index]

        # Load data and get label
        X = self.transform(Image.open(img_path)).\
            to(dtype = torch.float32)
        y = self.labels[index]

        return X, y

# cocodata = Hw04_Coco_Dataset("/home/bjyang/
# 695/hw4/hw04_coco_data/Train")
# print(cocodata.__len__())
# img, label = cocodata.__getitem__(2500)
# print(img)
# print(label)
