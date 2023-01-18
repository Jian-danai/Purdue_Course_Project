import torch
import glob
# import os
from PIL import Image


class CelebA_Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, datapath1, datapath2, transform):
        'Initialization'
        self.transform = transform
        self.list_IDs = sorted(glob.glob(datapath1 + '/*'))+\
                        sorted(glob.glob(datapath2 + '/*'))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        img_path = self.list_IDs[index]
        X = self.transform(Image.open(img_path)).to(
            dtype = torch.float32)

        return X

