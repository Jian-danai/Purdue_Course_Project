import random
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import torchvision.transforms.functional as TF
import os
from utils import tensor2img


class Unpaired_Data(Dataset):
    def __init__(self, root="/data_new/bjyang/ece50024_cyclegan/", mode='train', name='apple2orange'):
        self.root = root
        self.mode = mode
        train_filesA = glob.glob(self.root+name+'/'+mode+'A/*.jpg')
        train_filesB = glob.glob(self.root+name+'/'+mode+'B/*.jpg')
        self.input_filesA = train_filesA[:]  
        self.input_filesB = train_filesB[:]
        self.len = max(len(train_filesA), len(train_filesB))

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        img_nameA = self.input_filesA[index%len(self.input_filesA)].split('/')[-1]
        img_inputA = Image.open(self.input_filesA[index%len(self.input_filesA)]).convert('RGB').resize((256, 256))
        img_nameB = self.input_filesB[index%len(self.input_filesB)].split('/')[-1]
        img_inputB = Image.open(self.input_filesB[index%len(self.input_filesB)]).convert('RGB').resize((256, 256))
        if self.mode == 'train':
            # random rotate
            if random.random() > 0.5:
                img_inputA = TF.rotate(img_inputA, 180)
            if random.random() > 0.5:
                img_inputB = TF.rotate(img_inputB, 180)
            # random horizontal flip
            if random.random() > 0.5:
                img_inputA = TF.hflip(img_inputA)
            if random.random() > 0.5:
                img_inputB = TF.hflip(img_inputB)
            # random vertical flip
            if random.random() > 0.5:
                img_inputA = TF.vflip(img_inputA)
            if random.random() > 0.5:
                img_inputB = TF.vflip(img_inputB)
        # convert to [-1, 1] tensor
        img_inputA = 2*TF.to_tensor(img_inputA)-1
        img_inputB = 2*TF.to_tensor(img_inputB)-1
        return {"img_name": img_nameA, "ref_name": img_nameB, "img": img_inputA, "ref": img_inputB}


class test_img(Dataset):
    def __init__(self, root="/data_new/bjyang/ece50024_cyclegan/", mode='test', name='apple2orange', category='A'):
        self.root = root
        self.mode = mode
        train_filesA = glob.glob(self.root+name+'/'+mode+category+'/*.jpg')
        self.input_filesA = train_filesA[:]  
        self.len = len(train_filesA)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        img_nameA = self.input_filesA[index%len(self.input_filesA)].split('/')[-1]
        img_inputA = Image.open(self.input_filesA[index%len(self.input_filesA)]).convert('RGB').resize((256, 256))
        # convert to [-1, 1] tensor
        img_inputA = 2*TF.to_tensor(img_inputA)-1
        return {"img_name": img_nameA, "img": img_inputA}
