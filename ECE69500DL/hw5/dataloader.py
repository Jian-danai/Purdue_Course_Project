import torch
import glob
import os
from PIL import Image
import numpy as np
from pycocotools.coco import COCO

class Hw05_Coco_Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, datapath, transform, resize_w,
               resize_h, jsonpath, resized_data_path):
        'Initialization'
        self.skipnumber = 0
        self.transform = transform
        path_list = glob.glob(datapath + '/*')
        coco = COCO(jsonpath)
        class_list = [os.path.splitext(os.path.basename(p))[0] for p in path_list]
        self.list_IDs = []
        self.labels = []
        self.bbox = []
        n_class = len(path_list)

        # for each class
        for p in range(n_class):
            img_path_list = sorted(glob.glob(path_list[p] + '/*'))
            catIds = coco.getCatIds(catNms=class_list[p])
            imgIds = coco.getImgIds(catIds=catIds)

            # for each sample
            for k in range(len(img_path_list)):

                # get annotions for bbox
                annIds = coco.getAnnIds(imgIds=imgIds[k], catIds=catIds, iscrowd=False)
                annotions = coco.loadAnns(annIds)
                Max = 0
                Max_index = 0
                for j in range(len(annotions)):
                    ann_j = annotions[j]
                    bbox = ann_j['bbox'] # bbox = [x,y,w,h]
                    w, h = bbox[2], bbox[3]
                    Area = w * h
                    if Area > Max:
                        Max = Area
                        Max_index = j
                annotion = annotions[Max_index]
                boundingbox = annotion['bbox']

                # get annotions for imgs
                img_id = annotion['image_id']
                img_id_str = str(img_id)
                img_id_str = img_id_str.zfill(12)
                img_path = os.path.join(datapath, class_list[p],
                                        'COCO_val2014_' + img_id_str + ".jpg")
                img_sample = Image.open(img_path)
                w, h = img_sample.size

                # skip small boundbox samples
                if boundingbox[2]< w/3 or boundingbox[3]< w/3:
                    self.skipnumber += 1
                    print("skip", self.skipnumber)
                    continue

                # save resized images
                resized_save_path = os.path.join(resized_data_path,
                                                 class_list[p], img_id_str + ".jpg")
                if not os.path.exists(resized_save_path):
                    img_resize = img_sample.resize((resize_h, resize_w), Image.BOX)
                    img_resize.save(resized_save_path)

                # rescale boundingbox
                boundingbox[0] = boundingbox[0] * resize_w / w
                boundingbox[1] = boundingbox[1] * resize_h / h
                boundingbox[2] = boundingbox[2] * resize_w / w
                boundingbox[3] = boundingbox[3] * resize_h / h

                # append list_ids, labels, and bbox for getitem function
                self.list_IDs.append(resized_save_path)
                self.labels.append(p)
                self.bbox.append(boundingbox)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_path = self.list_IDs[index]
        label = self.labels[index]
        bbox = self.bbox[index]

        # Load data
        img = Image.open(img_path)
        if img.mode!='RGB':
            img = img.convert('RGB')
        img = self.transform(img).to(dtype = torch.float32)

        return img, label, bbox, img_path
