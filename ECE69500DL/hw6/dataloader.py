import torch
import torchvision
import torchvision.transforms as tvt
from torch.utils.data import DataLoader, Dataset
import glob
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
import cv2
from skimage import io
import matplotlib as plt

class hw6_dataloader(Dataset):
    def __init__(self, datapath, cocodatapath, transform, resize_w, resize_h, resized_path):
        self.tvtNorm = transform
        max_instance = 5
        self.num_archor_box = max_instance
        class_list = ['dog', 'cat', 'horse']
        self.label_list = []
        self.ID_list    = []
        self.bbox_list  = []
        self.exists     = []
        self.filename   = []
        self.skip = 0
        self.n_class = len(class_list)
        coco = COCO(cocodatapath)
        
        for i in range(self.n_class):
            if not os.path.exists(resized_path + class_list[i]):
                os.makedirs(resized_path + class_list[i])
            catIds = coco.getCatIds(catNms=class_list[i])
            imgIds = coco.getImgIds(catIds=catIds)
            # print(imgIds)
            # file_target = os.path.join(datapath, class_list[i])
            number_of_img = len(imgIds)
            print("class: " + class_list[i])
            print(number_of_img)
            for k in range(number_of_img):
                # print(k)
                annIds = coco.getAnnIds(imgIds=imgIds[k], catIds=catIds, iscrowd=False)
                anns = coco.loadAnns(annIds)
                # Select only images with 2 to max_instance interested bboxs
                num_of_bbox_interested = 0
                for ann in range(len(anns)):
                    if anns[ann]['category_id']==17 or anns[ann]['category_id']==18 or anns[ann]['category_id']==19:
                        num_of_bbox_interested+=1
                # print("num_of_bbox_interested: ", num_of_bbox_interested)
                if num_of_bbox_interested > max_instance:
                    # print("num_of_bbox_interested >5: ", num_of_bbox_interested)
                    continue
                elif num_of_bbox_interested < 2:
                    # print("num_of_bbox_interested <2: ", num_of_bbox_interested)
                    continue
                else:
                    labels = np.zeros(max_instance)
                    bboxs = np.zeros((max_instance ,4))
                    exists = np.zeros(max_instance)
                    image_id_str = str(imgIds[k])
                    image_id_str = image_id_str.zfill(12)
                    file_list = os.path.join(datapath, class_list[i], image_id_str + ".jpg")
                    # print(file_list)
                    if not os.path.exists(file_list):
                        # print("file not exist: " + image_id_str + ".jpg")
                        continue
                    image = Image.open(file_list)
                    w_i, h_i = image.size
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    # Resize image to 128*128
                    resized_file_path = resized_path + class_list[i] + "/" + image_id_str + ".jpg"
                    if not os.path.exists(resized_file_path):
                        img_resize = image.resize((resize_h, resize_w), Image.BOX)
                        img_resize.save(resized_file_path)
                    for j in range(len(anns)):
                        image_ann   = anns[j]
                        category_id = image_ann['category_id']
                        if category_id!=17 and category_id!=18  and category_id!=19:
                            # print("skip!!", category_id)
                            continue
                        else:
                            labels[j] = category_id-17
                            exists[j]   = 1
                            image_bbox = np.array(image_ann['bbox'])
                            # Resize bbox
                            image_bbox[0] = image_bbox[0] * resize_w / w_i
                            image_bbox[1] = image_bbox[1] * resize_h / h_i
                            image_bbox[2] = image_bbox[2] * resize_w / w_i
                            image_bbox[3] = image_bbox[3] * resize_h / h_i
                            # if image_bbox[0]+image_bbox[2]>128 or image_bbox[1]+image_bbox[3]>128:
                            #     print("image_bbox", image_bbox)
                            bboxs[j,:] = image_bbox

                    self.bbox_list.append(bboxs)
                    self.label_list.append(labels)
                    self.exists.append(exists)
                    self.ID_list.append(resized_file_path)
                    # print('image ' + str(imgIds[k]) + ' is done!')
                    # print(labels)


    def __getitem__ (self, ID) :
        # print(self.ID_list)
        img_path = self.ID_list[ID]
        # Data in tensor
        X = self.tvtNorm(Image.open(img_path).convert('RGB'))
        # Label
        y_label = self.label_list[ID]#((self.num_archor_box-1)*ID):(self.num_archor_box*ID)
        bbox_label = self.bbox_list[ID]#((self.num_archor_box-1)*ID):(self.num_archor_box*ID)
        num_obj_in_img = np.sum(self.exists[ID])#((self.num_archor_box-1)*ID):(self.num_archor_box*ID)

        return X, bbox_label, y_label, num_obj_in_img, img_path

    def __len__ (self) :
        return len(self.ID_list) # Total nb of sample
