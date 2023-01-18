# Instruction for Execution
# python hw04_coco_downloader.py --root_path /home/yangbj/695/PEC/hw5/COCO_Download/Val/
# --coco_json_path /home/yangbj/695/PEC/hw4/annotations/instances_val2014.json
# --class_list "refrigerator" "airplane" "giraffe" "cat" "elephant" "dog" "train" "horse" "boat" "truck" --images_per_class 500
#
# python hw04_coco_downloader.py --root_path /home/yangbj/695/PEC/hw5/COCO_Download/Train/
# --coco_json_path /home/yangbj/695/PEC/hw4/annotations/instances_train2017.json
# --class_list "refrigerator" "airplane" "giraffe" "cat" "elephant" "dog" "train" "horse" "boat" "truck" --images_per_class 2000
# #
# python hw04_coco_downloader.py --root_path /home/yangbj/695/PEC/hw5/COCO_Download/Train/
# --coco_json_path /home/yangbj/695/PEC/hw4/annotations/instances_train2014.json
# --class_list "refrigerator" "airplane" "giraffe" "cat" "elephant" "dog" "train" "horse" --images_per_class 1500


import argparse
import json
import ast
import requests
import os
from PIL import Image
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import logging
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import time
# import progressbar

# required argparse arguments
parser = argparse.ArgumentParser(description = 'HW05 COCO downloader')
parser.add_argument('--root_path', required = True, type = str)
parser.add_argument('--coco_json_path', required = True, type = str)
parser.add_argument('--class_list', required = True, nargs='*', type=str,)
parser.add_argument('--images_per_class', required=True, type=int)
args, args_other = parser.parse_known_args()

def Coco_Downloader(args):
    coco = COCO(args.coco_json_path)
    urls = dict.fromkeys(args.class_list)
    folder_dict = dict.fromkeys(args.class_list)

    for c in args.class_list:
        if not os.path.exists(args.root_path + c):
            os.makedirs(args.root_path + c)
        folder_dict[c] = args.root_path + c
        catIds = coco.getCatIds(c)
        imgIds = coco.getImgIds(catIds=catIds)
        imgs = coco.loadImgs(imgIds)
        urls[c] = [i['coco_url'] for i in imgs]

    for c in args.class_list:
        folder = folder_dict[c]
        url = urls[c]
        print("Downloading " + c + " class")
        for i in tqdm(range(args.images_per_class)):
            per_url = url[i]
            img_name = per_url.split('/')[-1]
            file_path = os.path.join(folder, img_name)

            if os.path.exists(file_path):
                # print("File already exists: " + per_url + "\n " +
                #       "Will skip it and continue to the next one.")
                continue

            try: response = requests.get(per_url, timeout=1)
            except Exception:
                try: response = requests.get(per_url, timeout=1) # try again
                except Exception:
                    print("Tried twice and still no response for: " + per_url + "\n " +
                           "Will skip it and continue to the next one.")
                    continue


            with open(file_path, 'wb') as im:
                im.write(response.content)

            img = Image.open(file_path)
            # img_resize = img.resize((64, 64), Image.BOX)
            # img_resize.save(file_path)
            img.save(file_path)
        print("Class "+ c +" Finished!")

Coco_Downloader(args)
