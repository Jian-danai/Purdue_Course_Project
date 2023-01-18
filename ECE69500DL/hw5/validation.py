import torch
import torch.nn as nn
import network
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import testdataloader, testdata_loader
import copy
import sklearn.metrics
import os
import sys
import glob
import seaborn
import cv2

def test(name, net, transform, device,
         batch_size, data_path, save_path,
         jsonpath, resized_data_path, resize):

    coco_data = testdata_loader.Hw05_Coco_Dataset\
        (datapath = data_path, transform = transform,
         resize_w = resize[0], resize_h = resize[1],
         jsonpath = jsonpath, resized_data_path = resized_data_path)

    test_data = torch.utils.data.DataLoader(coco_data,
                     batch_size=batch_size, shuffle=False, num_workers=2)
    print("test data len: ",len(test_data))

    net = copy.deepcopy(net)
    net.load_state_dict(torch.load('/home/yangbj/695/hw5/net50lr1e-4_wo_softmax.pth'))
    net = net.to(device)
    net.eval()

    print("\n\nStarting testing...")
    output_total = []
    label_total = []

    for i, data in enumerate(test_data):
        inputs, labels, bbox, img_path = data
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        bbox = torch.transpose(torch.stack(bbox), 0, 1)
        bbox = bbox.int().numpy()#.to(torch.float32)#.to(device)
        classify, boundingbox = net(inputs)
        boundingbox = boundingbox.cpu().int().numpy()
        prediction = [torch.argmax(output).cpu() for output in classify]
        output_total = output_total + prediction
        labels = [label.cpu() for label in labels]
        label_total = label_total + labels

        for i in range(inputs.shape[0]):
            img = cv2.imread(img_path[i])
            cv2.rectangle(img, (bbox[i][0], bbox[i][1]),
                          (bbox[i][0]+bbox[i][2], bbox[i][1]+bbox[i][3]),
                          (0, 0, 255), 1)
            cv2.rectangle(img, (boundingbox[i][0], boundingbox[i][1]),
                          (boundingbox[i][0]+boundingbox[i][2],
                           boundingbox[i][1]+boundingbox[i][3]), (0, 255, 0), 1)
            if not os.path.exists(save_path+'tested_imgs/'+
                                  str(labels[i]) +'/'):
                os.makedirs(save_path+'tested_imgs/'+
                            str(labels[i]) +'/')
            cv2.imwrite(save_path+'tested_imgs/'+
                        str(labels[i]) +'/'+img_path[i][-16:], img)
            print("save an img to:" + save_path+
                  'tested_imgs/'+str(labels[i]) +'/'+img_path[i][-16:])

    # calculate confusion matrix with sklearn module
    confus_matrix = sklearn.metrics.confusion_matrix(
        label_total, output_total, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(confus_matrix)

    # classification accuracy calculation
    acc = 0
    for i in range(confus_matrix.shape[0]):
        acc += confus_matrix[i][i]
    Accuracy = acc / confus_matrix.sum() * 100
    print('Accuracy:'+str(Accuracy)+'%')

    # plot confusion matrix
    plt_labels = []
    path_list = sorted(glob.glob(data_path + '/*'))
    for p in path_list:
        plt_labels.append(os.path.splitext(os.path.basename(p))[0])
    plt.figure(figsize = (10,7))
    seaborn.heatmap(confus_matrix, annot=True, fmt= 'd', linewidths = .5,
                    xticklabels= plt_labels, yticklabels= plt_labels)
    plt.title("Net" + name + " " + 'Accuracy:'+str(Accuracy)+'%')
    plt.savefig(save_path + "net_"+name+"confusion_matrix.jpg")



if __name__ == '__main__':

    # settings
    device  = torch.device('cuda:2')
    transform = tvt.Compose([tvt.ToTensor(),
                             tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_path = "/home/yangbj/695/YANG/hw5/COCO_Download/Val"
    save_path = "/home/yangbj/695/hw5/"
    jsonpath = "/home/yangbj/695/YANG/hw4/annotations/instances_val2014.json"
    resized_data_path = "/home/yangbj/695/YANG/hw5/hw5_coco_data/Val/"
    batch_size = 1
    name = '1'
    resize = [128, 128]
    net = network.Single_LOAD_Net()

    # Run test
    test(name = name,  net= net,
         transform = transform, device = device, batch_size = batch_size,
         data_path = dataset_path, save_path = save_path,
         jsonpath = jsonpath,
         resized_data_path = resized_data_path, resize = resize)
