import torch
import torch.nn as nn
import network
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import dataloader
import copy
import sklearn.metrics
import os
import sys
import glob
import seaborn
# import sklearn.metrics.confusion_matrix
# as confusion_matrix

def test(name, net1, transform, device, lr = 1e-3,
         momentum = 0.9, epochs = 10,
          batch_size = 10, data_path = "/home/bjyang"
          "/695/hw4/hw04_coco_data/Val",
          save_path = "/home/bjyang/695/hw4/"):

    coco_data = dataloader.Hw04_Coco_Dataset\
        (data_path, transform)
    test_data = torch.utils.data.DataLoader\
        (coco_data, batch_size=batch_size,
         shuffle=True, num_workers=2)
    # net1 = network.Net1()
    net1 = copy.deepcopy(net1)
    net1.load_state_dict(torch.load(save_path
                                    + 'net'+name+'.pth'))
    net1 = net1.to(device)
    net1.eval()

    print("\n\nStarting testing loop")
    output_total = []
    label_total = []
    for i, data in enumerate(test_data):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net1(inputs)
        prediction = [torch.argmax(output).cpu()
                      for output in outputs]
        output_total = output_total + prediction
        labels = [label.cpu() for label in labels]
        # print(prediction)
        # print(labels)
        label_total = label_total + labels
    confus_matrix = sklearn.metrics.confusion_matrix\
        (label_total, output_total, labels=[0, 1, 2,
                                            3, 4, 5,
                                            6, 7, 8,
                                            9])
    print(confus_matrix)
    acc = 0
    for i in range(confus_matrix.shape[0]):
        acc += confus_matrix[i][i]
    Accuracy = acc / confus_matrix.sum() * 100
    plt_labels = []
    path_list = sorted(glob.glob(data_path + '/*'))
    for p in path_list:
        plt_labels.append(os.path.splitext(os.path.
                                           basename
                                           (p))[0])
    plt.figure(figsize = (10,7))
    seaborn.heatmap(confus_matrix, annot=True,
                    fmt= 'd', linewidths = .5,
                    xticklabels= plt_labels,
                    yticklabels= plt_labels)
    plt.title("Net" + name + " " + 'Accuracy:'
              +str(Accuracy)+'%')

    # cmd = sklearn.metrics.ConfusionMa
    # trixDisplay(confus_matrix, display_labels=plt_labels)
    # cmd.plot(xticks_rotation=15.0)
    plt.savefig(save_path + "net_"+name+
                "confusion_matrix.jpg")



if __name__ == '__main__':
    device  = torch.device('cuda:0')
    transform = tvt.Compose([tvt.ToTensor(),
                             tvt.Normalize
                             ((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    dataset_path = "/home/bjyang/695/hw4/" \
                   "hw04_coco_data/Val"
    save_path = "/home/bjyang/695/hw4/"

    test(name = '1',  net1= network.Net1(),
         transform = transform, device =
         device, lr = 1e-3, momentum = 0.9,
          epochs = 10, batch_size = 10,
         data_path = dataset_path,
         save_path = save_path)
    test(name = '2',  net1=network.Net2(),
         transform=transform, device=
         device, lr=1e-3, momentum=0.9,
         epochs=10, batch_size=10,
         data_path=dataset_path,
         save_path=save_path)
    test(name = '3',  net1=network.Net3(),
         transform=transform, device=
         device, lr=1e-3, momentum=0.9,
         epochs=10, batch_size=10,
         data_path=dataset_path,
         save_path=save_path)
