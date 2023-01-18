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
import cv2
from torchvision.utils import save_image

def test(transform, device, data_path, data_path2, save_path,
         generator_path, discriminator_path):
    CelebA_data = dataloader.CelebA_Dataset(data_path, data_path2, transform)
    test_data = torch.utils.data.DataLoader(CelebA_data, batch_size=1,
                                            shuffle=False, num_workers=2)
    generator = network.Generator()
    generator.load_state_dict(torch.load(generator_path))
    generator = generator.to(device)
    discriminator = network.Discriminator()
    discriminator.load_state_dict(torch.load(discriminator_path))
    discriminator = discriminator.to(device)
    generator.eval()
    discriminator.eval()

    print("\n\nStarting testing...")
    fake_pred = []
    real_pred = []

    for i, data in enumerate(test_data):
        real_img = data
        real_img = real_img.to(device)
        noise = torch.randn(real_img.size(0), 100, 1, 1, device=device)
        fake_img = generator(noise)
        fake_output = discriminator(fake_img).squeeze(0).squeeze(-1).squeeze(-1)
        real_output = discriminator(real_img).squeeze(0).squeeze(-1).squeeze(-1)
        if fake_output<0.5:
            fake_pred.append(1)
        else:
            fake_pred.append(0)
        if real_output>=0.5:
            real_pred.append(1)
        else:
            real_pred.append(0)
        save_image((fake_img[0]+1)/2, save_path + 'img' + str(i) + '.png')
        # for i in range(inputs.shape[0]):
        #     img = cv2.imread(img_path[i])
        #     cv2.rectangle(img, (bbox[i][0], bbox[i][1]),
        #                   (bbox[i][0]+bbox[i][2], bbox[i][1]+bbox[i][3]),
        #                   (0, 0, 255), 1)
        #     cv2.rectangle(img, (boundingbox[i][0], boundingbox[i][1]),
        #                   (boundingbox[i][0]+boundingbox[i][2],
        #                    boundingbox[i][1]+boundingbox[i][3]), (0, 255, 0), 1)
        #     if not os.path.exists(save_path+'tested_imgs/'+
        #                           str(labels[i]) +'/'):
        #         os.makedirs(save_path+'tested_imgs/'+
        #                     str(labels[i]) +'/')
        #     cv2.imwrite(save_path+'tested_imgs/'+
        #                 str(labels[i]) +'/'+img_path[i][-16:], img)
        #     print("save an img to:" + save_path+
        #           'tested_imgs/'+str(labels[i]) +'/'+img_path[i][-16:])

    # calculate confusion matrix with sklearn module
    # confus_matrix = sklearn.metrics.confusion_matrix(
    #     label_total, output_total, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(confus_matrix)

    # classification accuracy calculation
    # acc = 0
    # for i in range(confus_matrix.shape[0]):
    #     acc += confus_matrix[i][i]
    # Accuracy = acc / confus_matrix.sum() * 100
    fake_acc = sum(fake_pred)/len(fake_pred)*100
    real_acc = sum(real_pred) / len(real_pred) * 100

    print('Fake Image Classification Accuracy: '+str(fake_acc)+'%')
    print('Real Image Classification Accuracy: ' + str(real_acc) + '%')

    # plot confusion matrix
    # plt_labels = []
    # path_list = sorted(glob.glob(data_path + '/*'))
    # for p in path_list:
    #     plt_labels.append(os.path.splitext(os.path.basename(p))[0])
    # plt.figure(figsize = (10,7))
    # seaborn.heatmap(confus_matrix, annot=True, fmt= 'd', linewidths = .5,
    #                 xticklabels= plt_labels, yticklabels= plt_labels)
    # plt.title("Net" + name + " " + 'Accuracy:'+str(Accuracy)+'%')
    # plt.savefig(save_path + "net_"+name+"confusion_matrix.jpg")



if __name__ == '__main__':

    # settings
    device  = torch.device('cuda:1')
    transform = tvt.Compose([tvt.ToTensor(),
                             tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_path = "/home/yangbj/695/hw7/Data/Test"
    save_path = "/home/yangbj/695/hw7/Results7/"
    generator_path = "/home/yangbj/695/hw7/generator7_9.pth"
    discriminator_path = "/home/yangbj/695/hw7/discriminator7_9.pth"
    # batch_size = 1
    # name = '1'
    # resize = [128, 128]
    # net = network.Single_LOAD_Net()

    # Run test
    test(transform = transform, device = device,
         data_path = dataset_path,
         data_path2="/home/yangbj/695/hw7/Data/Train",
         save_path = save_path, generator_path = generator_path,
         discriminator_path=discriminator_path)
