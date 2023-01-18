import torch
import torch.nn as nn
import network
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import dataloader
import copy

def train(transform, device, lr = 1e-3, momentum = 0.9, epochs = 10,
          batch_size = 10, data_path = "/home/yangbj/695/YANG/hw5/COCO_Download/Train",
          save_path = "/home/yangbj/695/hw5/",
          cocodatapath = "/home/yangbj/695/YANG/hw4/annotations/instances_train2017.json"):

    coco_data = dataloader.hw5_dataloader(datapath=data_path, cocodatapath=cocodatapath,
                                     transform=transform, resize_w=128, resize_h=128)
    train_data = torch.utils.data.DataLoader(coco_data, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    net = network.Single_LOAD_Net()
    net = copy.deepcopy(net)
    # net.load_state_dict(torch.load("/home/yangbj/695/hw5/net50.pth"))
    net = net.to(device)
    running_loss1 = []
    running_loss2 = []
    loss_item1 = 0
    loss_item2 = 0
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    print("\n\nStarting training...")
    for epoch in range(epochs):
        print("")
        running_loss = 0.0
        for i, data in enumerate(train_data):
            inputs, labels, bbox = data
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            bbox = torch.transpose(torch.stack(bbox), 0, 1)
            bbox = bbox.to(torch.float32).to(device)
            optimizer.zero_grad()
            classify, boundingbox = net(inputs)
            loss1 = criterion1(classify, labels)
            # print("classify: ",classify)
            # print("labels: ", labels)
            # print("loss1: ",loss1)
            loss1.backward(retain_graph=True)
            loss2  = criterion2(boundingbox, bbox)
            # print("boundingbox: ", boundingbox)
            # print("bbox: ", bbox)
            # print("loss2: ", loss2)
            loss2.backward()
            optimizer.step()
            loss_item1 += loss1.item()
            loss_item2 += loss2.item()
            if (i+1) % 500 == 0:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" %
                      (epoch + 1, i + 1, loss_item1 / float(500)))
                print("\n[epoch:%d, batch:%5d] loss: %.3f" %
                      (epoch + 1, i + 1, loss_item2 / float(500)))
                running_loss1.append(loss_item1/float(500))
                running_loss2.append(loss_item2 / float(500))
                loss_item1, loss_item2 = 0.0, 0.0
    torch.save(net.state_dict(), save_path+'net50lr1e-4_wo_softmax.pth')

    return running_loss1, running_loss2


if __name__ == '__main__':
    device  = torch.device('cuda:1')
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_path = "/home/yangbj/695/YANG/hw5/COCO_Download/Train"
    coco_json = "/home/yangbj/695/YANG/hw4/annotations/instances_train2017.json"
    save_path = "/home/yangbj/695/hw5/"
    running_loss1, running_loss2 = train(transform = transform, device = device,
    lr = 1e-4, momentum = 0.9,epochs = 50, batch_size = 10,
    data_path = trainset_path, save_path = save_path,
    cocodatapath = "/home/yangbj/695/YANG/hw4/annotations/instances_train2017.json")

    plt.figure()
    plt.title('Train Loss1')
    plt.xlabel('Per 500 Iterations')
    plt.ylabel('Loss')
    plt.plot(running_loss1, label = 'Loss1')
    # plt.plot(running_loss2, label = 'Loss2')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path + "train_loss1_1lr1e-4_wo_softmax" + ".jpg")
    plt.figure()
    plt.title('Train Loss2')
    plt.xlabel('Per 500 Iterations')
    plt.ylabel('Loss')
    # plt.plot(running_loss1, label='Loss1')
    plt.plot(running_loss2, label='Loss2')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path + "train_loss2_1lr1e-4_wo_softmax" + ".jpg")
