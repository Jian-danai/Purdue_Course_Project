import torch
import torch.nn as nn
import network
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import dataloader
import copy

def train(transform, device, lr = 1e-3, momentum = 0.9, epochs = 10,
          batch_size = 10,
          data_path = "/home/bjyang/695/hw4/hw04_coco_data/Train",
          save_path = "/home/bjyang/695/hw4/"):

    coco_data = dataloader.Hw04_Coco_Dataset(data_path,
                                             transform)
    train_data = torch.utils.data.DataLoader(coco_data,
                 batch_size=batch_size, shuffle=True, num_workers=2)
    net1 = network.Net1() # {Net1, Net2, Net3}
    net1 = copy.deepcopy(net1)
    net1 = net1.to(device)
    net2 = network.Net2()  # {Net1, Net2, Net3}
    net2 = copy.deepcopy(net2)
    net2 = net2.to(device)
    net3 = network.Net3()  # {Net1, Net2, Net3}
    net3 = copy.deepcopy(net3)
    net3 = net3.to(device)
    running_loss1 = []
    running_loss2 = []
    running_loss3 = []

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net1.parameters(), lr=lr,
                                momentum=momentum)
    print("\n\nStarting training loop1")
    for epoch in range(epochs):
        print("")
        running_loss = 0.0
        for i, data in enumerate(train_data):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net1(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 500 == 0:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" %
                      (epoch + 1, i + 1, running_loss / float(500)))
                running_loss1.append(running_loss/float(500))
                running_loss = 0.0
    torch.save(net1.state_dict(), save_path+'net1.pth')

    optimizer = torch.optim.SGD(net2.parameters(), lr=lr,
                                momentum=momentum)
    print("\n\nStarting training loop2")
    for epoch in range(epochs):
        print("")
        running_loss = 0.0
        for i, data in enumerate(train_data):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net2(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 500 == 0:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" %
                      (epoch + 1, i + 1, running_loss / float(500)))
                running_loss2.append(running_loss/float(500))
                running_loss = 0.0
    torch.save(net2.state_dict(), save_path+'net2.pth')

    optimizer = torch.optim.SGD(net3.parameters(),
                                lr=lr, momentum=momentum)
    print("\n\nStarting training loop3")
    for epoch in range(epochs):
        print("")
        running_loss = 0.0
        for i, data in enumerate(train_data):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net3(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 500 == 0:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" %
                      (epoch + 1, i + 1, running_loss / float(500)))
                running_loss3.append(running_loss/float(500))
                running_loss = 0.0
    torch.save(net3.state_dict(), save_path+'net3.pth')
    return running_loss1, running_loss2, running_loss3


if __name__ == '__main__':
    device  = torch.device('cuda:0')
    transform = tvt.Compose([tvt.ToTensor(),
                             tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_path = "/home/bjyang/695/hw4/hw04_coco_data/Train"
    save_path = "/home/bjyang/695/hw4/"
    running_loss1, running_loss2, running_loss3 = \
        train(transform = transform, device = device,
              lr = 1e-3, momentum = 0.9,
          epochs = 10, batch_size = 10,
              data_path = trainset_path, save_path = save_path)

    plt.figure()
    plt.title('Train Loss Comparison')
    plt.xlabel('Per 500 Iterations')
    plt.ylabel('Loss')
    plt.plot(running_loss1, label = 'Net1')
    plt.plot(running_loss2, label = 'Net2')
    plt.plot(running_loss3, label = 'Net3')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(save_path + "train_loss" + ".jpg")
