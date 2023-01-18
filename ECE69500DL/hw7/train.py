import torch
import torch.nn as nn
import network
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import dataloader
import copy


def weights_init(m):
    """
    Uses the DCGAN initializations for the weights
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

adversarial_loss = nn.BCELoss()

def generator_loss(fake_output, label):
    gen_loss = adversarial_loss(fake_output, label)
    return gen_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss


def train(transform, device, lr1 = 1e-3, lr2 = 1e-3, epochs = 50, batch_size = 16,
          data_path1 = "/home/yangbj/695/hw7/data/Train",
          data_path2 = "/home/yangbj/695/hw7/data/Test",
          per = 500, save_path = "/home/yangbj/695/hw7/"):

    CelebA_data = dataloader.CelebA_Dataset(data_path1, data_path2, transform)
    # print(CelebA_data.list_IDs)
    train_data = torch.utils.data.DataLoader(CelebA_data,
                batch_size=batch_size, shuffle=True, num_workers=2)
    generator = copy.deepcopy(network.Generator())
    # copy.deepcopy(net)
    generator = generator.to(device)
    generator.apply(weights_init)
    discriminator = copy.deepcopy(network.Discriminator())
    discriminator = discriminator.to(device)
    discriminator.apply(weights_init)

    G_optimizer = torch.optim.Adam(generator.parameters(), lr=lr1, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr2, betas=(0.5, 0.999))
    D_loss_list1, D_loss_list2, G_loss_list = [], [], []

    print("\nStarting training...")
    for epoch in range(epochs):
        for i, data in enumerate(train_data):
            real_img = data
            real_img = real_img.to(device)

            from torch.autograd import Variable
            noise = Variable(torch.randn(real_img.size(0),
                            100, 1, 1, device=device), requires_grad = True)
            zero_vector = Variable(torch.zeros(real_img.size(0),
                            1, device=device), requires_grad = True)
            one_vector = Variable(torch.ones(real_img.size(0),
                            1, device=device), requires_grad = True)#*0.9
            D_optimizer.zero_grad()
            # discriminator = discriminator.train()
            # generator = generator.eval()
            fake_img = generator(noise)
            # fake_img.requires_grad = True
            # print(fake_img.requires_grad)

            fake_output = discriminator(fake_img.detach())  # not train generator
            D_fake_loss = discriminator_loss(fake_output, zero_vector)
            D_fake_loss.backward()

            real_output = discriminator(real_img)
            D_real_loss = discriminator_loss(real_output, one_vector)
            D_real_loss.backward()

            D_loss = D_real_loss.item() + D_fake_loss.item()
            D_loss_list1.append(D_fake_loss.item())
            D_loss_list2.append(D_real_loss.item())
            D_optimizer.step()

            # generator = generator.train()
            # discriminator = discriminator.eval()
            G_optimizer.zero_grad()
            gan_output = discriminator(fake_img)
            G_loss = generator_loss(gan_output, one_vector)
            G_loss_list.append(G_loss.item())
            G_loss.backward()
            G_optimizer.step()
            # print(G_loss)
            # print(D_loss)

            if (i + 1) % per == 0:
                print("[epoch:%d, batch:%5d, lr: %.9f] discriminator_loss: %.13f" %
                      (epoch + 1, i + 1, lr1, D_fake_loss))  #sum(D_loss_list[epoch*(i-per+1) : epoch*(i+1)]) / float(per)
                print("[epoch:%d, batch:%5d, lr: %.9f] generator_loss: %.13f" %
                      (epoch + 1, i + 1, lr2, G_loss.item()))#sum(G_loss_list[epoch*(i-per+1) : epoch*(i+1)]) / float(per)
                print("[epoch:%d, batch:%5d, lr: %.9f] discriminator_loss_total: %.13f" %
                      (epoch + 1, i + 1, lr1, D_loss))

                plt.figure()
                plt.title('Discriminator Loss')
                plt.xlabel('Per '+ str(per)+ ' Iterations')
                plt.ylabel('Loss')
                plt.plot(D_loss_list1, label='Discriminator Loss')
                plt.legend(loc='upper right')
                plt.show()
                plt.savefig(save_path + "Discriminator_loss9_1" + ".jpg")

                plt.figure()
                plt.title('Discriminator Loss')
                plt.xlabel('Per ' + str(per) + ' Iterations')
                plt.ylabel('Loss')
                plt.plot(D_loss_list2, label='Discriminator Loss')
                plt.legend(loc='upper right')
                plt.show()
                plt.savefig(save_path + "Discriminator_loss9_2" + ".jpg")

                plt.figure()
                plt.title('Generator Loss')
                plt.xlabel('Per '+ str(per)+ ' Iterations')
                plt.ylabel('Loss')
                plt.plot(G_loss_list, label='Generator Loss')
                plt.legend(loc='upper right')
                plt.show()
                plt.savefig(save_path + "Generator_loss9" + ".jpg")
        torch.save(generator.state_dict(), save_path + 'generator9_'+str(epoch)+'.pth')
        torch.save(discriminator.state_dict(), save_path + 'discriminator9_'+str(epoch)+'.pth')
    return D_loss_list1, D_loss_list2, G_loss_list

if __name__ == '__main__':
    device  = torch.device('cuda:1')
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_path1 = "/home/yangbj/695/hw7/Data/Train"
    trainset_path2 = "/home/yangbj/695/hw7/Data/Test"
    save_path = "/home/yangbj/695/hw7/"
    running_loss1, running_loss2, running_loss3 = train(
        transform = transform, device = device, lr1 = 2e-4, lr2 = 2e-4,
          epochs = 10, batch_size = 32, data_path1 = trainset_path1,
        data_path2 = trainset_path2, per = 50,
          save_path = save_path)