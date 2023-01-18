import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        layers = []
        latent_dim = 100
        curr_dim = 512
        layers.append(nn.ConvTranspose2d(latent_dim, curr_dim,
               kernel_size=4, stride=1, padding=0, bias=False)) # B*100*1*1 -> B*512*4*4
        layers.append(nn.InstanceNorm2d(curr_dim, affine=True))
        # layers.append(nn.BatchNorm2d(curr_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        for i in range(3):  # B*512*4*4->B*512//8*32*32
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2,
                    kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            # layers.append(nn.BatchNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers.append(nn.ConvTranspose2d(curr_dim, 3, kernel_size=4,
                stride=2, padding=1, bias=False))  # B*512//8*32*32 -> B*3*64*64
        # layers.append(nn.InstanceNorm2d(3, affine=True))
        # layers.append(nn.BatchNorm2d(3, affine=True))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


from torch.nn.utils import spectral_norm
class Discriminator(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=3):
        super(Discriminator, self).__init__()
        layers = []
        # layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4,
        # stride=2, padding=1))) # B*3*64*64 -> B*64*32*32
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2,
                        padding=1, bias=False))
        # layers.append(nn.BatchNorm2d(conv_dim, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        curr_dim = conv_dim
        for _ in range(repeat_num): #B*64*32*32 -> B*(64*2^3)*4*4
            # layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2,
            # kernel_size=4, stride=2, padding=1, bias=False)))
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4,
                            stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_dim = curr_dim * 2

        # layers.append(SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=3,
        # stride=1, padding=1, bias=False))) #B*(64*2^5)*1*1 -> B*1*1*1
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1,
                            padding=0, bias=False))
        # layers.append(nn.BatchNorm2d(1, affine=True))
        layers.append(nn.Sigmoid())
        layers.append(nn.Flatten())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x
