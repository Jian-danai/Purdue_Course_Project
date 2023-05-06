import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d,
                 use_bias=True, scale_factor=1):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=use_bias),
            norm_layer(in_channels * scale_factor),
            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=use_bias),
        )

    def forward(self, x):
        return self.conv(x)


norm_layer = nn.InstanceNorm2d
class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), norm_layer(f, affine=True), nn.ReLU(),
                                  nn.Conv2d(f, f, 3, 1, 1))
        self.norm = norm_layer(f, affine=True)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)+x))

class Generator(nn.Module):
    def __init__(self, f=64, blocks=6):
        super(Generator, self).__init__()
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(  3,   f, 7, 1, 0), norm_layer(  f, affine=True), nn.ReLU(True),
                  nn.Conv2d(  f, 2*f, 3, 2, 1), norm_layer(2*f, affine=True), nn.ReLU(True),
                  nn.Conv2d(2*f, 4*f, 3, 2, 1), norm_layer(4*f, affine=True), nn.ReLU(True)]
        for i in range(int(blocks)):
            layers.append(ResBlock(4*f))
        layers.extend([
                nn.ConvTranspose2d(4*f, 4*2*f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(2*f, affine=True), nn.ReLU(True),
                nn.ConvTranspose2d(2*f,   4*f, 3, 1, 1), nn.PixelShuffle(2), norm_layer(  f, affine=True), nn.ReLU(True),
                nn.ReflectionPad2d(3), nn.Conv2d(f, 3, 7, 1, 0),
                nn.Tanh()])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)

nc=3
ndf=64
class Discriminator(nn.Module):  
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc,ndf,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf,ndf*2,4,2,1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf*4,ndf*8,4,1,1),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 15 x 15
            nn.Conv2d(ndf*8,1,4,1,1)
            # state size. 1 x 14 x 14
        )

    def forward(self, input):
        return self.main(input)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class Joint_Generator(nn.Module):

    def __init__(self, conv_dim=64, repeat_num=9, input_nc=3):
        super(Joint_Generator, self).__init__()

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        layers_branch.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_0 = nn.Sequential(*layers_branch)

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        layers_branch.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_1 = nn.Sequential(*layers_branch)

        # Down-Sampling, branch merge
        layers = []
        curr_dim = conv_dim * 2 #128           #256
        layers.append(nn.Conv2d(curr_dim * 2, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2  #256

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers_1 = []
        layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.Tanh())
        self.branch_1 = nn.Sequential(*layers_1)
        layers_2 = []
        layers_2.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_2.append(nn.Tanh())
        self.branch_2 = nn.Sequential(*layers_2)
    def forward(self, x, y):
        input_x = self.Branch_0(x)
        input_y = self.Branch_1(y)
        input_fuse = torch.cat((input_x, input_y), dim=1)
        out = self.main(input_fuse)
        out_A = self.branch_1(out)
        out_B = self.branch_2(out)
        return out_A, out_B


from spectral_norm import spectral_norm as SpectralNorm
class Beauty_Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=3):
        super(Beauty_Discriminator, self).__init__()
        layers = []

        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for _ in range(1, repeat_num):

            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1)))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim *2

        self.main = nn.Sequential(*layers)
        self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))

    def forward(self, x):
        h = self.main(x)
        out_makeup = self.conv1(h)
        return out_makeup.squeeze()


# # FLOPs
# pre_model = RGB2uv()
# input = torch.randn(1, 3, 480, 640)
# input = pre_model(input)
# print(input.shape)
# model = Illuminant()

# print(model)
# from thop import clever_format, profile

# macs, params = profile(model, inputs=(input, ))
# # macs, params = clever_format([macs, params], "%.3f")
# print("FLOPs: ", macs/10**6)