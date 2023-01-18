import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init

class MODL(nn.Module):
    '''
    Multi object detection and localization network
    It contains a deep feature extraction stream with a multi-scale style,
    it is consisted of 3 CNNs, 6 Residual Blocks,
    and 3 skip connections in the same scale;
    and 2 final FC layers;
    a ReLU is utilized at the end of the regression stream
    for non-negative output.
    '''
    def __init__(self):
        super(MODL, self).__init__()
        # main stream
        fc_num = 4 * 4 * 512
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                               out_channels=32,
                                               kernel_size=3,
                                               stride=2, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.2, True))
        self.conv1_2 = nn.Sequential(ResBlock(num_feat=32),
                                   ResBlock(num_feat=32))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=32,
                                               out_channels=64,
                                               kernel_size=3,
                                               stride=2, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(0.2, True))
        self.conv2_2 = nn.Sequential(ResBlock(num_feat=64),
                                     ResBlock(num_feat=64))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channels=64,
                                               out_channels=128,
                                               kernel_size=3,
                                               stride=2, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU(0.2, True))
        self.conv3_2 = nn.Sequential(ResBlock(num_feat=128),
                                     ResBlock(num_feat=128),
                                     )
        self.conv4_1 = nn.Sequential(nn.Conv2d(in_channels=128,
                                               out_channels=256,
                                               kernel_size=3,
                                               stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, True))
        self.conv4_2 = nn.Sequential(ResBlock(num_feat=256),
                                     ResBlock(num_feat=256),
                                     )
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=256,
                                               out_channels=512,
                                               kernel_size=3,
                                               stride=2, padding=1),
                                    nn.LeakyReLU(0.2, True))

        # fc stream
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(fc_num, 4096),
                                nn.LeakyReLU(0.2, True),
                                nn.Linear(4096, 2048),
                                nn.LeakyReLU(0.2, True),
                                nn.Linear(2048, 1440 + 5*6*6))


    def forward(self, x):
        x = self.conv1_1(x)
        x = x + self.conv1_2(x)
        x = self.conv2_1(x)
        x = x + self.conv2_2(x)
        x = self.conv3_1(x)
        x = x + self.conv3_2(x)
        x = self.conv4_1(x)
        x = x + self.conv4_2(x)
        x = self.conv5(x)
        x = self.fc(x)

        return x

class ResBlock(nn.Module):
    """Residual block with BN.

    It has a style of:
        ---Conv-BN-ReLU-Conv-BN+-ReLU
         |_____________________|

    """

    def __init__(self, num_feat=64, res_scale=1, dilation=1):
        super().__init__()
        self.res_scale = res_scale
        padding = get_valid_padding(3, dilation)

        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, bias=True, \
                            padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(num_feat)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, bias=True, \
                            padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(num_feat)
        self.act = nn.LeakyReLU(0.2, True)

        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))
        return identity + out * self.res_scale

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights with kaiming_normal.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def get_valid_padding(kernel_size, dilation):
    # get valid padding number
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding
