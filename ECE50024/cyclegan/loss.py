import torch
import torch.nn as nn
from torch.autograd import Variable

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label).to(input.device)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label).to(input.device)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def _gram_mat(x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

def style_loss(gen,style):
    #Calculating the gram matrix for the style and the generated image
    # batch_size, channel, height, width = gen.shape
    G = _gram_mat(gen)
    A = _gram_mat(style)
    # G = torch.einsum('bij, bjk -> bik', gen.view(batch_size, channel, height*width), torch.transpose(gen.view(batch_size, channel, height*width), 1, 2))
    # A = torch.einsum('bij, bjk -> bik', style.view(batch_size, channel, height*width), torch.transpose(style.view(batch_size, channel, height*width), 1, 2))
        
    #Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
    style_l = torch.nn.functional.l1_loss(G, A)
    return style_l


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
            Default: False.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 norm_img=False,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5

        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(
                        x_features[k] - gt_features[k],
                        p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(
                        x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) -
                        self._gram_mat(gt_features[k]),
                        p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(
                        self._gram_mat(x_features[k]),
                        self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


from collections import OrderedDict
from torchvision.models import vgg as vgg
from torchvision.models import alexnet as alexnet
from torchvision.models import mobilenet_v3_small as mobilenet_v3_small

NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
        'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4',
        'pool5'
    ],
    'alexnet': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1',
        'relu2_1', 'pool2', 'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3'
    ],
    'mobilenet_v3_small': [
        'conv0', 'IR1', 'IR2', 'IR3',
        'IR4', 'IR5', 'IR6', 'IR7',
        'IR8', 'IR9', 'IR10', 'IR11', 'conv12'
    ],
}


def insert_bn(names):
    """Insert bn layer after each conv.
    Args:
        names (list): The list of layer names.
    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.
    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.
    Args:
        pretrained: whether to load pretrained model from offical torchvision
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2,
                 pretrained=True):
        super(VGGFeatureExtractor, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx
        if vgg_type == "alexnet":
            features = alexnet(pretrained=pretrained).features[:max_idx + 1]
        elif vgg_type == "mobilenet_v3_small":
            features = mobilenet_v3_small(pretrained=pretrained).features[:max_idx + 1]
        else:
            features = getattr(vgg,
                            vgg_type)(pretrained=pretrained).features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(
                        kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output