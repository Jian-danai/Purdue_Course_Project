import torch
import numpy as np
import cv2

EPS = 1e-9
PI = 22.0 / 7.0


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def to_tensor(img):
    '''convert numpy.ndarray to torch tensor. \n
        if the image is uint8 , it will be divided by 255;\n
        if the image is uint16 , it will be divided by 65535;\n
        if the image is float , it will not be divided, we suppose your image range should between [0~1] ;\n
    
    Arguments:
        img {numpy.ndarray} -- image to be converted to tensor.
    '''
    if not _is_numpy_image(img):
        raise TypeError('data should be numpy ndarray. but got {}'.format(type(img)))

    if img.ndim == 2:
        img = img[:, :, None]

    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255
    elif img.dtype == np.uint16:
        img = img.astype(np.float32)/65535
    elif img.dtype in [np.float32, np.float64]:
        img = img.astype(np.float32)/1
    else:
        raise TypeError('{} is not support'.format(img.dtype))
    
    img = torch.from_numpy(img.transpose((2, 0, 1)))

    return img

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.
    After clamping to [min, max], values will be normalized to [0, 1].
    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.
    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        # if n_dim == 4:
        #     img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
        #     img_np = img_np.transpose(1, 2, 0)
        #     if rgb2bgr:
        #         img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
