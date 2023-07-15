from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import torch
from collections import OrderedDict
import cv2
import glob
import math
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import _LRScheduler
#from model_summary import get_model_activation, get_model_flops

def AdjustLearningRate(optimizer, lr):
    for param_group in optimizer.param_groups:
        print('param_group',param_group['lr'])
        param_group['lr'] = lr

def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s


def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


def modcrop(im, modulo):
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * 65535.0).round()

    return img_np.astype(out_type)


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Cyclic learning rate
def cycle(iteration, stepsize):
    return math.floor(1 + iteration / (2 * stepsize))

def abs_pos(cycle_num, iteration, stepsize):
    return abs(iteration / stepsize - 2 * cycle_num + 1)

def rel_pos(iteration, stepsize):
    return max(0, (1-abs_pos(cycle(iteration, stepsize), iteration, stepsize)))

def cyclic_learning_rate(min_lr, max_lr, stepsize):
    return lambda iteration: min_lr + (max_lr - min_lr) * rel_pos(iteration, stepsize)

class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


def load_state_dict(path):
    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit

def load_network(load_path, network, strict=True):
    if os.path.isfile(load_path):
        print("===> loading models '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        for k, _ in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        network.load_state_dict(pretrained_dict, strict=strict)
    else:
        print("===> no models found at '{}'".format(load_path))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


## only for reading LR (np.uint8 images)
def read_img(filename):
    ## read image by cv2, return HWC, BGR, [0, 1]
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = np.float32(img) / 255.0
    return img


def read_seq_imgs(img_seq_path):
    '''read a sequence of images

    Returns:
        imgs (Tensor):size (T, C, H, W), RGB, [0, 1]
    '''
    img_path_l = sorted(glob.glob(img_seq_path + '/*.png'))
    img_l = [read_img(v) for v in img_path_l]
    # stack to TCHW
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]  # BGR to RGB
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs

def index_generation(crt_i, max_n, N, padding='new_info'):
    '''
    padding: replicate | reflection | new_info | circle
    '''
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
        # Important. Unlike matlab, numpy.unit16() WILL NOT round by default.
    elif out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def single_forward(model, x, shave=32):
    b, t, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2

    h_size, w_size = h_half + shave - (h_half + shave) % 16, w_half + shave - (w_half + shave) % 16
    inputlist = [
        x[:, :, :, 0:h_size, 0:w_size],
        x[:, :, :, 0:h_size, (w - w_size):w],
        x[:, :, :, (h - h_size):h, 0:w_size],
        x[:, :, :, (h - h_size):h, (w - w_size):w]]

    outputlist = []

    with torch.no_grad():
        for i in range(0, 4, 1):
            input_batch = torch.cat(inputlist[i:i+1], dim=0)
            output_batch = model(input_batch)

            outputlist.extend(output_batch.chunk(1, dim=0))

        output = torch.zeros(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] \
            = outputlist[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def model_summary(model):
    # input_dim = (3, 240, 360)  # set the input dimension
    input_dim = (5, 3, 480, 270)
    # device = torch.device('cuda')
    # model = model().to(device)

    activations, num_conv2d = get_model_activation(model, input_dim)
    print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations / 10 ** 6))
    print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

    flops = get_model_flops(model, input_dim, False)
    print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops / 10 ** 9))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters / 10 ** 6))

def pixel_reverse(image):
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for c in range(image.shape[2]):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv  # 相当于取反 例如白的变成黑的，黑的变成白的，
