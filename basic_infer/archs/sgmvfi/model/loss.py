import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

class Ternary(nn.Module):
    def __init__(self, device):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class Charbonnier_L1(nn.Module):
    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        else:
            loss = (((diff ** 2 + 1e-6) ** 0.5) * mask).mean() / (mask.mean() + 1e-9)
        return loss


class Charbonnier_Ada(nn.Module):
    def __init__(self):
        super(Charbonnier_Ada, self).__init__()

    def forward(self, diff, weight):
        alpha = weight / 2
        epsilon = 10 ** (-(10 * weight - 1) / 3)
        loss = ((diff ** 2 + epsilon ** 2) ** alpha).mean()
        return loss


def get_robust_weight(flow_pred, flow_gt, beta):
    epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=1, keepdim=True) ** 0.5
    robust_weight = torch.exp(-beta * epe)
    return robust_weight


def flow_loss_func(flow_preds, flow_gt, valid=None,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # for i in range(n_predictions):

    i_loss = (flow_preds - flow_gt).abs()

    flow_loss = i_loss.mean()

    # epe = torch.sum((flow_preds - flow_gt) ** 2, dim=1).sqrt()
    #
    # metrics = {
    #     'epe': epe.mean().item(),
    #     '1px': (epe > 1).float().mean().item(),
    #     '3px': (epe > 3).float().mean().item(),
    #     '5px': (epe > 5).float().mean().item(),
    # }
    metrics = None

    return flow_loss, metrics

# def flow_loss_func(flow_preds, flow_gt, valid=None,
#                    gamma=0.9,
#                    max_flow=400,
#                    **kwargs,
#                    ):
#     n_predictions = len(flow_preds)
#     flow_loss = 0.0
#
#     for i in range(n_predictions):
#         i_weight = gamma ** (n_predictions - i - 1)
#
#         i_loss = (flow_preds[i] - flow_gt).abs()
#
#         flow_loss += i_weight * i_loss.mean()
#
#     # epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
#
#     metrics = {
#         # 'epe': epe.mean().item(),
#         # '1px': (epe > 1).float().mean().item(),
#         # '3px': (epe > 3).float().mean().item(),
#         # '5px': (epe > 5).float().mean().item(),
#     }
#
#     return flow_loss, metrics