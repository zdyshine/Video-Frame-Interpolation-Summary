import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.WaveletVFI.IFNet import IFNet
from archs.WaveletVFI.utils import bwarp, gumbel_softmax
# from archs.WaveletVFI.loss import Charbonnier_L1, Ternary
from pytorch_wavelets import DWT, IDWT


class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        mask = x[:, -1:, :, :]
        x = x[:, :-1, :, :]
        y = F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) * mask
        return y


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)


def upsample(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='nearest', align_corners=None, recompute_scale_factor=True)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.LeakyReLU(negative_slope=0.1)
    )


def sparseconvrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        SparseConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.LeakyReLU(negative_slope=0.1)
    )


class WaveletEncoder(nn.Module):
    def __init__(self):
        super(WaveletEncoder, self).__init__()
        self.dwt = DWT(J=1, wave='haar', mode='reflect')

    def forward(self, img):
        img_l1, img_h1 = self.dwt(img)
        img_l2, img_h2 = self.dwt(img_l1)
        img_l3, img_h3 = self.dwt(img_l2)
        img_l4, img_h4 = self.dwt(img_l3)
        return [img_l4, img_h4[0], img_h3[0], img_h2[0], img_h1[0]]


# l1_loss = Charbonnier_L1()
# tr_loss = Ternary(7)

# def get_wavelet_loss(wavelet_pred_list, wavelet_gt_list):
#     loss = 0.0
#     for j in range(5):
#         loss += l1_loss(wavelet_pred_list[j] - wavelet_gt_list[j])
#     return loss

thresh_dict = torch.tensor([0.000, 0.005, 0.010, 0.015]).float()
cost_total = ((144+432) * 216 + 216 * 9) / 4.0 + ((288+648) * 432 + 432 * 9) / 16.0 + ((432+576) * 648 + 648 * 9) / 64.0


class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.pconv1_1 = convrelu(3, 48, 3, 2)
        self.pconv1_2 = convrelu(48, 96, 3, 2)
        self.pconv1_3 = convrelu(96, 144, 3, 2)
        self.pconv1_4 = convrelu(144, 192, 3, 2)
        self.pconv2_1 = convrelu(3+2+2+1, 48, 3, 2)
        self.pconv2_2 = convrelu(48, 96, 3, 2)
        self.pconv2_3 = convrelu(96, 144, 3, 2)
        self.pconv2_4 = convrelu(144, 192, 3, 2)

    def forward(self, img0, img1, imgt_pred_, flow_t0, flow_t1, occ_t):
        f0_1 = self.pconv1_1(img0)
        f0_2 = self.pconv1_2(f0_1)
        f0_3 = self.pconv1_3(f0_2)
        f0_4 = self.pconv1_4(f0_3)
        f1_1 = self.pconv1_1(img1)
        f1_2 = self.pconv1_2(f1_1)
        f1_3 = self.pconv1_3(f1_2)
        f1_4 = self.pconv1_4(f1_3)
        ft_1 = self.pconv2_1(torch.cat([imgt_pred_, flow_t0 / 20.0, flow_t1 / 20.0, occ_t], 1))
        ft_2 = self.pconv2_2(ft_1)
        ft_3 = self.pconv2_3(ft_2)
        ft_4 = self.pconv2_4(ft_3)

        flow_t0_1 = resize(flow_t0, scale_factor=0.5) * 0.5
        flow_t0_2 = resize(flow_t0_1, scale_factor=0.5) * 0.5
        flow_t0_3 = resize(flow_t0_2, scale_factor=0.5) * 0.5
        flow_t0_4 = resize(flow_t0_3, scale_factor=0.5) * 0.5
        flow_t1_1 = resize(flow_t1, scale_factor=0.5) * 0.5
        flow_t1_2 = resize(flow_t1_1, scale_factor=0.5) * 0.5
        flow_t1_3 = resize(flow_t1_2, scale_factor=0.5) * 0.5
        flow_t1_4 = resize(flow_t1_3, scale_factor=0.5) * 0.5
        
        c1 = torch.cat([bwarp(f0_1, flow_t0_1), bwarp(f1_1, flow_t1_1), ft_1], 1)
        c2 = torch.cat([bwarp(f0_2, flow_t0_2), bwarp(f1_2, flow_t1_2), ft_2], 1)
        c3 = torch.cat([bwarp(f0_3, flow_t0_3), bwarp(f1_3, flow_t1_3), ft_3], 1)
        c4 = torch.cat([bwarp(f0_4, flow_t0_4), bwarp(f1_4, flow_t1_4), ft_4], 1)
        return c1, c2, c3, c4


class WaveletVFI(nn.Module):
    def __init__(self):
        super(WaveletVFI, self).__init__()
        self.ifnet = IFNet()
        self.contextnet = ContextNet()
        self.waveletencoder = WaveletEncoder()
        self.conv4 = convrelu(576, 576, 3, 1)
        self.conv4_l = nn.Conv2d(576, 3*1, 3, 1, 1, bias=True)
        self.conv4_h = nn.Conv2d(576, 3*3, 3, 1, 1, bias=True)
        self.conv3 = sparseconvrelu(432+576, 648, 3, 1)
        self.conv3_h = SparseConv2d(648, 3*3, 3, 1, 1, bias=True)
        self.conv2 = sparseconvrelu(288+648, 432, 3, 1)
        self.conv2_h = SparseConv2d(432, 3*3, 3, 1, 1, bias=True)
        self.conv1 = sparseconvrelu(144+432, 216, 3, 1)
        self.conv1_h = SparseConv2d(216, 3*3, 3, 1, 1, bias=True)
        self.idwt = IDWT(wave='haar', mode='zero')
        self.dilate = nn.MaxPool2d(3, stride=1, padding=1)
        self.tau = 1.0
        
    def sparse_decode(self, w4_l, w4_h, w3_l, c1, c2, c3, f4, thresh):
        thresh3 = (w3_l.max(2, keepdim=True)[0].max(3, keepdim=True)[0] - w3_l.min(2, keepdim=True)[0].min(3, keepdim=True)[0]) * thresh
        mask3 = upsample((torch.abs(w4_h).max(2)[0] > thresh3).max(1, keepdim=True)[0].float(), 2)
        f3 = self.conv3(torch.cat([c3, upsample(f4, 2), self.dilate(mask3)], 1))
        w3_h = self.conv3_h(torch.cat([f3, mask3], 1)).view(f3.shape[0], 3, 3, f3.shape[2], f3.shape[3]) * 4.0
        w2_l = self.idwt((w3_l, [w3_h]))

        thresh2 = (w2_l.max(2, keepdim=True)[0].max(3, keepdim=True)[0] - w2_l.min(2, keepdim=True)[0].min(3, keepdim=True)[0]) * thresh
        mask2 = upsample((torch.abs(w3_h).max(2)[0] > thresh2).max(1, keepdim=True)[0].float(), 2)
        f2 = self.conv2(torch.cat([c2, upsample(f3, 2), self.dilate(mask2)], 1))
        w2_h = self.conv2_h(torch.cat([f2, mask2], 1)).view(f2.shape[0], 3, 3, f2.shape[2], f2.shape[3]) * 2.0
        w1_l = self.idwt((w2_l, [w2_h]))

        thresh1 = (w1_l.max(2, keepdim=True)[0].max(3, keepdim=True)[0] - w1_l.min(2, keepdim=True)[0].min(3, keepdim=True)[0]) * thresh
        mask1 = upsample((torch.abs(w2_h).max(2)[0] > thresh1).max(1, keepdim=True)[0].float(), 2)
        f1 = self.conv1(torch.cat([c1, upsample(f2, 2), self.dilate(mask1)], 1))
        w1_h = self.conv1_h(torch.cat([f1, mask1], 1)).view(f1.shape[0], 3, 3, f1.shape[2], f1.shape[3]) * 1.0
        w0_l = self.idwt((w1_l, [w1_h]))

        imgt_pred = torch.clamp(w0_l, 0, 1)

        return imgt_pred

    def inference(self, img0, img1, embt=0.5, thresh=None):
        flow_t0, flow_t1, occ_t, class_prob_ = self.ifnet(img0, img1)

        img0_warp = bwarp(img0, flow_t0)
        img1_warp = bwarp(img1, flow_t1)
        imgt_pred_ = occ_t * img0_warp + (1 - occ_t) * img1_warp

        c1, c2, c3, c4 = self.contextnet(img0, img1, imgt_pred_, flow_t0, flow_t1, occ_t)

        f4 = self.conv4(c4)
        w4_l = self.conv4_l(f4) * 16.0
        w4_h = self.conv4_h(f4).view(f4.shape[0], 3, 3, f4.shape[2], f4.shape[3]) * 8.0
        w3_l = self.idwt((w4_l, [w4_h]))

        self.thresh = thresh_dict.to(img0)

        if thresh == None:
            class_prob = class_prob_.softmax(dim=1)
            thresh_ = self.thresh.index_select(dim=0, index=class_prob.argmax(dim=1)).view(-1, 1, 1, 1)
        else:
            thresh_ = torch.tensor(thresh).to(c4)
            
        imgt_pred = self.sparse_decode(w4_l, w4_h, w3_l, c1, c2, c3, f4, thresh_)
        print('===========', img0.shape, img1.shape, imgt_pred.shape)
        return imgt_pred