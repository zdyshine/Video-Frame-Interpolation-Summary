import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.WaveletVFI.utils import bwarp


def centralize(img0, img1):
    rgb_mean = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
    return img0 - rgb_mean, img1 - rgb_mean, rgb_mean


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.LeakyReLU(negative_slope=0.1)
    )


class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, is_bottom=False):
        super(Decoder, self).__init__()
        self.is_bottom = is_bottom
        self.conv1 = convrelu(in_channels, mid_channels, 3, 1)
        self.conv2 = convrelu(mid_channels, mid_channels, 3, 1)
        self.conv3 = convrelu(mid_channels, mid_channels, 3, 1)
        self.conv4 = convrelu(mid_channels, mid_channels, 3, 1)
        self.conv5 = convrelu(mid_channels, mid_channels, 3, 1)
        self.conv6 = nn.ConvTranspose2d(mid_channels, 5, 4, 2, 1, bias=True)
        if self.is_bottom:
            self.classifier = nn.Sequential(
                convrelu(mid_channels, mid_channels, 3, 2), 
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Flatten(1), 
                nn.Linear(mid_channels, mid_channels, bias=True), 
                nn.LeakyReLU(negative_slope=0.1), 
                nn.Linear(mid_channels, 4, bias=True)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out = self.conv6(out5)
        if self.is_bottom:
            class_prob_ = self.classifier(out5)
            return out, class_prob_
        else:
            return out


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.pconv1 = nn.Sequential(convrelu(3, 32, 3, 2), convrelu(32, 32, 3, 1))
        self.pconv2 = nn.Sequential(convrelu(32, 64, 3, 2), convrelu(64, 64, 3, 1))
        self.pconv3 = nn.Sequential(convrelu(64, 96, 3, 2), convrelu(96, 96, 3, 1))
        self.pconv4 = nn.Sequential(convrelu(96, 128, 3, 2), convrelu(128, 128, 3, 1))

        self.decoder4 = Decoder(256, 192, True)
        self.decoder3 = Decoder(197, 160, False)
        self.decoder2 = Decoder(133, 128, False)
        self.decoder1 = Decoder(69, 64, False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img0, img1):
        img0, img1, _ = centralize(img0, img1)

        f0_1 = self.pconv1(img0)
        f1_1 = self.pconv1(img1)
        f0_2 = self.pconv2(f0_1)
        f1_2 = self.pconv2(f1_1)
        f0_3 = self.pconv3(f0_2)
        f1_3 = self.pconv3(f1_2)
        f0_4 = self.pconv4(f0_3)
        f1_4 = self.pconv4(f1_3)

        out4, class_prob_ = self.decoder4(torch.cat([f0_4, f1_4], 1))
        up_flow_t0_4 = out4[:, 0:2]
        up_flow_t1_4 = out4[:, 2:4]
        up_occ_t_4 = out4[:, 4:5]

        f0_3_warp = bwarp(f0_3, up_flow_t0_4)
        f1_3_warp = bwarp(f1_3, up_flow_t1_4)
        out3 = self.decoder3(torch.cat([f0_3_warp, f1_3_warp, up_flow_t0_4, up_flow_t1_4, up_occ_t_4], 1))
        up_flow_t0_3 = out3[:, 0:2] + resize(up_flow_t0_4, 2.0) * 2.0
        up_flow_t1_3 = out3[:, 2:4] + resize(up_flow_t1_4, 2.0) * 2.0
        up_occ_t_3 = out3[:, 4:5] + resize(up_occ_t_4, 2.0)

        f0_2_warp = bwarp(f0_2, up_flow_t0_3)
        f1_2_warp = bwarp(f1_2, up_flow_t1_3)
        out2 = self.decoder2(torch.cat([f0_2_warp, f1_2_warp, up_flow_t0_3, up_flow_t1_3, up_occ_t_3], 1))
        up_flow_t0_2 = out2[:, 0:2] + resize(up_flow_t0_3, 2.0) * 2.0
        up_flow_t1_2 = out2[:, 2:4] + resize(up_flow_t1_3, 2.0) * 2.0
        up_occ_t_2 = out2[:, 4:5] + resize(up_occ_t_3, 2.0)

        f0_1_warp = bwarp(f0_1, up_flow_t0_2)
        f1_1_warp = bwarp(f1_1, up_flow_t1_2)
        out1 = self.decoder1(torch.cat([f0_1_warp, f1_1_warp, up_flow_t0_2, up_flow_t1_2, up_occ_t_2], 1))
        up_flow_t0_1 = out1[:, 0:2] + resize(up_flow_t0_2, 2.0) * 2.0
        up_flow_t1_1 = out1[:, 2:4] + resize(up_flow_t1_2, 2.0) * 2.0
        up_occ_t_1 = torch.sigmoid(out1[:, 4:5] + resize(up_occ_t_2, 2.0))

        return up_flow_t0_1, up_flow_t1_1, up_occ_t_1, class_prob_
