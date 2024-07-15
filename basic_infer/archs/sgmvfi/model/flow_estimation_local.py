import torch
import torch.nn as nn
import torch.nn.functional as F

from .refine import *

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, layers=4, scale=4, in_else=17):
        super(IFBlock, self).__init__()
        self.scale = scale

        self.conv0 = nn.Sequential(
            conv(in_planes + in_else, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
        )

        self.convblock = nn.Sequential(
            *[conv(c, c) for _ in range(layers)]
        )

        self.lastconv = conv(c, 5)

    def forward(self, x, flow=None, feature=None):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / self.scale, mode="bilinear",
                                 align_corners=False) * 1. / self.scale
            x = torch.cat((x, flow), 1)
        if feature != None:
            x = torch.cat((x, feature), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        flow_s = tmp[:, :4]
        tmp = F.interpolate(tmp, scale_factor=self.scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * self.scale
        mask = tmp[:, 4:5]
        return flow, mask, flow_s


class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.feature_bone = backbone
        self.scale = [1, 2, 4, 8]
        self.num_key_points = [None]
        self.block = nn.ModuleList(
            [IFBlock(kargs['embed_dims'][-1] * 2, 128, 2, self.scale[-1], in_else=7),  # 1/8
             IFBlock(kargs['embed_dims'][-2] * 2, 128, 2, self.scale[-2], in_else=18)])  # 1/4
        self.contextnet = Contextnet(kargs['c'] * 2)
        self.unet = Unet(kargs['c'] * 2)

    def calculate_flow(self, imgs, timestep):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        flow_s = None

        af = self.feature_bone(img0, img1)

        for i in range(2):
            t = (img0[:B, :1].clone() * 0 + 1) * timestep
            af0 = af[-1 - i][:B]
            af1 = af[-1 - i][B:]
            if flow != None:
                flow_d, mask_d, flow_s_d = self.block[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, t), 1),
                    flow,
                    torch.cat([af0, af1], 1),
                )
                flow = flow + flow_d
                mask = mask + mask_d
                flow_s = F.interpolate(flow_s, scale_factor=2, mode="bilinear", align_corners=False) * 2
                flow_s = flow_s + flow_s_d

            else:
                flow, mask, flow_s = self.block[i](
                    torch.cat((img0, img1, t), 1),
                    None,
                    torch.cat([af0, af1], 1))
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        return flow, mask

    def coraseWarp_and_Refine(self, imgs, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        mask_ = torch.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        pred = torch.clamp(merged + res, 0, 1)
        return pred

    def forward(self, x, timestep=0.5):
        img0, img1 = x[:, :3], x[:, 3:6]
        B = x.size(0)
        flow_list, mask_list = [], []
        merged, merged_fine = [], []
        warped_img0, warped_img1 = img0, img1
        flow, mask, flow_s = None, None, None
        af = self.feature_bone(img0, img1)

        for i in range(2):
            af0 = af[-1 - i][:B]
            af1 = af[-1 - i][B:]
            t = (img0[:B, :1].clone() * 0 + 1) * timestep
            if flow != None:
                flow_d, mask_d, flow_s_d = self.block[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, t), 1),
                    flow,
                    torch.cat([af0, af1], 1),
                )
                flow = flow + flow_d
                mask = mask + mask_d
                flow_s = F.interpolate(flow_s, scale_factor=2, mode="bilinear", align_corners=False) * 2
                flow_s = flow_s + flow_s_d
            else:
                flow, mask, flow_s = self.block[i](
                    torch.cat((img0, img1, t), 1),
                    None,
                    torch.cat([af0, af1], 1))
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        pred = torch.clamp(merged[-1] + res, 0, 1)
        merged.extend(merged_fine)
        return flow_list, mask_list, merged, pred
