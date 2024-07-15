import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp as backwarp
from .softsplat import softsplat
from .utils import *


# for random sample ablation
def random_sample(feature, num_points=256):
    rand_ind = torch.randint(low=0, high=feature.shape[1], size=(feature.shape[0], num_points)).unsqueeze(-1).to(
        feature.device)
    kp = torch.gather(feature, dim=1, index=rand_ind.expand(-1, -1, feature.shape[2]))
    return rand_ind, kp

def sample_key_points(importance_map, feature, num_points=256):
    importance_map = importance_map.view(-1, 1, importance_map.shape[2] * importance_map.shape[3]).permute(0, 2, 1)
    _, kp_ind = torch.topk(importance_map, num_points, dim=1)
    kp = torch.gather(feature, dim=1, index=kp_ind.expand(-1, -1, feature.shape[2]))
    return kp_ind, kp


def forward_warp(tenIn, tenFlow, z=None):
    if z is None:
        z = torch.ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]).to(tenIn.device)
    else:
        z = torch.where(z == 0, -20, 1)
    out = softsplat(tenIn, tenFlow, tenMetric=z, strMode='soft')
    return out


def warp_twice(imgA, target, flow_tA, flow_tB):
    It_warp = backwarp(imgA, flow_tA)  # backward warp(I1,Ft1)
    z = torch.ones([imgA.shape[0], 1, imgA.shape[2], imgA.shape[3]]).to(imgA.device)
    IB_warp = softsplat(tenIn=It_warp, tenFlow=flow_tB, tenMetric=z, strMode='soft')
    return IB_warp


def build_map(imgA, imgB, flow_tA, flow_tB):
    # build map for img B
    IB_warp = warp_twice(imgA, imgB, flow_tA, flow_tB)
    difference_map = IB_warp - imgB  # [B, 3, H, W], difference map on IB
    difference_map = torch.sum(torch.abs(difference_map), dim=1, keepdim=True)  # B, 1, H, W
    return difference_map


def build_hole_mask(img_template, flow_tA, flow_tB):
    # build hole mask
    with torch.no_grad():
        ones = torch.ones(img_template.shape[0], 1, img_template.shape[2], img_template.shape[3]).to(
            img_template.device)
        out = warp_twice(ones, ones, flow_tA, flow_tB)
        hole_mask = torch.where(out == 0, 0, 1)
    return hole_mask


def gen_importance_map(img0, img1, flow):
    I1_dmap = build_map(img0, img1, flow[:, 0:2], flow[:, 2:4])
    I0_dmap = build_map(img1, img0, flow[:, 2:4], flow[:, 0:2])

    I1_hole_mask = build_hole_mask(img0, flow[:, 0:2], flow[:, 2:4])
    I0_hole_mask = build_hole_mask(img1, flow[:, 2:4], flow[:, 0:2])

    I1_dmap = I1_dmap * I1_hole_mask
    I0_dmap = I0_dmap * I0_hole_mask

    I0_prob = warp_twice(I1_dmap, I1_dmap, flow[:, 2:4], flow[:, 0:2])
    I1_prob = warp_twice(I0_dmap, I0_dmap, flow[:, 0:2], flow[:, 2:4])

    importance_map = torch.cat([I0_prob, I1_prob], dim=0)  # 2B, 1, H, W
    return importance_map


def global_matching(key_feature, global_feature, key_index, H, W):
    b, n, c = global_feature.shape
    query = key_feature
    key = global_feature
    correlation = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, k, H*W]

    prob = F.softmax(correlation, dim=-1)
    init_grid = coords_grid(b, H, W, homogeneous=False, device=global_feature.device)
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]
    out = torch.matmul(prob, grid)  # B, k, 2
    if key_index is not None:
        flow_fix = torch.zeros_like(grid)
        # key_index: [B, K, 1], out: [B, K, 2], flow_fix: [B, H*W, 2]
        flow_fix = torch.scatter(flow_fix, dim=1, index=key_index.expand(-1, -1, 2), src=out)
        flow_fix = flow_fix.view(b, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # for grid, points in grid and not in key_index, set to 0
        grid_new = torch.zeros_like(grid)
        key_pos = torch.ones_like(out)
        grid_new = torch.scatter(grid_new, dim=1, index=key_index.expand(-1, -1, 2), src=key_pos)
        grid = (grid * grid_new).reshape(b, H, W, 2).permute(0, 3, 1, 2)
        flow_fix = flow_fix - grid
    else:
        flow_fix = out.view(b, H, W, 2).permute(0, 3, 1, 2)
        flow_fix = flow_fix - init_grid
    return flow_fix, prob


def extract_topk(foo, k):
    b, _, h, w = foo.shape
    foo = foo.view(b, 1, h * w).permute(0, 2, 1)
    kp, kp_ind = torch.topk(foo, k, dim=1)
    grid = torch.zeros(b, h * w, 1).to(foo.device)
    out = torch.scatter(grid, dim=1, index=kp_ind, src=kp)
    out = out.permute(0, 2, 1).reshape(b, 1, h, w)
    return out


def flow_shift(flow_fix, timestep, num_key_points=None, select_topk=False):
    B = flow_fix.shape[0] // 2
    z = torch.where(flow_fix == 0, 0, 1).detach().sum(1, keepdim=True) / 2
    zt0, zt1 = z[B:], z[:B]
    flow_fix_t0 = forward_warp(flow_fix[B:] * timestep, flow_fix[B:] * (1 - timestep), z=zt0)
    flow_fix_t1 = forward_warp(flow_fix[:B] * (1 - timestep), flow_fix[:B] * timestep, z=zt1)
    flow_fix_t = torch.cat([flow_fix_t0, flow_fix_t1], 0)
    if select_topk and num_key_points != -1:
        warp_map_t0 = softsplat(zt0, flow_fix[B:] * (1 - timestep), None, 'sum')
        warp_map_t1 = softsplat(zt1, flow_fix[:B] * timestep, None, 'sum')

        warp_map = torch.cat([warp_map_t0, warp_map_t1], 0)
        warp_map_topk = extract_topk(warp_map, num_key_points)
        warp_map_topk = torch.where(warp_map_topk != 0, 1, 0)
        flow_fix_t = flow_fix_t * warp_map_topk
    return flow_fix_t


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes=64, out_planes=64, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=True),
        nn.PReLU(out_planes)
    )

class FlowRefine(nn.Module):
    def __init__(self, in_planes, scale=4, c=64, n_layers=8):
        super(FlowRefine, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c, 3, 1, 1),
            conv(c, c, 3, 1, 1),
        )
        self.convblock = nn.Sequential(
            *[conv(c, c) for _ in range(n_layers)]
        )
        self.lastconv = conv(c, 5)
        self.scale = scale

    def forward(self, x, flow_s, flow):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / self.scale, mode="bilinear",
                                 align_corners=False) * 1. / self.scale
            x = torch.cat((x, flow), 1)
        if flow_s is not None:
            x = torch.cat((x, flow_s), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        x = self.lastconv(x)
        tmp = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * self.scale
        mask = tmp[:, 4:5]
        return flow, mask


class MergingBlock(nn.Module):
    def __init__(self, radius=3, input_dim=256, hidden_dim=256):
        super(MergingBlock, self).__init__()
        self.r = radius
        self.rf = radius ** 2
        self.conv = nn.Sequential(nn.Conv2d(8 + 2*input_dim, hidden_dim, 3, 1, 1),
                                       nn.PReLU(hidden_dim),
                                       nn.Conv2d(hidden_dim, 2*2*self.rf, 1, 1, 0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(0.1 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, feature, init_flow, flow_fix):
        """
        :param feature: [B, C, H, W] -> (local feature) or (local feature + matching feature)
        :param init_flow: [B, 2, H, W] -> (local init flow)
        :param flow_fix: [B, 2, H, W] -> (matching output, flow_fix (after patching, no hollows))
        """
        b, flow_channel, h, w = init_flow.shape
        concat = torch.cat((init_flow, flow_fix, feature), dim=1)
        mask = self.conv(concat)
        assert init_flow.shape == flow_fix.shape, f"different flow shape not implemented yet"
        mask = mask.view(b, 1, 2 * 2 * self.rf, h, w)
        mask0 = mask[:, :, :2 * self.rf, :, :]
        mask1 = mask[:, :, 2 * self.rf:, :, :]
        mask = torch.cat([mask0, mask1], dim=0)
        mask = torch.softmax(mask, dim=2)

        init_flow_all = torch.cat([init_flow[:, 0:2], init_flow[:, 2:4]], dim=0)
        flow_fix_all = torch.cat([flow_fix[:, 0:2], flow_fix[:, 2:4]], dim=0)

        init_flow_grid = F.unfold(init_flow_all, [self.r, self.r], padding=self.r//2)
        init_flow_grid = init_flow_grid.view(2*b, 2, self.rf, h, w)  # [B, 2, 9, H, W]
        flow_fix_grid = F.unfold(flow_fix_all, [self.r, self.r], padding=self.r//2)
        flow_fix_grid = flow_fix_grid.view(2*b, 2, self.rf, h, w)  # [B, 2, 9, H, W]

        flow_grid = torch.cat([init_flow_grid, flow_fix_grid], dim=2)  # [B, 2, 2*9, H, W]

        merge_flow = torch.sum(mask * flow_grid, dim=2)  # [B, 2, H, W]
        return merge_flow


class MatchingBlock(nn.Module):
    def __init__(self, scale, c, dim, num_layers=2, gm=True):
        super(MatchingBlock, self).__init__()
        self.gm = gm
        self.dim = dim
        self.scale = scale
        self.merge = MergingBlock(radius=3, input_dim=dim+128, hidden_dim=256)
        self.refine_block = FlowRefine(27, scale, c, num_layers)

    def forward(self, img0, img1, x, main_x, init_flow, init_flow_s, init_mask,
                warped_img0, warped_img1, num_key_points, scale_factor, timestep=0.5):
        result_dict = {}

        _, c, h, w = x.shape
        B = main_x.shape[0] // 2
        # NOTE:
        #  1. we stop sparse selecting points when the image resolution
        #     becomes too small (1/8 feature map resolution <= 32, i.e., h <= 256)
        #     (see `random_rescale` in train_x4k.py)
        #  2. This limitation should be deleted when evaluating on low-resolution images (<=256x256)
        if num_key_points != -1 and h > 32:
            num_key_points = int(num_key_points * (h * w))
        else:
            num_key_points = -1  # -1 stands for global matching

        feature = x.permute(0, 2, 3, 1).reshape(2 * B, h*w, c)
        feature_reverse = torch.cat([feature[B:], feature[:B]], 0)
        
        if num_key_points == -1:
            flow_fix_norm, _ = global_matching(feature, feature_reverse, None, h, w)
        else:
            imap = gen_importance_map(img0, img1, init_flow)
            imap_s = F.interpolate(imap, size=(h, w), mode="bilinear", align_corners=False)
            kp_ind, kp_feature = sample_key_points(imap_s, feature, num_key_points)
            flow_fix_norm, _ = global_matching(kp_feature, feature_reverse, kp_ind, h, w)

        flow_fix = flow_shift(flow_fix_norm, timestep, num_key_points, select_topk=True)
        flow_fix = torch.cat([flow_fix[:B], flow_fix[B:]], 1)
        flow_r = torch.where(flow_fix == 0, init_flow_s, flow_fix)
        flow_merge = self.merge(torch.cat([x[:B], x[B:], main_x[:B], main_x[B:]], dim=1), init_flow_s, flow_r)
        flow_merge = torch.cat([flow_merge[:B], flow_merge[B:]], dim=1)
        img0_s = F.interpolate(img0, scale_factor=1 / scale_factor, mode="bilinear", align_corners=False)
        img1_s = F.interpolate(img1, scale_factor=1 / scale_factor, mode="bilinear", align_corners=False)
        warped_img0_fine_s_m = backwarp(img0_s, flow_merge[:, 0:2])
        warped_img1_fine_s_m = backwarp(img1_s, flow_merge[:, 2:4])

        flow_t, mask_t = self.refine_block(torch.cat((img0, img1, warped_img0, warped_img1, init_mask), 1),
                                           torch.cat([warped_img0_fine_s_m, warped_img1_fine_s_m, flow_merge], 1),
                                           init_flow)

        result_dict.update({'flow_t': flow_t})
        result_dict.update({'mask_t': mask_t})
        return result_dict
