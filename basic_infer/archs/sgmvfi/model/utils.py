import os
import torch
import torch.nn.functional as F
from .position import PositionEmbeddingSine
from .geometry import coords_grid, generate_window_grid, normalize_coords
import numpy as np
import cv2


def show(foo):
    import matplotlib.pyplot as plt
    if len(foo.shape) == 4 and foo.shape[1] in [1, 3]:
        tmp = foo[0].permute(1, 2, 0)
    elif len(foo.shape) == 3 and foo.shape[0] in [1, 3]:
        tmp = foo.permute(1, 2, 0)
    elif len(foo.shape) == 3 and foo.shape[0] in [2, 4]:
        print(f'showing flow: {foo.shape}')
        return show_flow(foo)
    elif len(foo.shape) == 2:
        tmp = foo
    else:
        assert 0, f'input flow shape: {foo.shape}'
    tmp = tmp.cpu().detach().numpy()
    plt.figure()
    plt.imshow(tmp)
    plt.colorbar()
    plt.show()

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def show_flow(flow):
    import matplotlib.pyplot as plt
    if flow.shape[0] == 2:
        flow = flow.permute(1, 2, 0).cpu().detach().numpy()
        map = flow2rgb(flow)
    elif flow.shape[0] == 4:
        flow01 = flow[0:2].permute(1, 2, 0).cpu().detach().numpy()
        flow10 = flow[2:4].permute(1, 2, 0).cpu().detach().numpy()
        map01 = flow2rgb(flow01)
        map10 = flow2rgb(flow10)
        map = np.concatenate([map01, map10], axis=1)
    else:
        assert 0, f'input flow shape: {flow.shape}'
    plt.figure()
    plt.imshow(map)
    plt.colorbar()
    plt.show()


def split_feature(feature,
                  num_splits=2,
                  channel_last=False,
                  ):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                               ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

    return feature

def merge_splits(splits,
                 num_splits=2,
                 channel_last=False,
                 ):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

    return merge


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', padding_factor=8, additional_pad=False):
        self.ht, self.wd = dims[-2:]
        add_pad = padding_factor*2 if additional_pad else 0
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor + add_pad
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor + add_pad
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
