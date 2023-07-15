import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Adapted from "Tunable U-Net implementation in PyTorch"
# https://github.com/jvanvugt/pytorch-unet

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=5,
        padding=True,
    ):
        super(UNet, self).__init__()

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding)
            )
            prev_channels = 2 ** (wf + i)
        self.midconv = nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), padding)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=3,padding=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.midconv(x), negative_slope = 0.1)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.LeakyReLU(0.1))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetUpBlock, self).__init__()

        self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            )
        self.conv_block = UNetConvBlock(in_size, out_size, padding)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]
    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat((up, crop1), 1)
        out = self.conv_block(out)
        return out


def warp(img, flow):
    _, _, H, W = img.size()
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.tensor(gridX, requires_grad=False).cuda()
    gridY = torch.tensor(gridY, requires_grad=False).cuda()
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    x = gridX.unsqueeze(0).expand_as(u).float()+u
    y = gridY.unsqueeze(0).expand_as(v).float()+v
    normx = 2*(x/W-0.5)
    normy = 2*(y/H-0.5)
    grid = torch.stack((normx,normy), dim=3)
    warped = F.grid_sample(img, grid)
    return warped

class RRIN(nn.Module):
    def __init__(self,level=3):
        super(RRIN, self).__init__()
        self.Mask = UNet(16, 2, 4)
        self.Flow_L = UNet(6, 4, 5)
        self.refine_flow = UNet(10, 4, 4)
        self.final = UNet(9, 3, 4)

    def process(self, x0, x1, t):

        x = torch.cat((x0, x1), 1)
        Flow = self.Flow_L(x) # UNet1
        Flow_0_1, Flow_1_0 = Flow[:, :2, :, :], Flow[:, 2:4, :, :]
        Flow_t_0 = -(1-t) * t * Flow_0_1 + t * t * Flow_1_0
        Flow_t_1 = (1-t) * (1-t) * Flow_0_1 - t * (1-t) * Flow_1_0
        Flow_t = torch.cat((Flow_t_0, Flow_t_1, x), 1) #　光流和输入
        Flow_t = self.refine_flow(Flow_t) # UNet1
        Flow_t_0 = Flow_t_0+Flow_t[:, :2, :, :]
        Flow_t_1 = Flow_t_1+Flow_t[:, 2:4, :, :]
        xt1 = warp(x0,Flow_t_0)
        xt2 = warp(x1,Flow_t_1)
        temp = torch.cat((Flow_t_0,Flow_t_1,x,xt1,xt2),1)
        # Mask = F.sigmoid(self.Mask(temp))
        Mask = torch.sigmoid(self.Mask(temp))
        w1, w2 = (1-t)*Mask[:,0:1,:,:], t*Mask[:,1:2,:,:]
        output = (w1*xt1+w2*xt2)/(w1+w2+1e-8)

        return output

    def forward(self, input0, input1, t=0.5):

        output = self.process(input0, input1, t)
        compose = torch.cat((input0, input1, output), 1)
        final = self.final(compose) + output
        final = final.clamp(0, 1)

        return final

if __name__ == '__main__':
    x1 = torch.rand(1, 3, 1024, 576).cuda()
    x2 = torch.rand(1, 3, 1024, 576).cuda()
    net = RRIN().cuda()
    # for i in range(100):
    out = net(x1, x2)
    print(out.shape)