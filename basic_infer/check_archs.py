import torch
import torch.nn.functional as F
import cv2
import numpy as np


class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor = 16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

padder = InputPadder(torch.rand(1,6,1080,1920).shape, divisor=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from define_load_model import get_WaveletVFI
model = get_WaveletVFI()
model.eval()
model.to(device)
I0 = cv2.imread('images/0.png')
I1 = cv2.imread('images/1.png')
I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
I0, I1 = padder.pad(I0, I1)

mid = model.inference(I0, I1)

mid = padder.unpad(mid)

mid = np.round((mid[0] * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
cv2.imwrite('images/mid.png', mid)
