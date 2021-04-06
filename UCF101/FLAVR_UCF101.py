import os
import sys
sys.path.append('.')
import cv2
import torch
import numpy as np
from model.FLAVR_arch import UNet_3D_3D
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# import time

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_3D_3D(block='unet_18', n_inputs=4, n_outputs=1, joinType='concat')
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load('../pretrained_model/FLAVR_2x.pth')["state_dict"], strict=True)
    model.eval()
    model.to(device)

    path = '../../data/UCF101/ucf101_interp_ours/'
    dirs = os.listdir(path)

    psnr_list = []
    ssim_list = []
    time_list = []
    # print('=========>Start Calculate PSNR and SSIM')
    for d in tqdm(dirs):
        img0 = (path + d + '/frame_00.png')
        img1 = (path + d + '/frame_02.png')
        gt = (path + d + '/frame_01_gt.png')
        img0 = (torch.tensor(cv2.imread(img0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
        img1 = (torch.tensor(cv2.imread(img1).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
        gt = (torch.tensor(cv2.imread(gt).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)

        # inference
        pred = model([img0, img0, img1, img1])[0][0]
        pred = torch.clamp(pred, 0, 1)
        # Calculate indicators
        out = pred.detach().cpu().numpy().transpose(1, 2, 0)
        out = np.round(out * 255) / 255.
        gt = gt[0].cpu().numpy().transpose(1, 2, 0)
        psnr = compute_psnr(gt, out)
        ssim = compute_ssim(gt, out)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    # print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
    # print('=========>Start Calculate Inference Time')

    # inference time
    for i in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pred = model([img0, img0, img1, img1])[0][0]
        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end))
    time_list.remove(min(time_list))
    time_list.remove(max(time_list))
    print("Avg PSNR: {} SSIM: {} Time: {}".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list) / 98))


def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p

def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s

if __name__ =='__main__':
    main()