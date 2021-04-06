import os
import sys
sys.path.append('.')
import cv2
import torch
import numpy as np
from networks import EDSC
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# import time
from torchvision import transforms
transform = transforms.ToTensor()
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EDSC.Network(isMultiple=False).cuda() # EDSC_s
    checkpoint = torch.load('../pretrained_model/EDSC_s_l1.ckpt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)

    # path = '../../data/UCF101/ucf101_interp_ours/'
    path = '/data/codec/zhangdy/video_interpolation/source_data/ucf101_interp_ours/'
    dirs = os.listdir(path)

    psnr_list = []
    ssim_list = []
    time_list = []
    # print('=========>Start Calculate PSNR and SSIM')
    for d in tqdm(dirs):
        img0 = (path + d + '/frame_00.png')
        img1 = (path + d + '/frame_02.png')
        gt = (path + d + '/frame_01_gt.png')
        # img0 = (torch.tensor(Image.open(img0).unsqueeze(0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
        # img1 = (torch.tensor(Image.open(img1).unsqueeze(0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
        # gt = (torch.tensor(Image.open(gt).unsqueeze(0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
        img0 = Image.open(img0)
        img1 = Image.open(img1)
        gt = Image.open(gt)

        img0 = transform(img0).unsqueeze(0).cuda()
        img1 = transform(img1).unsqueeze(0).cuda()
        gt = transform(gt).unsqueeze(0).cuda()
        if img1.size(1)==1:
            img0 = img0.expand(-1, 3,-1,-1)
            img1 = img1.expand(-1, 3,-1,-1)
        # inference
        pred = model([img0, img1])[0]
        # pred = torch.clamp(pred, 0, 1)
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
        pred = model([img0, img1])[0]
        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end))
    time_list.remove(min(time_list))
    time_list.remove(max(time_list))
    print("Avg PSNR: {} SSIM: {} Time: {}".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list) / 100))


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