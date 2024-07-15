import os

import torch
import torch.nn.functional as F

from config import *


# NOTE: we didn't use any TTA when evaluating
class Model:
    def __init__(self, local_rank):
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        # self.device()
        # BEGIN: load gmflow
        ckpt = torch.load(f'./pretrained/gmflow_sintel-0c07dcb3.pth')
        print(f"Loading GMFlow ckpt")
        model_dict = self.net.gmflow.state_dict()
        partial_dict = {k: v for k, v in ckpt['model'].items() if k in model_dict}
        self.net.gmflow.load_state_dict(partial_dict, strict=False)
        for param in self.net.gmflow.parameters():
            param.requires_grad = False
        print("GMFlow Parameters Loaded")
        # END: load gmflow

        # BEGIN: load local branch
        name = 'ours-local'  # small model: 15M Parameters
        # name = '11-4-base-model'  # base model: 59M Parameters
        self.load_model(name=name, rank=local_rank)
        print(f'{name} ckpt Loaded.')
        for param in self.net.feature_bone.parameters():
            param.requires_grad = False
        for param in self.net.block.parameters():
            param.requires_grad = False
        for param in self.net.unet.parameters():
            param.requires_grad = False

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def to(self, device):
        self.net.to(device)
        
    def device(self):
        self.net.to(torch.device("cuda"))

    def load_model(self, name=None, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }

        if rank <= 0:
            if name is None:
                name = self.name
            print(f"Loading {name} ckpt")
            ckpt = torch.load(f'./pretrained/{name}.pkl')
            self.net.load_state_dict(convert(ckpt['model']), strict=False)            

    @torch.no_grad()
    def hr_inference(self, img0, img1, TTA=False, down_scale=1.0, timestep=0.5, fast_TTA=False):
        '''
        Infer with down_scale flow
        Note: return BxCxHxW
        '''

        def infer(imgs):
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)

            flow, mask = self.net.calculate_flow(imgs_down, timestep)

            flow = F.interpolate(flow, scale_factor=1 / down_scale, mode="bilinear", align_corners=False) * (
                        1 / down_scale)
            mask = F.interpolate(mask, scale_factor=1 / down_scale, mode="bilinear", align_corners=False)

            pred = self.net.coraseWarp_and_Refine(imgs, flow, mask)
            return pred

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2

    @torch.no_grad()
    def inference(self, img0, img1, TTA=False, timestep=0.5, fast_TTA=False):
        imgs = torch.cat((img0, img1), 1)
        '''
        Noting: return BxCxHxW
        '''
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            inputs = torch.cat((imgs, imgs_), 0)
            _, _, _, preds, _ = self.net(inputs, timestep=timestep)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        _, _, _, pred, _ = self.net(imgs, timestep=timestep)
        if TTA == False:
            return pred
        else:
            _, _, _, pred2, _ = self.net(imgs.flip(2).flip(3), timestep=timestep)
            return (pred + pred2.flip(2).flip(3)) / 2
