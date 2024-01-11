import argparse
import os.path
import torch
from copy import deepcopy
from collections import OrderedDict
from network.msrswvsr_arch import MSRSWVSR


def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='input test image folder or video path')
    parser.add_argument('-o', '--output', type=str, default='results', help='save image/video path')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='AnimeSR_v2',
        help='Model names: AnimeSR_v2 | AnimeSR_v1-PaperModel. Default:AnimeSR_v2')
    parser.add_argument(
        '-s',
        '--outscale',
        type=int,
        default=2,
        help='The netscale is x4, but you can achieve arbitrary output scale (e.g., x2) with the argument outscale'
        'The program will further perform cheap resize operation after the AnimeSR output. '
        'This is useful when you want to save disk space or avoid too large-resolution output')
    parser.add_argument(
        '--expname', type=str, default='animesr', help='A unique name to identify your current inference')
    parser.add_argument(
        '--netscale',
        type=int,
        default=2,
        help='the released models are all x4 models, only change this if you train a x2 or x1 model by yourself')
    parser.add_argument(
        '--mod_scale',
        type=int,
        default=2,
        help='the scale used for mod crop, since AnimeSR use a multi-scale arch, so the edge should be divisible by 4')
    parser.add_argument('--fps', type=int, default=None, help='fps of the sr videos')
    parser.add_argument('--half', action='store_true', help='use half precision to inference')

    return parser

def load_checkpoint_basicsr(model, model_path, strict=True):
    # load checkpoint
    # param_key = 'params'
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    if 'params_ema' in load_net:
        param_key = 'params_ema'
    else:
        param_key = 'params'

    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
        load_net = load_net[param_key]

    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
        # else:
        #     k = k.replace('repaire.', '')
        #     print(k)
        #     load_net[k] = v
        #     load_net.pop(k)
    model.load_state_dict(load_net, strict=strict)

def load_checkpoint_mmediting(model, model_path, strict=True):
    load_net = torch.load(model_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'generator_ema.'
    for k, v in load_net['state_dict'].items():
        # print('1111111111')
        # print(k) # 查看待替换的开头

        # if k.startswith('generator.'):     # 使用 generator的权重
        #     load_net_clean[k[10:]] = v

        # if k.startswith('generator.'):  # 使用 generator_ema的权重
        #     continue
        if k.startswith('generator_ema.'):  # 使用 generator_ema的权重
            load_net_clean[k[14:]] = v
        # else:
        #     raise RuntimeError('load error')
    model.load_state_dict(load_net_clean, strict=strict)

def get_inference_model(args, device) -> MSRSWVSR:
    """return an on device model with eval mode"""
    # set up model
    # SR
    # model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=4)
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/code_ffmpegv2/weights/AnimeSR_v2.pth'  # 目前最优
    # loadnet = torch.load(model_path)
    # model.load_state_dict(loadnet, strict=True)
    # SRx2 音综修复
    model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=2)
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/AnimeSR/experiments/001_train_srx2_step1_net_gan/models/net_g_60000.pth' # 目前最优
    model_path = f'/test/zhangdy/code_zdy/code_zdy/AwesomeAI/AnimeSR/experiments/001_train_srx2_step1_net_psnr/models/net_g_64000.pth' # 目前最优
    # repaire
    # model = MSRSWVSR(num_feat=64, num_block=[6, 4, 2], netscale=1)

    # model_path = f'weights/{args.model_name}.pth'
    #################### 超分
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/BasicSR/experiments/train_sruhd_step1_net_animesr/models/net_g_170000.pth'
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/BasicSR/experiments/train_sruhd_step2_net_animesr_gan/models/net_g_285000.pth'
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/BasicSR/experiments/train_sruhd_step2_net_animesr_gan/models/net_g_285000.pth'
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/BasicSR/experiments/001_train_srx2_step1_net_gan_huaweiyun/models/net_g_28000.pth' # 20230322最优
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/AnimeSR/experiments/001_train_srx2_step1_net_gan/models/net_g_60000.pth' # 目前最优
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/BasicSR/experiments/001_train_srx2_step1_net_psnr_huaweiyun/models/net_g_16000.pth'
    # model_path = f'/test/zhangdy/code_zdy/code_real/mmediting/experiments/vsrmsrsw_wogan_c64b532_nf15_lr1e-4_800k_mgtv/iter_104000.pth'

    #################### 修复
    # model_path = f'/test/zhangdy/code_zdy/code_basicsr/AnimeSR/experiments/003_train_repaire_gan/net_g_16000.pth'

    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'
    print('load from: ', model_path)
    try:
        load_checkpoint_basicsr(model, model_path)
    except:
        print('load_checkpoint_basicsr error, load others ...')
        loadnet = torch.load(model_path)
        model.load_state_dict(loadnet, strict=True)

    # load_checkpoint_mmediting(model, model_path)

    # loadnet = torch.load(model_path)
    # model.load_state_dict(loadnet, strict=True)
    model.eval()
    model = model.to(device)

    # num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # print(num_parameters)
    # exit(0)

    return model.half() if args.half else model
