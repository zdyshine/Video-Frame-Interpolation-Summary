import argparse
import os.path

def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='input test image folder or video path')
    parser.add_argument('-o', '--output', type=str, default='results', help='save image/video path')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='IFRNet',
        help='Model names: AnimeSR_v2 | AnimeSR_v1-PaperModel. Default:AnimeSR_v2')
    parser.add_argument(
        '-s',
        '--outscale',
        type=int,
        default=1,
        help='The netscale is x4, but you can achieve arbitrary output scale (e.g., x2) with the argument outscale'
        'The program will further perform cheap resize operation after the AnimeSR output. '
        'This is useful when you want to save disk space or avoid too large-resolution output')
    parser.add_argument(
        '--expname', type=str, default='IFRNet', help='A unique name to identify your current inference')
    parser.add_argument(
        '--netscale',
        type=int,
        default=1,
        help='the released models are all x4 models, only change this if you train a x2 or x1 model by yourself')
    parser.add_argument(
        '--mod_scale',
        type=int,
        default=1,
        help='the scale used for mod crop, since AnimeSR use a multi-scale arch, so the edge should be divisible by 4')
    parser.add_argument('--fps', type=int, default=None, help='fps of the sr videos')
    parser.add_argument('--half', action='store_true', help='use half precision to inference')
    parser.add_argument('--large4k', action='store_true', help='4k to inference')

    return parser

def get_inference_model(args, device):
    """return an on device model with eval mode"""
    # set up model
    if args.expname == 'IFRNet':
        from define_load_model import get_IFRNet
        model = get_IFRNet()
        model = model.to(device)
    elif args.expname == 'WaveletVFI':
        from define_load_model import get_WaveletVFI
        model = get_WaveletVFI()
        model = model.to(device)
    elif args.expname == 'EMAVFI':
        from define_load_model import get_EMAVFI
        model = get_EMAVFI()
        model.device(device)
    else:
        print('please define model by args.expname ...')
        exit()
    # model = model.to(device)

    return model.half() if args.half else model
