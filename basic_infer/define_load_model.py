import os
import torch


def get_IFRNet():
    """return an on device model with eval mode"""
    from archs.vfi.IFRNet import Model
    # set up model
    model = Model()
    model_path = f'./checkpoints/IFRNet_Vimeo90K.pth'
    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    return model

def get_WaveletVFI():
    """return an on device model with eval mode"""

    def convert(param):
        return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

    from archs.WaveletVFI.WaveletVFI import WaveletVFI
    # set up model
    model = WaveletVFI()
    model_path = f'./checkpoints/waveletvfi_latest.pth'
    assert os.path.isfile(model_path), \
        f'{model_path} does not exist, please make sure you successfully download the pretrained models ' \
        f'and put them into the weights folder'

    model.load_state_dict(convert(torch.load('./checkpoints/waveletvfi_latest.pth', map_location='cpu')))
    model.eval()
    return model

def get_EMAVFI():
    """return an on device model with eval mode"""
    import archs.ema_vfi.config as cfg
    from archs.ema_vfi.Trainer import Model

    model_name = 'ours'  # ours | ours_small | ours_t |　ours_small_t
    TTA = True
    if model_name == 'ours_small':
        TTA = False
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=16,
            depth=[2, 2, 2, 2, 2]
        )
    else:
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F=32,
            depth=[2, 2, 2, 4, 4]
        )
    model = Model(-1)
    # load checkpoint
    model_path = f'./checkpoints/ema_ckpt/{model_name}.pkl'
    model.load_model(model_path)
    model.eval()
    return model
