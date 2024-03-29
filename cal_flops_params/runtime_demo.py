import os
import torch
import pathlib
import logging
import argparse
from tqdm import tqdm
from collections import OrderedDict

from utils import util_logger
from utils.model_summary import get_model_flops


def main(args):
    """
    SETUP DIRS
    """
    pathlib.Path(os.path.join(args.save_dir, args.submission_id, "results")).mkdir(parents=True, exist_ok=True)
    
    """
    SETUP LOGGER
    """
    util_logger.logger_info("IACC-VFI", log_path=os.path.join(args.save_dir, args.submission_id, f"Submission_{args.submission_id}.txt"))
    logger = logging.getLogger("IACC-VFI")
    
    """
    BASIC SETTINGS
    """
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    """
    LOAD MODEL
    """
    from RRIN import RRIN
    model = RRIN()
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    
    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>> model name: {} <<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model_name))
    logger.info('Params number: {:.6f} M'.format(number_parameters / 1e6))

    """
    SETUP RUNTIME
    """
    test_results = OrderedDict()
    test_results["runtime"] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    """
    TESTING
    """  
    input_data = torch.randn((1, 3, args.crop_size[0], args.crop_size[1])).to(device)

    # GPU warmp up
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(20): # 50
            out = model(input_data)
        print('input data shape: {} \n model out shape: {}'.format(input_data.shape, out.shape))
    print("Start timing ...")
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in tqdm(range(args.repeat)):
            start.record()
            _ = model(input_data)
            end.record()

            torch.cuda.synchronize()

            test_results["runtime"].append(start.elapsed_time(end))  # milliseconds

        ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"])
        logger.info('------> Average runtime of ({}) is : {:.6f} ms'.format(args.model_name, ave_runtime / args.batch_size))
        logger.info('------> FPS of ({}) is : {:.2f} fps'.format(args.model_name, 1 / (ave_runtime / args.batch_size / 1000.)))

        input_dim = (6, int(args.crop_size[0]/args.scale), int(args.crop_size[1]/args.scale))
        flops = get_model_flops(model, input_dim, print_per_layer_stat=False)
        flops = flops / 10 ** 9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters / 10 ** 6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))


'''
python runtime_demo.py --submission-id RRIN --model-name RRIN
'''
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # specify submission
    parser.add_argument("--submission-id", type=str)
    parser.add_argument("--model-name", type=str)

    # specify dirs
    parser.add_argument("--save-dir", type=str, default="log")
    
    # specify test case
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, nargs="+", default=[1920, 1080], help="We use 4K images for final testing. During the development phase we provide a validation set with GT size of [2040, 1080].")

    args = parser.parse_args()
        
    main(args)
