import shutil
from tqdm import tqdm

from ffmpeg_utils import *
from utils import video_util
from inference_config import get_base_argument_parser, get_inference_model
from utils.img_util import img2tensor, tensor2img
from utils.logger import AvgTimer
from utils.pytorch_msssim import ssim_matlab


@torch.no_grad()
def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # prepare model
    model = get_inference_model(args, device)
    TTA = False

    def make_inference_4k(I0, I1, embt, exp):
        middle = torch.zeros_like(I0)
        # middle = model.inference(I0, I1, embt)
        I0_, I1_ = I0[:, :, :, :1952], I1[:, :, :, :1952] # 1920 + 32
        middle1 = model.inference(I0_, I1_ , TTA=TTA, fast_TTA=TTA)
        I0_, I1_ = I0[:, :, :, 1888:], I1[:, :, :, 1888:]  # 1920 - 32
        middle2 = model.inference(I0_, I1_, TTA=TTA, fast_TTA=TTA)
        # print(middle2.shape, middle1.shape, middle.shape)
        middle[:, :, :, :1920] = middle1[:, :, :, :1920]
        middle[:, :, :, 1920:] = middle2[:, :, :, 32:]
        middle = torch.clamp(middle, 0, 1)
        if exp == 1:
            return [middle]
        first_half = make_inference_4k(I0, middle, embt, exp=exp - 1)
        first_half = torch.clamp(first_half, 0, 1)
        second_half = make_inference_4k(middle, I1, embt, exp=exp - 1)
        second_half = torch.clamp(second_half, 0, 1)
        return [*first_half, middle, *second_half]

    def make_inference(I0, I1, embt, exp):
        middle = model.inference(I0, I1, TTA=TTA, fast_TTA=TTA)
        middle = torch.clamp(middle, 0, 1)
        if exp == 1:
            return [middle]
        first_half = make_inference(I0, middle, embt, exp=exp - 1)
        first_half = torch.clamp(first_half, 0, 1)
        second_half = make_inference(middle, I1, embt, exp=exp - 1)
        second_half = torch.clamp(second_half, 0, 1)
        return [*first_half, middle, *second_half]

    # prepare reader and writer
    reader = Reader(args, total_workers, worker_idx, device=device)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    # height = height - height % args.mod_scale
    # width = width - width % args.mod_scale
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, video_save_path, fps * 2)

    # initialize pre/cur/nxt frames, pre sr frame, and pre hidden state for inference
    end_flag = False
    prev = reader.get_frame()
    # cur = prev
    try:
        nxt = reader.get_frame()
    except StopIteration:
        end_flag = True
        nxt = prev

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    model_timer = AvgTimer()  # model inference time tracker
    i_timer = AvgTimer()  # I(input read) time tracker
    o_timer = AvgTimer()  # O(output write) time tracker

    embt = torch.tensor(1 / 2).view(1, 1, 1, 1).float().cuda()
    exp = 1
    while True:
        # inference at current step
        torch.cuda.synchronize(device=device)
        model_timer.start()
        I0_small = F.interpolate(prev, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(nxt, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small.float(), I1_small.float())
        if ssim < 0.5:
            out = []
            step = 1 / (2 ** exp)
            alpha = 0
            for i in range((2 ** exp) - 1):
                alpha += step
                beta = 1 - alpha
                lastframe = tensor2img(prev, rgb2bgr=False)
                frame = tensor2img(nxt, rgb2bgr=False)
                out.append(torch.from_numpy(np.transpose(
                    (cv2.addWeighted(frame, alpha, lastframe, beta, 0)[:, :, ::-1].copy()),
                    (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        else:
            if args.large4k:
                out = make_inference_4k(prev, nxt, embt.half(), exp=1)
            else:
                out = make_inference(prev, nxt, embt.half(), exp=1)
        torch.cuda.synchronize(device=device)
        model_timer.record()

        # write current sr frame to video stream
        torch.cuda.synchronize(device=device)
        o_timer.start()
        prev_frame = tensor2img(prev, rgb2bgr=False)
        writer.write_frame(prev_frame[:height, :width])
        for mid in out:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))  # rgb
            writer.write_frame(mid[:height, :width])
            # writer.write_frame(output_frame)
        torch.cuda.synchronize(device=device)
        o_timer.record()

        # if end of stream, break
        if end_flag:
            break

        # move the sliding window
        torch.cuda.synchronize(device=device)
        i_timer.start()
        # prev = cur
        prev = nxt
        try:
            nxt = reader.get_frame()
        except StopIteration:
            nxt = prev
            end_flag = True
        torch.cuda.synchronize(device=device)
        i_timer.record()

        # update&print infomation
        pbar.update(1)
        pbar.set_description(
            f'I: {i_timer.get_avg_time():.4f} O: {o_timer.get_avg_time():.4f} Model: {model_timer.get_avg_time():.4f}')

    reader.close()
    writer.close()


def run(args):
    if args.suffix is None:
        args.suffix = ''
    else:
        args.suffix = f'_{args.suffix}'
    # video_save_path = osp.join(args.output, f'{args.video_name}{args.suffix}.mp4')
    video_save_path = osp.join(args.output, f'{args.video_name}{args.suffix}.ts')

    # set up multiprocessing
    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    if num_process == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_video(args, video_save_path, device=device)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    out_sub_videos_dir = osp.join(args.output, 'out_sub_videos')
    os.makedirs(out_sub_videos_dir, exist_ok=True)
    os.makedirs(osp.join(args.output, 'inp_sub_videos'), exist_ok=True)

    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(out_sub_videos_dir, f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'out_sub_videos/{i:03d}.mp4\'\n')
    # To avoid video&audio desync as mentioned in https://github.com/xinntao/Real-ESRGAN/issues/388
    # we use the solution provided in https://stackoverflow.com/a/52156277 to solve this issue
    cmd = [
        args.ffmpeg_bin,
        '-f', 'concat',
        '-safe', '0',
        '-i', f'{args.output}/vidlist.txt',
        '-c:v', 'copy',
        '-af', 'aresample=async=1000',
        video_save_path,
        '-y',
    ]  # yapf: disable
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(out_sub_videos_dir)
    shutil.rmtree(osp.join(args.output, 'inp_sub_videos'))
    os.remove(f'{args.output}/vidlist.txt')


def main():
    """Inference demo for AnimeSR.
    It mainly for restoring anime videos.
    """
    parser = get_base_argument_parser()
    parser.add_argument(
        '--extract_frame_first',
        action='store_true',
        help='if input is a video, you can still extract the frames first, other wise AnimeSR will read from stream')
    parser.add_argument(
        '--num_process_per_gpu', type=int, default=1, help='the total process is number_process_per_gpu * num_gpu')
    parser.add_argument(
        '--suffix', type=str, default=None, help='you can add a suffix string to the sr video name, for example, x2')
    args = parser.parse_args()
    # args.ffmpeg_bin = os.environ.get('ffmpeg_exe_path', 'ffmpeg')
    args.ffmpeg_bin = '/test/ffmpeg4.4'

    args.input = args.input.rstrip('/').rstrip('\\')

    if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
        is_video = True
    else:
        is_video = False

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    # prepare input and output
    args.video_name = osp.splitext(osp.basename(args.input))[0]
    # args.output = osp.join(args.output, args.expname, 'videos', args.video_name)
    args.output = osp.join(args.output, args.expname)
    os.makedirs(args.output, exist_ok=True)
    if args.extract_frame_first:
        inp_extracted_frames = osp.join(args.output, 'inp_extracted_frames')
        os.makedirs(inp_extracted_frames, exist_ok=True)
        video_util.video2frames(args.input, inp_extracted_frames, force=True, high_quality=True)
        video_meta = get_video_meta_info(args.input)
        args.fps = video_meta['fps']
        args.input = inp_extracted_frames

    run(args)

    if args.extract_frame_first:
        shutil.rmtree(args.input)


# single gpu and single process inference
# CUDA_VISIBLE_DEVICES=0 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 2 --expname animesr_v2 --num_process_per_gpu 1 --suffix 1gpu1process
# # single gpu and multi process inference (you can use multi-processing to improve GPU utilization)
# CUDA_VISIBLE_DEVICES=0 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 2 --expname animesr_v2 --num_process_per_gpu 3 --suffix 1gpu3process
# # multi gpu and multi process inference
# CUDA_VISIBLE_DEVICES=0,1 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 2 --expname animesr_v2 --num_process_per_gpu 3 --suffix 2gpu6process
if __name__ == '__main__':
    main()
    '''
    视频插帧算法应用
    CUDA_VISIBLE_DEVICES=0 python inference_ema_vfi.py -i ./casesc.mp4 -o ./results -n IFRNet -s 1 --expname EMAVFI --num_process_per_gpu 1 --suffix 50fpsv1_half
    '''
