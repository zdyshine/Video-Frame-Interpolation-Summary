import cv2
import ffmpeg
import glob
import mimetypes
import numpy as np
import os
import subprocess
import torch
from os import path as osp
from torch.nn import functional as F
from utils.img_util import img2tensor

def get_video_meta_info(video_path):
    """get the meta info of the video by using ffprobe with python interface"""
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    # ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['audio'] = None
    try:
        ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    except KeyError:  # bilibili transcoder dont have nb_frames
        ret['duration'] = float(probe['format']['duration'])
        ret['nb_frames'] = int(ret['duration'] * ret['fps'])
        print(ret['duration'], ret['nb_frames'])
    return ret


def get_sub_video(args, num_process, process_idx):
    """Cut the whole video into num_process parts, return the process_idx-th part"""
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    out_path = osp.join(args.output, 'inp_sub_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin,
        f'-i {args.input}',
        f'-ss {part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '',
        '-async 1',
        out_path,
        '-y',
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path


class Reader:
    """read frames from a video stream or frames list"""

    def __init__(self, args, total_workers=1, worker_idx=0, device=torch.device('cuda')):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            # read bgr from stream, which is the same format as opencv
            self.stream_reader = (
                ffmpeg
                    .input(video_path)
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
                    .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )  # yapf: disable  # noqa
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])  # lazy load
            self.width, self.height = tmp_img.size
        self.idx = 0
        self.device = device

        tmp = max(32, int(32 / args.outscale))
        ph = ((self.height - 1) // tmp + 1) * tmp
        pw = ((self.width - 1) // tmp + 1) * tmp
        self.padding = (0, pw - self.width, 0, ph - self.height)

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        """the fps of sr video is set to the user input fps first, followed by the input fps,
        If the first two values are None, then the commonly used fps 24 is set"""
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        """return the number of frames for this worker, however, this may be not accurate for video stream"""
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            # end of stream
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            img = self.get_frame_from_stream()
        else:
            img = self.get_frame_from_list()

        if img is None:
            raise StopIteration

        # bgr uint8 numpy -> rgb float32 [0, 1] tensor on device
        img = img.astype(np.float32) / 255.
        # img = mod_crop(img, self.args.mod_scale)
        img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(self.device)
        img = F.pad(img, self.padding)
        if self.args.half:
            # half precision won't make a big impact on visuals
            img = img.half()
        return img

    def close(self):
        # close the video stream
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:
    """write frames to a video stream"""

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        vsp = video_save_path
        if audio is not None:
            self.stream_writer = (
                ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}', framerate=fps)
                    .output(audio, vsp, pix_fmt='yuv420p', vcodec='libx264', loglevel='error', acodec='copy')
                    .overwrite_output()
                    .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )  # yapf: disable  # noqa
        else:
            # self.stream_writer = (
            #     ffmpeg
            #     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}', framerate=fps)
            #     .output(vsp, pix_fmt='yuv420p', vcodec='libx264', loglevel='error')
            #     .overwrite_output()
            #     .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            # )  # yapf: disable  # noqa

            self.stream_writer = (
                ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}', framerate=fps)
                    .output(vsp, aspect='16:9', preset='slower', pix_fmt='yuv420p',
                            # vcodec='libx264', x264opts=f'force-cfr:fps={fps}:qp=8:colorprim=bt709:transfer=bt709:colormatrix=bt709',
                            vcodec='libx264', x264opts=f'force-cfr:fps={fps}:qp=10',
                            loglevel='error')
                    .overwrite_output()
                    .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )

        self.out_width = out_width
        self.out_height = out_height
        self.args = args

    def write_frame(self, frame):
        if self.args.outscale != self.args.netscale:
            frame = cv2.resize(frame, (self.out_width, self.out_height), interpolation=cv2.INTER_LANCZOS4)
        self.stream_writer.stdin.write(frame.tobytes())

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()
'''
# crop video filter
writer = (ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(int(scenes['W']), int(scenes['H'])), r=fps)
            .filter('crop', w, h,  x0, y0)
            # .output(save_cropped_face_path, vcodec='libx264', pix_fmt='yuv444p', video_bitrate='10M', r=fps, **{'strict': 2, 'qscale': 0})
            .output(save_cropped_face_path, vcodec='libx264', pix_fmt='yuv444p', r=fps, **{'strict': 2, 'qscale': 0, 'crf': 10})
            .global_args('-hide_banner').global_args('-loglevel', 'quiet')
            .overwrite_output()
            .run_async(pipe_stdin=True, cmd=ffmpeg_bin))
writer.stdin.write(np.ascontiguousarray(frame).tobytes())   
writer.stdin.close()
writer.wait()
'''
