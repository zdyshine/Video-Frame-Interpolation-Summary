import torch.utils.data as data
import os, glob
import random
import torch
import numpy as np
import cv2
import utils
from os.path import join, isdir

def _make_dataset(dir):
    """
    Creates a 2D list of all the frames in N clips containing
    M frames each.
    2D List Structure:
    [[frame00, frame01,...,frameM],  <-- clip0
     [frame00, frame01,...,frameM],  <-- clip1
     ...,
     [frame00, frame01,...,frameM]   <-- clipN
    ]
    Parameters
        dir : string
            root directory containing clips.
    Tips
        read 700 clips, each of them contains 100 frames (only read list)
    """
    framePath = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framePath.append([])
        # Find and loop over all the frames inside the clip.
        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framePath[index].append(os.path.join(clipsFolderPath, image))
    return framePath


def _make_video_dataset(dir):
    """
    Creates a 1D list of all the frames.
    1D List Structure:
    [frame0, frame1,...,frameN]
    """
    framePath = []
    # Find and loop over all the frames in root `dir`.
    for image in sorted(os.listdir(dir)):
        # Add path to list.
        framePath.append(os.path.join(dir, image))
    return framePath


class RandomCrop(object):
    def __init__(self, video_size, patch_size, scale):
        ih, iw = video_size

        self.tp = patch_size
        self.ip = self.tp // scale

        self.ix = random.randrange(0, iw - self.ip + 1)
        self.iy = random.randrange(0, ih - self.ip + 1)

        self.tx, self.ty = scale * self.ix, scale * self.iy

    def __call__(self, clip, mode='target'):
        if mode == 'target':
            ret = clip[:, self.ty:self.ty + self.tp, self.tx:self.tx + self.tp, :]  # [T, H, W, C]
        else:
            ret = clip[:, self.iy:self.iy + self.ip, self.ix:self.ix + self.ip, :]

        return ret


class CenterCrop(object):
    def __init__(self, video_size, patch_size, scale):
        ih, iw = video_size

        self.tp = patch_size
        self.ip = self.tp // scale

        self.ix = (iw - self.ip) // 2
        self.iy = (ih - self.ip) // 2

        self.tx, self.ty = scale * self.ix, scale * self.iy

    def __call__(self, clip, mode='target'):
        if mode == 'target':
            ret = clip[:, self.ty:self.ty + self.tp, self.tx:self.tx + self.tp, :]  # [T, H, W, C]

        else:
            ret = clip[:, self.iy:self.iy + self.ip, self.ix:self.ix + self.ip, :]

        return ret


class Agument(object):
    def __init__(self):
        self.hflip = random.random() < 0.5
        self.vflip = random.random() < 0.5
        self.rot90 = random.random() < 0.5

    def augment(self, video):
        if self.hflip: video = video[:, :, ::-1, :]
        if self.vflip: video = video[:, ::-1, :, :]
        if self.rot90: video = video.transpose(0, 2, 1, 3)  # T(0), H(1), W(2), C(3) --> T, W, H, C
        return video

    def __call__(self, video):
        return self.augment(video)


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]  # BGR --> RGB


def read_img_to_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]  # BGR --> RGB
    img = np.float32(img) / 255.0
    img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img_tensor

def get_ref_index(length, sample_length):
    # if random.uniform(0, 1) > 0.5:
    #     ref_index = random.sample(range(length), sample_length)
    #     ref_index.sort()
    # else:
    pivot = random.randint(0, length-sample_length)
    ref_index = [pivot+i for i in range(sample_length)]
    return ref_index

class MultiFramesDataset(data.Dataset):
    def __init__(self, opt):
        db_dir = opt.train + '/sequences'
        self.folder_list = [(db_dir + '/' + f) for f in os.listdir(db_dir) if isdir(join(db_dir, f))]
        self.triplet_list = []
        for folder in self.folder_list:
            self.triplet_list += [(folder + '/' + f) for f in os.listdir(folder) if isdir(join(folder, f))]

        self.triplet_list = np.array(self.triplet_list)
        if opt.debug:
            self.triplet_list = self.triplet_list[:100]
        print('=========>len   Train:', len(self.triplet_list))
        # print(self.triplet_list[:2])
        self.opt = opt

    def __getitem__(self, index):
        # current_clip = self.triplet_list[index]
        current_clip = sorted(glob.glob(self.triplet_list[index] + '/*.png'))
        # print(current_clip)
        frame_indexs = get_ref_index(len(current_clip), 3)
        # frame_indexs = [0, 1, 2]
        # print(frame_indexs)
        current_frames = []
        for t in frame_indexs:
            current_frames.append(default_loader(current_clip[t]))

        # data crop and augmentation
        # random crop
        get_patch = RandomCrop(current_frames[0].shape[:2], patch_size=self.opt.patch_size, scale=1)
        augment = Agument()

        clip_frame = np.stack(current_frames, axis=0)  # T = 7
        clip_frame = get_patch(clip_frame, mode='input')

        # if self.opt.geometry_aug:
        clip_frame = augment(clip_frame)

        # 进行30%的随机翻转
        if random.random() >= 0.3:
            clip_frame = clip_frame[::-1, :, :, :].copy()

        # convert (T, H, W, C) array to (T, C, H, W) tensor
        tensor_frame = torch.from_numpy(
            np.float32(np.ascontiguousarray(np.transpose(clip_frame, (0, 3, 1, 2)))) / 255.0)

        return tensor_frame  # tensor_damage --> (T, C, H, W), tensor_ref --> (C, H, W)

    def __len__(self):
        return len(self.triplet_list)

class FramesValDataset(data.Dataset):

    def __init__(self, opt):
        framesPath = _make_dataset(opt.root_val)  # all frames
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + opt.root_val + "\n")
        print('=========>len   val:', len(framesPath))
        self.opt = opt
        self.framesPath = framesPath
        self.frame_num = opt.frame_num

    def __getitem__(self, index):
        current_clip = self.framesPath[index][100:103]

        # frame_indexs = get_ref_index(len(current_clip), self.frame_num)

        current_frames = []
        for t in range(len(current_clip)):
            current_frames.append(default_loader(current_clip[t]))

        # data crop and augmentation
        # random crop
        get_patch = CenterCrop(current_frames[0].shape[:2], patch_size=512, scale=1)

        clip_frame = np.stack(current_frames, axis=0)  # T = 7
        clip_frame = get_patch(clip_frame, mode='input')

        # convert (T, H, W, C) array to (T, C, H, W) tensor
        tensor_frame = torch.from_numpy(
            np.float32(np.ascontiguousarray(np.transpose(clip_frame, (0, 3, 1, 2)))) / 255.0)

        return tensor_frame  # tensor_damage --> (T, C, H, W), tensor_ref --> (C, H, W)

    def __len__(self):
        return len(self.framesPath)


class Middlebury_other(data.Dataset):
    def __init__(self, opt):
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper',
                        'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.opt = opt
    def __getitem__(self, index):
        item = self.im_list[index]
        current_frames = []
        current_frames.append(default_loader(self.opt.test_dir + '/' + item + '/frame10.png'))
        current_frames.append(default_loader(self.opt.gt_dir + '/' + item + '/frame10i11.png'))
        current_frames.append(default_loader(self.opt.test_dir + '/' + item + '/frame11.png'))

        clip_frame = np.stack(current_frames, axis=0)  # T = 7

        # convert (T, H, W, C) array to (T, C, H, W) tensor
        tensor_frame = torch.from_numpy(
            np.float32(np.ascontiguousarray(np.transpose(clip_frame, (0, 3, 1, 2)))) / 255.0)

        return tensor_frame  # tensor_damage --> (T, C, H, W), tensor_ref --> (C, H, W)

    def __len__(self):
        return len(self.im_list)



if __name__=='__main__':
    frame_indexs = get_ref_index(100, 5)
    print(frame_indexs)