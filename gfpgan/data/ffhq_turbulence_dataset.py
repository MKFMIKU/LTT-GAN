import cv2
import math
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)


@DATASET_REGISTRY.register()
class FFHQTurbulenceDataset(data.Dataset):

    def __init__(self, opt):
        super(FFHQTurbulenceDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']

        self.paths = paths_from_folder(self.gt_folder)

        # degradations
        if self.opt['phase'] == 'train':
            self.blur_kernel_size = opt['blur_kernel_size']
            self.kernel_list = opt['kernel_list']
            self.kernel_prob = opt['kernel_prob']
            self.blur_sigma = opt['blur_sigma']
            self.downsample_range = opt['downsample_range']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        img_lq, img_gt = img_gt[:, 512:], img_gt[:, :512]
        # random horizontal flip
        if self.opt['phase'] == 'train':
            img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
            img_lq, status = augment(img_lq, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
            h, w, _ = img_gt.shape
            # ------------------------ generate lq image ------------------------ #
            # blur
            # kernel = degradations.random_mixed_kernels(
            #     self.kernel_list,
            #     self.kernel_prob,
            #     self.blur_kernel_size,
            #     self.blur_sigma,
            #     self.blur_sigma, [-math.pi, math.pi],
            #     noise_range=None)
            # img_lq = cv2.filter2D(img_lq, -1, kernel)
            # # downsample
            # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
            # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
