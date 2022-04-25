import os
from os import PathLike
from pathlib import Path

import einops
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

DOWNSAMPLE_SHORT_EDGE_LENGTH = 512
NETWORK_IMAGE_DIMENSIONS = (512, 512)


class FidderDataSet(Dataset):
    """Fiducial mask dataset.

    - Images are in subfolders of root_dir called 'images' and 'masks'
    - Image shape for network (512, 512)
    - Images are downsampled so that the short edge is 1024px long

    - train/eval mode activated via methods of the same name
    """

    def __init__(self, root_dir: PathLike):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'images'
        self.mask_dir = self.root_dir / 'masks'

        # basic check only
        if len(self.image_files) != len(self.mask_files):
            raise FileNotFoundError(
                "masks and images directories must contain the same number of images"
            )

        self.is_training = True

    @property
    def is_validating(self):
        return not self.is_training

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    @property
    def image_files(self):
        return sorted(list(self.image_dir.glob('*.tif')))

    @property
    def mask_files(self):
        return sorted(list(self.mask_dir.glob('*.tif')))

    def check_files(self, *files: Path):
        for file in files:
            if file.exists() and file.is_file():
                continue
            else:
                raise FileNotFoundError(f'{file} not found')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx].name

        image_file, mask_file = self.image_dir / name, self.mask_dir / name
        self.check_files(image_file, mask_file)

        image = torch.tensor(imageio.imread(image_file), dtype=torch.float32)
        mask = torch.tensor(imageio.imread(mask_file), dtype=torch.float32)

        # add channel dim for torchvision transforms
        image = einops.rearrange(image, 'h w -> 1 h w')
        mask = einops.rearrange(mask, 'h w -> 1 h w')

        # simple resize of short edge to 1024px
        image, mask = self.resize(image, mask, size=DOWNSAMPLE_SHORT_EDGE_LENGTH)

        # augment if training, random crop if validating
        # if self.is_training:
        #     image, mask = self.augment(image, mask)
        # else:
        #     image, mask = self.random_crop(image, mask)

        # normalise image
        image = (image - torch.mean(image)) / torch.std(image)

        # remove channel dim for loss evaluation
        mask = einops.rearrange(mask, '1 h w -> h w')

        sample = {
            'image': image.float().contiguous(),
            'mask': mask.long().contiguous()
        }
        return sample

    def resize(self, *images, size: int):
        return [TF.resize(image, size) for image in images]

    def augment(self, *images):
        # input images have a short edge length 1024
        target_area = np.prod(NETWORK_IMAGE_DIMENSIONS)
        image_area = np.prod(images[0].shape)
        target_scale = target_area / image_area
        crop_parameters = T.RandomResizedCrop.get_params(
            images[0], scale=[0.75 * target_scale, 1.33 * target_scale], ratio=[1, 1])

        images = [TF.crop(image, *crop_parameters) for image in images]
        images = [
            TF.resize(image, size=NETWORK_IMAGE_DIMENSIONS)
            for image
            in images
        ]

        # random flips
        if np.random.uniform(low=0, high=1) > 0.5:
            images = [TF.hflip(image) for image in images]
        if np.random.uniform(low=0, high=1) > 0.5:
            images = [TF.vflip(image) for image in images]

        return images

    def random_crop(self, *images):
        crop_parameters = T.RandomCrop.get_params(
            images[0], output_size=NETWORK_IMAGE_DIMENSIONS
        )
        return [TF.crop(image, *crop_parameters) for image in images]
