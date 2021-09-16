import sys
from os.path import join, dirname, abspath
import random
from typing import Tuple
import numbers
from collections.abc import Sequence

from torch import Tensor
import torch
import munch
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

PROB_THRESHOLD = 0.5  # probability threshold.

"Credit: https://github.com/clovaai/wsolevaluation/blob/master/data_loaders.py"

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.functional import _functional as dlibf
from dlib.configure import constants

from dlib.utils.shared import reformat_id

_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_SPLITS = (constants.TRAINSET, constants.VALIDSET, constants.TESTSET)

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = join(metadata_root, 'image_ids_proxy.txt')
    metadata.class_labels = join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = join(metadata_root, 'image_sizes.txt')
    metadata.localization = join(metadata_root, 'localization.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels


def get_bounding_boxes(metadata):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
            x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, x1, y0, y1))
            else:
                boxes[image_id] = [(x0, x1, y0, y1)]
    return boxes


def get_mask_paths(metadata):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


def get_cams_paths(root_data_cams: str, image_ids: list) -> dict:
    paths = dict()
    for idx_ in image_ids:
        paths[idx_] = join(root_data_cams, '{}.pt'.format(reformat_id(idx_)))

    return paths


class WSOLImageLabelDataset(Dataset):
    def __init__(self,
                 data_root,
                 metadata_root,
                 transform,
                 proxy,
                 resize_size,
                 crop_size,
                 num_sample_per_class=0,
                 root_data_cams=''):

        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class

        self.cams_paths = None
        if os.path.isdir(root_data_cams):
            self.cams_paths = get_cams_paths(root_data_cams=root_data_cams,
                                             image_ids=self.image_ids)

        self.resize_size = resize_size
        self.crop_size = crop_size

        self._adjust_samples_per_class()

    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(join(self.data_root, image_id))
        image = image.convert('RGB')
        raw_img = image.copy()

        std_cam = None
        if self.cams_paths is not None:
            std_cam_path = self.cams_paths[image_id]
            # h', w'
            std_cam: torch.Tensor = torch.load(f=std_cam_path,
                                               map_location=torch.device('cpu'))
            assert std_cam.ndim == 2
            std_cam = std_cam.unsqueeze(0)  # 1, h', w'

        image, raw_img, std_cam = self.transform(image, raw_img, std_cam)

        raw_img = np.array(raw_img, dtype=np.float32)  # h, w, 3
        raw_img = dlibf.to_tensor(raw_img).permute(2, 0, 1)  # 3, h, w.

        if std_cam is None:
            std_cam = 0

        return image, image_label, image_id, raw_img, std_cam

    def __len__(self):
        return len(self.image_ids)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Compose(object):
    def __init__(self, mytransforms: list):
        self.transforms = mytransforms

        for t in mytransforms:
            assert any([isinstance(t, Resize), isinstance(t, RandomCrop),
                       isinstance(t, RandomHorizontalFlip),
                       isinstance(t, transforms.ToTensor),
                        isinstance(t, transforms.Normalize)]
                       )

    def chec_if_random(self, transf):
        if isinstance(transf, RandomCrop):
            return True

    def __call__(self, img, raw_img, std_cam):
        for t in self.transforms:
            if isinstance(t, (RandomHorizontalFlip, RandomCrop, Resize)):
                img, raw_img, std_cam = t(img, raw_img, std_cam)
            else:
                img = t(img)

        return img, raw_img, std_cam

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class _BasicTransform(object):
    def __call__(self, img, raw_img):
        raise NotImplementedError


class RandomHorizontalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, std_cam):
        if random.random() < self.p:
            std_cam_ = std_cam
            if std_cam_ is not None:
                std_cam_ = TF.hflip(std_cam)
            return TF.hflip(img), TF.hflip(raw_img), std_cam_

        return img, raw_img, std_cam

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(_BasicTransform):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]
                   ) -> Tuple[int, int, int, int]:

        w, h = TF._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image "
                "size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,
                 padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two "
                            "dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = TF._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        return img

    def __call__(self, img, raw_img, std_cam):
        img_ = self.forward(img)
        raw_img_ = self.forward(raw_img)
        assert img_.size == raw_img_.size

        i, j, h, w = self.get_params(img_, self.size)
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = self.forward(std_cam)
            std_cam_ = TF.crop(std_cam_, i, j, h, w)

        return TF.crop(img_, i, j, h, w), TF.crop(
            raw_img_, i, j, h, w), std_cam_

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding)


class Resize(_BasicTransform):
    def __init__(self, size,
                 interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. "
                            "Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, "
                             "it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, raw_img, std_cam):
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = TF.resize(std_cam_, self.size, self.interpolation)

        return TF.resize(img, self.size, self.interpolation), TF.resize(
            raw_img, self.size, self.interpolation), std_cam_

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


def get_data_loader(data_roots,
                    metadata_root,
                    batch_size,
                    workers,
                    resize_size,
                    crop_size,
                    proxy_training_set,
                    num_val_sample_per_class=0,
                    std_cams_folder=None,
                    get_splits_eval=None
                    ):

    def get_eval_tranforms():
        return Compose([
            Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ])

    if isinstance(get_splits_eval, list):
        assert len(get_splits_eval) > 0
        loaders = {
            split: DataLoader(
                WSOLImageLabelDataset(
                    data_root=data_roots[split],
                    metadata_root=join(metadata_root, split),
                    transform=get_eval_tranforms(),
                    proxy=False,
                    resize_size=resize_size,
                    crop_size=crop_size,
                    num_sample_per_class=0,
                    root_data_cams=''
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers)
            for split in get_splits_eval
        }
        return loaders

    dataset_transforms = dict(
        train=Compose([
            Resize((resize_size, resize_size)),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=get_eval_tranforms(),
        test=get_eval_tranforms()
    )

    loaders = {
        split: DataLoader(
            WSOLImageLabelDataset(
                data_root=data_roots[split],
                metadata_root=join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == constants.TRAINSET,
                resize_size=resize_size,
                crop_size=crop_size,
                num_sample_per_class=(num_val_sample_per_class
                                      if split == constants.VALIDSET else 0),
                root_data_cams=std_cams_folder[split]
            ),
            batch_size=batch_size,
            shuffle=split == constants.TRAINSET,
            num_workers=workers)
        for split in _SPLITS
    }
    return loaders
