import os
import torch
import numpy as np
import scipy.misc as m
from tqdm import tqdm

from torch.utils import data
from PIL import Image

from .utils import recursive_glob
from augmentations import *
from data.base_dataset import BaseDataset

import random


class Cityscapes_select_loader(BaseDataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
            self,
            cfg,
            writer,
            logger,
            augmentations=None,
    ):
        """__init__

        :param cfg: parameters of dataset
        :param writer: save the result of experiment
        :param logger: logging file
        :param augmentations:
        """

        self.cfg = cfg
        self.root = cfg['rootpath']
        self.split = cfg['split']
        self.is_transform = cfg.get('is_transform', True)
        self.augmentations = augmentations
        self.img_norm = cfg.get('img_norm', True)
        self.n_classes = 19
        self.img_size = (cfg['img_cols'], cfg['img_rows'])
        self.active_list_path = cfg['active_list_path'] if 'active_list_path' in cfg else None
        self.mean = np.array(self.mean_rgb['cityscapes'])
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.img_ids_full = recursive_glob(rootdir=self.images_base,
                                           suffix=".png")  # find all files from rootdir and subfolders with suffix = ".png"
        self.img_ids_full.sort()
        self.remaining_img_ids = self.img_ids_full.copy()
        self.img_ids_subset = []

        self.files = None
        self.img_ids_for_training = None

        if not self.img_ids_full:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.images_base))
        print("Found %d %s images" % (len(self.img_ids_full), self.split))

        if self.active_list_path != None:
            self.active_img_ids = [i_id.strip() for i_id in open(self.active_list_path)]
            self.active_img_files = [os.path.join(self.images_base, i_id) for i_id in self.active_img_ids]
            self.expand_training_list(self.active_img_files)
        else:
            self.alter_training_list(self.remaining_img_ids)

        print("====== Subset images: ", len(self.img_ids_subset))
        print("====== Remaining images: ", len(self.remaining_img_ids))

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))  # zip: return tuples

    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations != None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, self.files[index]

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        # img = m.imresize(
        #     img, (self.img_size[0], self.img_size[1])
        # )  # uint8 with RGB mode
        img = np.array(img)
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")  # TODO: compare the original and processed ones

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):  # todo: understanding the meaning
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def get_selections(self):
        return self.img_ids_for_training

    def get_remainings(self):
        return self.remaining_img_ids

    def get_subset(self):
        return self.img_ids_subset

    def alter_training_list(self, img_list):
        self.img_ids_for_training = img_list
        self.files = []
        self.files = [i_id for i_id in self.img_ids_for_training]
        # if self.cfg.get('shuffle'):
        #    np.random.shuffle(self.files)

    def expand_training_index(self, selection_index_list):
        for index in selection_index_list:
            self.img_ids_subset.append(self.remaining_img_ids[index])
        for x in self.img_ids_subset:
            if x in self.remaining_img_ids:
                self.remaining_img_ids.remove(x)
        self.alter_training_list(self.img_ids_subset)

    def expand_training_list(self, selection_list):
        for selection in selection_list:
            self.img_ids_subset.append(selection)
        for x in self.img_ids_subset:
            if x in self.remaining_img_ids:
                self.remaining_img_ids.remove(x)
        self.alter_training_list(self.img_ids_subset)