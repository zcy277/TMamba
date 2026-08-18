import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class PSTSeg(data.Dataset):

    def __init__(self, cfg, root, mode='trainval', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # self.dp_to_tensor = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        # ])
        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.root = root
        self.n_classes = cfg['n_classes']

        scales_array = cfg['scales_array'] # New: discrete scales list to align with CFDHI-Net
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            # ColorJitter(
            #     brightness=cfg['brightness'],
            #     contrast=cfg['contrast'],
            #     saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scales_array),
            RandomCrop(crop_size, pad_if_needed=True)
        ])


        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.4537, 44.2457, 31.6650, 46.4071, 30.1391])
            self.binary_class_weight = np.array([1.4507, 21.5033])
        else:
            raise (f"{cfg['class_weight']} not support.")

        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()


        # image = Image.open(os.path.join(self.root, 'rgb_resize', image_path + '.png'))
        # depth = Image.open(os.path.join(self.root, 'thermal_resize', image_path + '.png')).convert('RGB')
        # label = Image.open(os.path.join(self.root, 'labels_resize', image_path + '.png'))
        image = Image.open(os.path.join(self.root, 'rgb', image_path + '.png'))
        depth = Image.open(os.path.join(self.root, 'thermal', image_path + '.png')).convert('RGB')
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
        }

        if self.mode in ['train', 'trainval'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['label_path'] = image_path.strip().split('/')[-1] + '.png'
        return sample

    @property
    def cmap(self):
        return [
            [0, 0, 0], # background
            [0, 0, 255], # fire_extinguisher
            [0, 255, 0], # backpack
            [255, 0, 0], # drill
            [255, 255, 255], # survivor/rescue_randy
        ]

    @property
    def tcsvt2023_DSGBINet_cmap(self):
        return [
            [0, 0, 0], # background
            [64, 0, 128], # fire_extinguisher
            [64, 64, 0], # backpack
            [0, 128, 192], # drill
            [0, 0, 192], # survivor/rescue_randy
        ]



