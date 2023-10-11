from typing import Any
from torchvision import datasets, transforms
from data.imagenet_classnames import name_map, folder_label_map
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from pathlib import Path
import random
import pickle
import linecache


class ImageNet(datasets.ImageFolder):
    classes = [name_map[i] for i in range(1000)]
    name_map = name_map

    def __init__(
            self, 
            root:str, 
            split:str="val", 
            transform=None, 
            target_transform=None, 
            class_idcs=None, 
            start_sample: float = 0., 
            end_sample: int = 50000//1000,
            return_tgt_cls: bool = False,
            idx_to_tgt_cls_path = None,
            restart_idx: int = 0, 
            **kwargs
    ):
        _ = kwargs  # Just for consistency with other datasets.
        print(f"Loading ImageNet with start_sample={start_sample}, end_sample={end_sample} ")
        assert split in ["train", "val"]
        assert start_sample < end_sample and start_sample >= 0 and end_sample <= 50000//1000
        self.start_sample = start_sample

        assert 0 <= restart_idx < 50000
        self.restart_idx = restart_idx

        path = root if root[-3:] == "val" or root[-5:] == "train" else os.path.join(root, split)
        super().__init__(path, transform=transform, target_transform=target_transform)
        
        with open(idx_to_tgt_cls_path, 'r') as file:
            idx_to_tgt_cls = yaml.safe_load(file)
            if isinstance(idx_to_tgt_cls, dict):
                idx_to_tgt_cls = [idx_to_tgt_cls[i] for i in range(len(idx_to_tgt_cls))]
        self.idx_to_tgt_cls = idx_to_tgt_cls

        self.return_tgt_cls = return_tgt_cls

        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            samples = []
            idx_to_tgt_cls = []
            for i, (p, t) in enumerate(self.samples):
                if t in tgt_to_tgt_map:
                    samples.append((p, tgt_to_tgt_map[t]))
                    idx_to_tgt_cls.append(self.idx_to_tgt_cls[i])
            
            self.idx_to_tgt_cls = idx_to_tgt_cls
            #self.samples = [(p, tgt_to_tgt_map[t]) for i, (p, t) in enumerate(self.samples) if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

        if "val" == split: # reorder
            new_samples = []
            idx_to_tgt_cls = []
            for idx in range(50000//1000):
                new_samples.extend(self.samples[idx::50000//1000])
                idx_to_tgt_cls.extend(self.idx_to_tgt_cls[idx::50000//1000])
            self.samples = new_samples[int(start_sample*1000):end_sample*1000]
            self.idx_to_tgt_cls = idx_to_tgt_cls[int(start_sample*1000):end_sample*1000]

        else:
            raise NotImplementedError
        
        if self.restart_idx > 0:
            self.samples = self.samples[self.restart_idx:]
            self.idx_to_tgt_cls = self.idx_to_tgt_cls[self.restart_idx:]

        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.return_tgt_cls:
            return *sample, self.idx_to_tgt_cls[index], index + self.start_sample*1000 + self.restart_idx
        else:
            return sample, index + self.start_sample*1000 + self.restart_idx
        

class ImageNetSelect(datasets.ImageFolder):
    classes = [name_map[i] for i in range(1000)]
    name_map = name_map

    def __init__(
            self, 
            root:str, 
            split:str="val", 
            transform=None, 
            target_transform=None, 
            image_idcs=None,
            class_idcs=None, 
            start_sample: float = 0., 
            end_sample: int = 50000,
            return_tgt_cls: bool = False,
            lbl_to_tgt_cls_map = None,
            restart_idx: int = 0, 
            **kwargs
    ):
        _ = kwargs  # Just for consistency with other datasets.
        print(f"Loading ImageNet with start_sample={start_sample}, end_sample={end_sample} ")
        assert split in ["train", "val"]
        assert start_sample < end_sample and start_sample >= 0 and end_sample <= 50000
        self.start_sample = start_sample

        assert 0 <= restart_idx < 50000
        self.restart_idx = restart_idx

        path = root if root[-3:] == "val" or root[-5:] == "train" else os.path.join(root, split)
        super().__init__(path, transform=transform, target_transform=target_transform)
        
        self.lbl_to_tgt_cls_map = lbl_to_tgt_cls_map
        self.return_tgt_cls = return_tgt_cls

        #get out the classes desired
        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            self.classes = [self.classes[c] for c in class_idcs]
            samples = []
            for i, (p, t) in enumerate(self.samples):
                if t in class_idcs:
                    samples.append((p, t))
            self.samples = samples
            
        if "val" == split: # reorder
            new_samples = []
            if image_idcs is None:
                new_samples = self.samples[int(start_sample):end_sample]
            else:
                new_samples = [self.samples[i] for i in image_idcs]
            self.samples = new_samples

        else:
            raise NotImplementedError
        
        if self.restart_idx > 0:
            self.samples = self.samples[self.restart_idx:]

        self.class_labels = {i: folder_label_map[folder] for i, folder in zip(class_idcs, self.classes)}
        self.targets = np.array(self.samples)[:, 1]
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.return_tgt_cls:
            return *sample, self.lbl_to_tgt_cls_map[sample[-1]], index + self.start_sample + self.restart_idx
        else:
            return sample

class CelebADataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
        restart_idx: int = 0,
    ):
        partition_df = pd.read_csv(os.path.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

        self.restart_idx = restart_idx
        if self.restart_idx > 0:
            print("TODO")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['image_id']

        with open(os.path.join(self.data_dir, 'img_align_celeba', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, labels
        else:
            return img, {}


class CelebAHQDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
        restart_idx: int = 0,
        **kwargs
    ):
        from io import StringIO
        # read annotation files
        with open(os.path.join(data_dir, 'CelebAMask-HQ-attribute-anno.txt'), 'r') as f:
            datastr = f.read()[6:]
            datastr = 'idx ' +  datastr.replace('  ', ' ')

        with open(os.path.join(data_dir, 'CelebA-HQ-to-CelebA-mapping.txt'), 'r') as f:
            mapstr = f.read()
            mapstr = [i for i in mapstr.split(' ') if i != '']

        mapstr = ' '.join(mapstr)

        data = pd.read_csv(StringIO(datastr), sep=' ')
        partition_df = pd.read_csv(os.path.join(data_dir, 'list_eval_partition.csv'))
        mapping_df = pd.read_csv(StringIO(mapstr), sep=' ')
        # mapping_df.rename(columns={'orig_file': 'image_id'}, inplace=True)
        partition_df = pd.merge(mapping_df, partition_df, on='idx')

        self.data_dir = data_dir

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['split'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

        self.restart_idx = restart_idx
        if self.restart_idx > 0:
            self.data = self.data.iloc[self.restart_idx:]
            self.data.reset_index(inplace=True)
            self.data.replace(-1, 0, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['idx']

        with open(os.path.join(self.data_dir, 'CelebA-HQ-img', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels, self.restart_idx + idx

        if self.class_cond:
            return img, labels, self.restart_idx + idx
        else:
            return img, {}, self.restart_idx + idx
        
class CUB(Dataset):
    # Implementation from https://github.com/JonathanCrabbe/CARs

    N_ATTRIBUTES = 312
    N_CLASSES = 200
    attribute_map = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59,
                     63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131,
                     132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193,
                     194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240,
                     242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304,
                     305, 308, 309, 310, 311]

    def __init__(
            self, 
            pkl_file_paths, 
            use_attr, 
            no_img, 
            uncertain_label, 
            image_dir, 
            n_class_attr, 
            transform=None, 
            shard=0,
            num_shards=1,
            return_idx=True,
            **kwargs,
    ):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.return_idx = return_idx

        self.data = self.data[shard::num_shards]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_data = self.data[index]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 1:])
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((self.N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(self.N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            if self.return_idx:
                return img, class_label, index
            else:
                return img, class_label

    def get_raw_image(self, idx: int,  resol: int = 299):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 1:])
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')
        center_crop = transforms.Resize((resol, resol))
        return center_crop(img)

    def class_name(self, class_id) -> str:
        """
        Get the name of a class
        Args:
            class_id: integer identifying the concept

        Returns:
            String corresponding to the concept name
        """
        class_path = Path(self.image_dir) / "classes.txt"
        name = linecache.getline(str(class_path), class_id+1)
        name = name.split(".")[1]  # Remove the line number
        name = name.replace("_", " ")  # Put spacing in class names
        name = name[:-1]  # Remove breakline character
        return name.title()

    def get_class_names(self):
        """
        Get the name of all concepts
        Returns:
            List of all concept names
        """
        return [self.class_name(i) for i in range(self.N_CLASSES)]

class Flowers102(Dataset):
    def __init__(self, root, transform, shard: int = 0, num_shards: int = 1, **kwargs) -> None:
        super().__init__()
        target_transform = lambda x: x-1 # flowers starts from idx 1
        self.dataset = datasets.Flowers102(root=root, split="test", transform=transform, target_transform=target_transform, download=True)

        # compute shards
        self.dataset._image_files = self.dataset._image_files[shard::num_shards]
        self.dataset._labels = self.dataset._labels[shard::num_shards]
    
    def __getitem__(self, index: Any) -> Any:
        img, label = self.dataset.__getitem__(index)
        return img, label, index
    
    def __len__(self):
        return len(self.dataset)
    
class OxfordIIIPets(Dataset):
    def __init__(self, root, transform, shard: int = 0, num_shards: int = 1, **kwargs) -> None:
        super().__init__()
        self.dataset = datasets.OxfordIIITPet(root=root, split="test", target_types="category", transform=transform, download=True)

        # compute shards
        self.dataset._images = self.dataset._images[shard::num_shards]
        self.dataset._labels = self.dataset._labels[shard::num_shards]
    
    def __getitem__(self, index: Any) -> Any:
        img, label = self.dataset.__getitem__(index)
        return img, label, index
    
    def __len__(self):
        return len(self.dataset)