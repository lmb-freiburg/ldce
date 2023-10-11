import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import tarfile
import urllib.request

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from fastai.vision.all import *

import urllib.request


def reverse_dict(dict_to_reverse):
    return {v: k for k, v in dict_to_reverse.items()}

class ImageNette(Dataset):

    NAMES = {"full": "imagenette2", "320": "imagenette2-320", "160": "imagenette2-160"}
    URLS = {"full": URLs.IMAGENETTE, "320": URLs.IMAGENETTE_320, "160": URLs.IMAGENETTE_160}

    def __init__(self, id="full", process_images=True, data_root=None, random_crop=False, stage = "train", size = 256):
        self.id = id
        self.NAME = self.NAMES[id]
        self.URL = self.URLS[id]
        self.stage = stage
        self.size = size
        self.process_images = process_images
        self.random_crop = random_crop
        self.data_root = data_root
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def download_and_extract_tar(self, file_url, save_dir):
        # Create directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get file name from URL
        file_name = file_url.split("/")[-1]
        file_path = os.path.join(save_dir, file_name)

        path = os.path.join(save_dir, file_name.split(".")[0])
        if os.path.exists(path):
            print("File already exists. Skipping download.")
        else:
            print("Downloading file...")
            # Download the file
            urllib.request.urlretrieve(file_url, file_path)

            # Extract the tar file
            with tarfile.open(file_path) as tar:
                tar.extractall(path=save_dir)

            # Delete the tar file
            os.remove(file_path)
            #return the path to the extracted folder
            #create a PosixPath object
        return Path(path)



    def _prepare(self):

        #self.root = untar_data(self.URL, base = self.data_root)
        #download the data from self.URL
        
        self.root = self.download_and_extract_tar(self.URL, self.data_root)
        self.datadir = self.root/self.stage
        # get the synsets as a dict (class_id:synset) sorted ascendingly from the folder names in the datadir
        self.idx2syn = {int(k): v for k, v in enumerate(sorted(os.listdir(self.datadir)))}
        self.file_list = get_image_files(self.datadir)
        self.syn2idx = reverse_dict(self.idx2syn)

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)
        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        #pick available synsets in the syn2idx dict
        self.syn2h = {k: v for k, v in human_dict.items() if k in self.syn2idx.keys()}
        self.h2syn = reverse_dict(self.syn2h)

    def _prepare_human_to_integer_label(self):
        """
        We have synset to human and synset to integer label get the human to integer label
        """
        self.h2idx = {k: self.syn2idx[v] for k, v in self.h2syn.items()}
        self.idx2h = reverse_dict(self.h2idx)

    def _load(self):

        #loop over the files and get the synset from the PosixPath
        self.synsets =  [p.parts[-2] for p in self.file_list]
        self.abspaths = self.file_list
        #convert to str instead of PosixPath
        self.abspaths = [str(p) for p in self.abspaths]
        self.relpaths = [str(os.path.relpath(p, start = self.datadir)) for p in self.abspaths]



        self.class_labels = [self.syn2idx[s] for s in self.synsets]
        self.human_labels = [self.syn2h[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }

        if self.process_images:
            self.data = ImagePaths(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   random_crop=self.random_crop,
                                   )
        else:
            self.data = self.abspaths


if __name__ == "__main__":
    dataset = ImageNette(id="320", process_images=True, data_root="./data", random_crop=False, stage="train", size=256)
    print("Number of images in the dataset:", len(dataset))
    sample = dataset[0]
    print("Data type of the first sample:", type(sample))
    #create and show the image in the first sample convert the range of the image from -1 to 1 to 0 to 255
    img = Image.fromarray(((sample["image"] + 1) * 127.5).astype(np.uint8), "RGB")
    img.show()