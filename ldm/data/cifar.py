# load Cifar dataset from torchvision
# and preprocess it to be used in the model
import torch
from torchvision import datasets, transforms


# override the cifar dataset get_item method to return a dict with the image and to resize image to a given size
class Cifar10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, size=None):
        # size = config.get("size", 256)
        # root = config.get("root", "./data_cifar10")
        # train = config.get("train", True)

        # resize image to a given size
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.CIFAR10(root, train=train, transform=transform, download=True)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        #print(image.shape)
        return {"image": image.permute(1, 2 ,0), "class_label": label, 'human_label': self.dataset.classes[label]}

    def __len__(self):
        return len(self.dataset)
