# load CALTECH101 dataset from torchvision
# and preprocess it to be used in the model
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# override the cifar dataset get_item method to return a dict with the image and to resize image to a given size
class Caltech101(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, size=256):
        # size = config.get("size", 256)
        # root = config.get("root", "./data_cifar10")
        # train = config.get("train", True)

        # resize image to a given size
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            #transform to make the image with 3 channels in case it is grayscale
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.Caltech101(root, transform=transform, download=True)
        #CIFAR10(root, train=train, transform=transform, download=True)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        #print(image.shape)
        #get the human label of the image



        return {"image": image.permute(1, 2 ,0), "class_label": label, 'human_label': self.dataset.categories[label]}

    def __len__(self):
        return len(self.dataset)


# start the main part
if __name__ == "__main__":
    # load the dataset
    dataset = Caltech101(root="./data_caltech101", train=True, size=256)
    # visualize 100 images from the dataset
    for i in range(10000):
        img = dataset[i]["image"]
        print(img.shape)
        # show the image using pillow
        # import PIL
        # #scale the image to 0-255
        # img = (img + 1) / 2
        # # fix this error PIL TypeError: Cannot handle this data type: (1, 1, 3), <i8
        # img = img * 255
        # img = img.numpy().astype('uint8')
        # PIL.Image.fromarray(img).show()