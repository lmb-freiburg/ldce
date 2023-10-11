import torch
import torchvision
import torchvision.transforms.functional as F

class Normalizer(torch.nn.Module):
    '''
    normalizing module. Useful for computing the gradient
    to a x image (x in [0, 1]) when using a classifier with
    different normalization inputs (i.e. f((x - mu) / sigma))
    '''
    def __init__(self, classifier,
                 mu=[0.485, 0.456, 0.406],
                 sigma=[0.229, 0.224, 0.225]):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mu', torch.tensor(mu).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor(sigma).view(1, -1, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.classifier(x)

class Crop(torch.nn.Module):
    def __init__(self, classifier, crop_size: int=224) -> None:
        super().__init__()
        self.classifier = classifier
        self.crop_size = crop_size
        self.center_crop = torchvision.transforms.CenterCrop(crop_size)

    def forward(self, x):
        # assumes x in [0, 1]!
        x = self.center_crop(x)
        return self.classifier(x)

class CropAndNormalizer(torch.nn.Module):
    def __init__(self, classifier, crop_size: int=224, mu=[0.485, 0.456, 0.406], sigma=[0.229, 0.224, 0.225]) -> None:
        super().__init__()
        self.classifier = classifier
        self.crop_size = crop_size
        self.center_crop = torchvision.transforms.CenterCrop(crop_size)
        self.register_buffer('mu', torch.tensor(mu).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor(sigma).view(1, -1, 1, 1))

    def forward(self, x):
        # assumes x in [0, 1]!
        # x = F.center_crop(x, self.crop_size)
        x = self.center_crop(x)
        x = (x - self.mu) / self.sigma
        return self.classifier(x)
    

class ResizeAndNormalizer(torch.nn.Module):
    def __init__(self, classifier, resolution: tuple=(224, 224), mu=[0.485, 0.456, 0.406], sigma=[0.229, 0.224, 0.225]) -> None:
        super().__init__()
        self.classifier = classifier
        self.resolution = resolution
        self.resize = torchvision.transforms.Resize(resolution)
        self.register_buffer('mu', torch.tensor(mu).view(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor(sigma).view(1, -1, 1, 1))

    def forward(self, x):
        # assumes x in [0, 1]!
        x = self.resize(x)
        x = (x - self.mu) / self.sigma
        return self.classifier(x)

class GenericPreprocessing(torch.nn.Module):
    def __init__(self, classifier, preprocessor) -> None:
        super().__init__()
        self.classifier = classifier
        self.preprocessor = preprocessor

    def forward(self, x):
        # assumes x in [0, 1]!
        x = self.preprocessor(x)
        return self.classifier(x)