import torch
from torch import nn, Tensor
from typing import Tuple
from collections import OrderedDict
import torchvision

class ImageNormalizer(nn.Module):
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ) -> None:
        super(ImageNormalizer, self).__init__()
        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        print("Madry input shape", input.shape)
        return (input - self.mean) / self.std

def normalize_model(
    model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]
) -> nn.Module:
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return nn.Sequential(layers)

def Engstrom2019RobustnessNet(device):
    """def __init__(self):
        #super(Engstrom2019RobustnessNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        model_pt = model_class_dict['pt_resnet'](pretrained=False)
        #super
        #self.model = nn.DataParallel(model_pt.cuda().eval())
        self.model = model_pt.cuda().eval()
        self.model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()
    def forward(self, x, return_features=False):
        x = (x - self.mu) / self.sigma
        #return super(Engstrom2019RobustnessNet, self).forward(x, return_features=return_features)
        return self.model(x)
    def __call__(self, x):
        return self.forward(x)"""
    model_pt = torchvision.models.resnet50(pretrained=False).cuda(device)
    model_pt.eval()
    # model_pt.load_state_dict(ckpt, strict=True)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalized_model = normalize_model(model=model_pt, mean=mean, std=std)
    normalized_model.eval()
    print("model created")
    normalized_model.cuda(device)
    return normalized_model

def load_model(modelname, norm="other", device="cuda"):
    model_det = {"model": Engstrom2019RobustnessNet,
    "data": "imagenet_l2_3_0.pt",
    "ckpt_var": None,}

    model = model_det["model"](device)
    print("model loaded")
    print(type(model))
    return model

def MadryNet(ckpt, device):
    norm = "l2"
    model = load_model(
        modelname="Engstrom2019Robustness", norm=norm, device=device
    )
    state_dict = torch.load(ckpt, map_location="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    return model