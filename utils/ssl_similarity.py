import torch
from torch import nn
from tqdm import tqdm
import torchvision
from torchvision import models, transforms
import os
import json
import random

from data.datasets import CUB

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1):
        """
        Input:
            x1: first views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        return self.encoder(x1)

def get_simsiam_dist(weights_path):
    model = SimSiam(models.resnet50, dim=2048, pred_dim=512)
    state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
    model.load_state_dict(
        {k[7:]: v for k, v in state_dict.items()}
    )
    return model

def _convert_to_rgb(image):
    return image.convert('RGB')

def get_dataset(dataset, data_dir, pkl_paths=None):
    if "flowers" == dataset:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        target_transform = lambda x: x-1
        dataset = torchvision.datasets.Flowers102(root=data_dir, split="test", transform=transform, target_transform=target_transform, download=True)
    elif "pets" == dataset:
        out_size = 224
        transform_list = [
            transforms.Resize((out_size, out_size)),
            transforms.CenterCrop(out_size),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = transforms.Compose(transform_list)
        dataset = torchvision.datasets.OxfordIIITPet(root=data_dir, split="test", target_types="category", transform=transform, download=True)
    elif "cub" == dataset:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = CUB(
            pkl_file_paths=[pkl_paths],
            use_attr=False,
            no_img=True,
            uncertain_label=False,
            image_dir=data_dir,
            n_class_attr=-1,
            transform=transform,
            shard=0,
            num_shards=1,
            return_idx=False
        )
    
    return dataset

@torch.inference_mode()
def main(args):
    device = torch.device('cuda')
    
    # compute SSL features
    model = get_simsiam_dist(args.weights_path)
    model.to(device)
    model.eval()

    dataset = get_dataset(args.dataset, args.data_dir, args.pkl_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=16, pin_memory=True)
    ssl_features, targets = [], []
    for image, target in tqdm(loader, leave=False):
        ssl_feat = model(image.to(device))
        ssl_features.append(ssl_feat.detach().cpu())
        targets.append(target)

    ssl_features = torch.concat(ssl_features, dim=0).to(device)
    targets = torch.concat(targets, dim=0).to(device)
        
    # find closest pairs using cosine similarity
    random.seed(0)
    closest_pairs = {idx: [] for idx in range(len(dataset))}
    for idx in tqdm(range(len(dataset)), leave=False):
        idx_feat = ssl_features[idx]
        cosine_similarities = (idx_feat @ ssl_features.T) / (torch.norm(idx_feat, p=2, dim=-1)*torch.norm(ssl_features, p=2, dim=-1))
        available_indices = targets != targets[idx]
        for closest_idx in cosine_similarities.argsort(descending=True):
            if available_indices[closest_idx] and targets[closest_idx.item()].cpu().item() not in closest_pairs[idx]:
                closest_pairs[idx].append(targets[closest_idx.item()].cpu().item())
            if len(closest_pairs[idx]) == args.n_closest:
                random.shuffle(closest_pairs[idx])
                break

    # save it
    save_path = os.path.join(args.save_dir, f"{args.dataset}_closest_indices.json")
    with open(save_path, "w") as f:
        json.dump(closest_pairs, f, indent=4)
    return



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SSL similarity script.')
    parser.add_argument('--save-dir', required=True, type=str,
                        help='Save dir')
    parser.add_argument('--weights-path', default='pretrained_models/checkpoint_0099.pth.tar', type=str,
                        help='ResNet50 SimSiam model weights')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--dataset', default="pets", type=str, choices=["pets", "cub", "flowers"], required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--pkl-path', type=str, required=False)
    parser.add_argument('--n-closest', type=int, default=5, required=False)

    main(parser.parse_args())

    