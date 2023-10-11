import argparse
import os
import psutil
import yaml
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pathlib


import torch
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
from contextlib import nullcontext
from torch import autocast

from omegaconf import OmegaConf, open_dict
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image

from src.clipseg.models.clipseg import CLIPDensePredT
try:
    from segment_anything import build_sam, SamPredictor
except:
    print("segment_anything not installed")
from sampling_helpers import disabled_train, get_model, _unmap_img, generate_samples
from sampling_helpers import load_model_hf
import json


import sys
import regex as re
from ldm import *
from ldm.models.diffusion.cc_ddim import CCMDDIMSampler

from data.imagenet_classnames import name_map, openai_imagenet_classes

try:
    import open_clip
except:
    print("Install OpenClip via: pip install open_clip_torch")

from utils.DecisionDensenetModel import DecisionDensenetModel
from utils.preprocessor import Normalizer, CropAndNormalizer, ResizeAndNormalizer, GenericPreprocessing, Crop
from utils.vision_language_wrapper import VisionLanguageWrapper
from utils.madry_net import MadryNet
from utils.dino_linear import LinearClassifier, DINOLinear

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def get_classifier(cfg, device):
    if "ImageNet" in cfg.data._target_:
        classifier_name = cfg.classifier_model.name
        if classifier_name == "robust_resnet50":
            classifier_model = MadryNet(cfg.classifier_model.ckpt, device)
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                classifier_model = Crop(classifier_model)
        else:
            classifier_model = getattr(torchvision.models, classifier_name)(pretrained=True)
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                classifier_model = CropAndNormalizer(classifier_model)
    elif "CelebAHQDataset" in cfg.data._target_:
        assert cfg.data.query_label in [20, 31, 39], 'Query label MUST be 20 (Gender), 31 (Smile), or 39 (Age) for CelebAHQ'
        ql = 0
        if cfg.data.query_label in [31, 39]:
            ql = 1 if cfg.data.query_label == 31 else 2
        classifier_model = DecisionDensenetModel(3, pretrained=False,
                                           query_label=ql)
        classifier_model.load_state_dict(torch.load(cfg.classifier_model.classifier_path, map_location='cpu')['model_state_dict'])
        if cfg.classifier_model.classifier_wrapper:
            classifier_model = Normalizer(
                classifier_model,
                [0.5] * 3, [0.5] * 3
            )
    elif "Flowers102" in cfg.data._target_:
        # fine-tuned Dino ViT B/8: https://arxiv.org/pdf/2104.14294.pdf
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(device).eval()
        dim = dino.embed_dim
        linear_classifier = LinearClassifier(dim*cfg.classifier_model.n_last_blocks, 102)
        linear_classifier.load_state_dict(torch.load(cfg.classifier_model.classifier_path, map_location="cpu"), strict=True)
        linear_classifier = linear_classifier.eval().to(device)
        classifier_model = DINOLinear(dino, linear_classifier)
        transforms_list = [transforms.CenterCrop(224), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        classifier_model = GenericPreprocessing(classifier_model, transforms.Compose(transforms_list))
    elif "OxfordIIIPets" in cfg.data._target_:
        # zero-shot OpenClip: https://arxiv.org/pdf/2212.07143.pdf
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
        with open("data/pets_idx_to_label.json", "r") as f:
            pets_idx_to_classname = json.load(f)
        prompts = [f"a photo of a {label}, a type of pet." for label in pets_idx_to_classname.values()]
        classifier_model = VisionLanguageWrapper(model, tokenizer, prompts)
        # try running optimization on 224x224 pixel image
        # transforms_list = [preprocess.transforms[0], preprocess.transforms[1], preprocess.transforms[4]]
        if cfg.classifier_model.classifier_wrapper:
            transforms_list = [preprocess.transforms[1], preprocess.transforms[4]] # CenterCrop(224, 224), Normalize
            classifier_model = GenericPreprocessing(classifier_model, transforms.Compose(transforms_list))
    else:
        raise NotImplementedError
    return classifier_model

def get_dataset(cfg, last_data_idx: int = 0):
    if "ImageNet" in cfg.data._target_:
        out_size = 256
        transform_list = [
            transforms.Resize((out_size, out_size)),
            transforms.ToTensor()
        ]
        transform = transforms.Compose(transform_list)
        dataset = instantiate(cfg.data, start_sample=cfg.data.start_sample, end_sample=cfg.data.end_sample, transform=transform, restart_idx=last_data_idx)
    elif "CelebAHQDataset" in cfg.data._target_:
        dataset = instantiate(
            cfg.data,
            image_size=256, 
            data_dir=cfg.data.data_dir, 
            random_crop=False, 
            random_flip=False, 
            partition='test',
            query_label=cfg.data.query_label,
            normalize=False,
            shard=cfg.data.shard,
            num_shards=cfg.data.num_shards,
            restart_idx=last_data_idx
        )
    elif "Flowers102" in cfg.data._target_:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        dataset = instantiate(
            cfg.data, 
            shard=cfg.data.shard, 
            num_shards=cfg.data.num_shards, 
            transform=transform, 
            restart_idx=last_data_idx
        )
    elif "OxfordIIIPets" in cfg.data._target_: # try running on 224x224 img
        def _convert_to_rgb(image):
            return image.convert('RGB')
        out_size = 256
        transform_list = [
            transforms.Resize((out_size, out_size)),
            # transforms.CenterCrop(out_size),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)
        dataset = instantiate(
            cfg.data, 
            shard=cfg.data.shard, 
            num_shards=cfg.data.num_shards, 
            transform=transform, 
            restart_idx=last_data_idx
        )
    else:
        raise NotImplementedError
    return dataset

@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:
    if "verbose" not in cfg:
        with open_dict(cfg):
            cfg.verbose = True
    if "record_intermediate_results" not in cfg:
        with open_dict(cfg):
            cfg.record_intermediate_results = True

    if "verbose" in cfg and not cfg.verbose:
        blockPrint()

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.chmod(cfg.output_dir, 0o777)
    if "ImageNet" in cfg.data._target_:
        out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.start_sample}_{cfg.data.end_sample}")
    else:
        out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.shard}_{cfg.data.num_shards}")
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o777)
    checkpoint_path = os.path.join(out_dir, "last_saved_id.pth")

    config = {}
    if "ImageNet" in cfg.data._target_:
        run_id = f"{cfg.data.start_sample}_{cfg.data.end_sample}"
    else:
        run_id = f"{cfg.data.shard}_{cfg.data.num_shards}"
    if cfg.resume:
        print("run ID to resume: ", run_id)
    else:
        print("starting new run", run_id)
    config.update(OmegaConf.to_container(cfg, resolve=True))
    print("current run id: ", run_id)
    
    last_data_idx = 0
    if cfg.resume: # or os.path.isfile(checkpoint_path): resume only if asked to, allow restarts
        print(f"resuming from {checkpoint_path}")
        #check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print("checkpoint does not exist! starting from 0 ...")
        else:
            checkpoint = torch.load(checkpoint_path)# torch.load(restored_file.name)
            last_data_idx = checkpoint["last_data_idx"] + 1 if "last_data_idx" in checkpoint else 0
        print(f"resuming from batch {last_data_idx}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # there seems to be a CUDA/autograd instability in gradient computation
    print(f"using device: {device}")

    if "seg_model" in cfg and cfg.seg_model is not None:
        print("### Loading segmentation model ###")
        if "name" in cfg.seg_model and cfg.seg_model.name == "clipseg":
            model_seg = CLIPDensePredT(version=cfg.seg_model.version, reduce_dim=64) #int(cfg.seg_model.version.split('/')[-1]
            model_seg.eval()
            model_seg.load_state_dict(torch.load(cfg.seg_model.path, map_location=torch.device('cpu')), strict=False)
        elif "name" in cfg.seg_model and cfg.seg_model.name == "GD_SAM":
            detect_model = load_model_hf(repo_id=cfg.seg_model.dino.repo_id, filename= cfg.seg_model.dino.filename, dir = cfg.seg_model.dino.dir, ckpt_config_filename = cfg.seg_model.dino.ckpt_config_filename, device=device)
            sam_checkpoint = os.path.join(cfg.pretrained_models_dir, 'sam_vit_h_4b8939.pth')
            model_seg = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    model = get_model(cfg_path=cfg.diffusion_model.cfg_path, ckpt_path = cfg.diffusion_model.ckpt_path).to(device).eval()
    
    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()
    classifier_model.train = disabled_train

    ddim_steps = cfg.ddim_steps
    ddim_eta = cfg.ddim_eta
    scale = cfg.scale #for unconditional guidance
    strength = cfg.strength #for unconditional guidance

    if "seg_model" not in cfg or cfg.seg_model is None or "name" not in cfg.seg_model:
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= None, classifier_wrapper="classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper, record_intermediate_results=cfg.record_intermediate_results, verbose=cfg.verbose, **cfg.sampler)
    elif cfg.seg_model.name == "clipseg":
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= model_seg, classifier_wrapper="classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper, record_intermediate_results=cfg.record_intermediate_results, verbose=cfg.verbose, **cfg.sampler)
    else:
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= model_seg, detect_model = detect_model, classifier_wrapper="classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper, record_intermediate_results=cfg.record_intermediate_results, verbose=cfg.verbose, **cfg.sampler)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * len(sampler.ddim_timesteps))
    assert len(sampler.ddim_timesteps) == ddim_steps, "ddim_steps should be equal to len(sampler.ddim_timesteps)"
    n_samples_per_class = cfg.n_samples_per_class
    batch_size = cfg.data.batch_size
    shuffle = cfg.get("shuffle", False)
      

    #save config to the output directory
    #check if the config file already exists else create a config file
    config_path = os.path.join(out_dir, "config.yaml") 
    if os.path.exists(config_path):
        print("config file already exists! skipping ...")
    else:
        with open(os.path.join(out_dir, "config.yaml"), 'w') as f:
            print("saving config to ", os.path.join(out_dir, "config.yaml  ..."))
            yaml.dump(config, f)
            os.chmod(os.path.join(out_dir, "config.yaml"), 0o555)
    
    #data_path = cfg.data_path
    dataset = get_dataset(cfg, last_data_idx=last_data_idx)
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    if "ImageNet" in cfg.data._target_:
        i2h = name_map
    elif "CelebAHQDataset" in cfg.data._target_:
        # query label 31 (smile): label=0 <-> no smile and label=1 <-> smile
        # query label 39 (age): label=0 <-> old and label=1 <-> young
        assert cfg.data.query_label in [31, 39]
        if 31 == cfg.data.query_label:
            i2h = ["no smile", "smile"]
        elif 39 == cfg.data.query_label:
            i2h = ["old", "young"]
        else:
            raise NotImplementedError
    elif "Flowers102" in cfg.data._target_:
        with open("data/flowers_idx_to_label.json", "r") as f:
            flowers_idx_to_classname = json.load(f)
        flowers_idx_to_classname = {int(k)-1: v for k, v in flowers_idx_to_classname.items()}
        i2h = flowers_idx_to_classname
    elif "OxfordIIIPets" in cfg.data._target_:
        with open("data/pets_idx_to_label.json", "r") as f:
            pets_idx_to_classname = json.load(f)
        i2h = {int(k): v for k, v in pets_idx_to_classname.items()}
    else:
        raise NotImplementedError

    if "ImageNet" in cfg.data._target_:
        with open('data/synset_closest_idx.yaml', 'r') as file:
            synset_closest_idx = yaml.safe_load(file)
    elif "Flowers102" in cfg.data._target_:
        with open("data/flowers_closest_indices.json") as file:
            closest_indices = json.load(file)
        closest_indices = {int(k):v for k,v in closest_indices.items()}
    elif "OxfordIIIPets" in cfg.data._target_:
        with open("data/pets_closest_indices.json") as file:
            closest_indices = json.load(file)
        closest_indices = {int(k):v for k,v in closest_indices.items()}

    if not cfg.resume:
        torch.save({"last_data_idx": -1}, checkpoint_path)
    
    seed = cfg.seed if "seed" in cfg else 0
    set_seed(seed=seed)

    for i, batch in enumerate(data_loader):

        if "fixed_seed" in cfg:
            set_seed(seed=cfg.get("seed", 0)) if cfg.fixed_seed else None
            seed = seed if cfg.fixed_seed else -1
            
        if "return_tgt_cls" in cfg.data and cfg.data.return_tgt_cls:
            image, label, tgt_classes, unique_data_idx = batch
            tgt_classes = tgt_classes.to(device) #squeeze()
        else:
            image, label, unique_data_idx = batch
            if "ImageNet" in cfg.data._target_:
                tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
            elif "CelebAHQDataset" in cfg.data._target_:
                tgt_classes = (1 - label).type(torch.float32)
            elif "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
                tgt_classes = torch.tensor([closest_indices[unique_data_idx[l].item()*cfg.data.num_shards + cfg.data.shard][0] for l in range(label.shape[0])]).to(device)
            else:
                raise NotImplementedError

        image = image.to(device) #squeeze()
        label = label.to(device) #.item() #squeeze()
        #tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
        #tgt_classes = synset_closest_idx[label]
        #tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
        #shuffle tgt_classes
        #random.shuffle(tgt_classes)
        #get classifcation prediction
        with torch.inference_mode():
            #with precision_scope():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = classifier_model(image)
            else:
                logits = sampler.get_classifier_logits(_unmap_img(image)) #converting to -1, 1
            # TODO: handle binary vs multi-class
            if "ImageNet" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                in_class_pred = logits.argmax(dim=1)
                in_confid = logits.softmax(dim=1).max(dim=1).values
                in_confid_tgt =  logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
            else: # binary
                in_class_pred = (logits >= 0).type(torch.int8)
                in_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                in_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
            print("in class_pred: ", in_class_pred, in_confid)
        
        for j, l in enumerate(label):
            print(f"converting {i} from : {i2h[l.item()]} to: {i2h[int(tgt_classes[j].item())]}")
        
        init_image = image.clone() #image.repeat(n_samples_per_class, 1, 1, 1).to(device)
        sampler.init_images = init_image.to(device)
        sampler.init_labels = label # n_samples_per_class * [label]
        if isinstance(cfg.sampler.lp_custom, str) and "dino_" in cfg.sampler.lp_custom:
            if device != next(sampler.distance_criterion.dino.parameters()).device:
                sampler.distance_criterion.dino = sampler.distance_criterion.dino.to(device)
            sampler.dino_init_features = sampler.get_dino_features(sampler.init_images, device=device).clone()
        #mapped_image = _unmap_img(init_image)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(_unmap_img(init_image)))  # move to latent space
        
        if "txt" == model.cond_stage_key: # text-conditional
            if "ImageNet" in cfg.data._target_:
                prompts = [f"a photo of a {openai_imagenet_classes[idx.item()]}." for idx in tgt_classes]
            elif "CelebAHQDataset" in cfg.data._target_:
                # query label 31 (smile): label=0 <-> no smile and label=1 <-> smile
                # query label 39 (age): label=0 <-> old and label=1 <-> young
                assert cfg.data.query_label in [31, 39]
                prompts = []
                for target in tgt_classes:
                    if cfg.data.query_label == 31 and target == 0:
                        attr = "non-smiling"
                    elif cfg.data.query_label == 31 and target == 1:
                        attr = "smiling"
                    elif cfg.data.query_label == 39 and target == 0:
                        attr = "old"
                    elif cfg.data.query_label == 39 and target == 1:
                        attr = "young"
                    else:
                        raise NotImplementedError
                    prompts.append(f"a photo of a {attr} person")
            elif "OxfordIIIPets" in cfg.data._target_:
                # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
                prompts = [f"a photo of a {i2h[idx.item()]}, a type of pet." for idx in tgt_classes]
            elif "Flowers102" in cfg.data._target_:
                # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
                prompts = [f"a photo of a {i2h[idx.item()]}, a type of flower." for idx in tgt_classes]
            else:
                raise NotImplementedError
        else:
            prompts = None
        
        out = generate_samples(
            model, 
            sampler, 
            tgt_classes, 
            ddim_steps, 
            scale, 
            init_latent=init_latent.to(device),
            t_enc=t_enc, 
            init_image=init_image.to(device), 
            ccdddim=True, 
            latent_t_0=cfg.get("latent_t_0", False),
            prompts=prompts, 
            seed=seed,
        )

        all_samples = out["samples"]
        all_videos = out["videos"] 
        all_probs = out["probs"]
        all_masks = out["masks"] 
        all_cgs = out["cgs"]

        with torch.inference_mode():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = classifier_model(all_samples[0])
            else:
                logits = sampler.get_classifier_logits(_unmap_img(all_samples[0])) #converting to -1, 1 (it is converted back in the function)
            if "ImageNet" in cfg.data._target_ or "CUB" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                out_class_pred = logits.argmax(dim=1)
                out_confid = logits.softmax(dim=1).max(dim=1).values
                out_confid_tgt = logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
            else: # binary
                out_class_pred = (logits >= 0).type(torch.int8)
                out_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                out_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
            print("out class_pred: ", out_class_pred, out_confid)
            print(out_confid_tgt)

        # Loop through your data and update the table incrementally
        for j in range(batch_size):
            # Generate data for the current row
            src_image = copy.deepcopy(sampler.init_images[j].cpu()) #all_samples[j][0])
            gen_image = copy.deepcopy(all_samples[0][j].cpu())
            class_prediction = copy.deepcopy(all_probs[0][j]) if all_probs is not None else out_confid[j] # all_probs[j]
            
            source = i2h[label[j].item()]
            target = i2h[int(tgt_classes[j].item())]
            in_pred_cls = i2h[in_class_pred[j].item()]
            out_pred_cls = i2h[out_class_pred[j].item()]

            #diff =  (init_image - all_samples[j][1:])
            diff = sampler.init_images[j]-all_samples[0][j]   
            lp1 = int(torch.norm(diff, p=1, dim=-1).mean().cpu().numpy())
            lp2 = int(torch.norm(diff, p=2, dim=-1).mean().cpu().numpy())
            #print(f"lp1: {lp1}, lp2: {lp2}")

            data_dict = {
                "unique_id": unique_data_idx[j].item(), 
                "image": src_image, 
                "source": source, 
                "target": target, 
                "gen_image": gen_image,
                "target_confidence": class_prediction, 
                "in_pred": in_pred_cls, 
                "out_pred": out_pred_cls, 
                "out_confid": out_confid[j].cpu().item(), 
                "out_tgt_confid": out_confid_tgt[j].cpu().item(), 
                "in_confid": in_confid[j].cpu().item(), 
                "in_tgt_confid": in_confid_tgt[j].cpu().item(), 
                "closness_1": lp1, 
                "closness_2": lp2,
            }
            if cfg.record_intermediate_results:
                if all_videos is not None:
                    video_results = {
                        "video": (255. * all_videos[0][j]).to(torch.uint8).cpu(), 
                    }
                    data_dict = dict(data_dict, **video_results)
                if all_cgs is not None:
                    cgs_results = {
                        "cgs": (255.*all_cgs[0][j]).to(torch.float32).cpu(),
                    }
                    data_dict = dict(data_dict, **cgs_results)

            if "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
                uidx = unique_data_idx[j].item()*cfg.data.num_shards + cfg.data.shard
            else:
                uidx = unique_data_idx[j].item()

            dict_save_path = os.path.join(out_dir, f'{str(uidx).zfill(5)}.pth')
            torch.save(data_dict, dict_save_path)
            os.chmod(dict_save_path, 0o555)

            pathlib.Path(os.path.join(out_dir, 'original')).mkdir(parents=True, exist_ok=True, mode=0o777)
            os.chmod(os.path.join(out_dir, 'original'), 0o777)
            pathlib.Path(os.path.join(out_dir, 'counterfactual')).mkdir(parents=True, exist_ok=True, mode=0o777)
            os.chmod(os.path.join(out_dir, 'counterfactual'), 0o777)
            orig_save_path = os.path.join(out_dir, 'original', f'{str(uidx).zfill(5)}.png')
            save_image(src_image.clip(0, 1), orig_save_path)
            os.chmod(orig_save_path, 0o555)

            cf_save_path = os.path.join(out_dir, 'counterfactual', f'{str(uidx).zfill(5)}.png')
            save_image(gen_image.clip(0, 1), cf_save_path)
            os.chmod(cf_save_path, 0o555)

        if (i + 1) % cfg.log_rate == 0:
            last_data_idx = unique_data_idx[-1].item()
            torch.save({
                #"table": copy.deepcopy(my_table),
                "last_data_idx": last_data_idx,
            }, checkpoint_path)
            os.chmod(checkpoint_path, 0o777)
            print(f"saved {checkpoint_path}, with data_id {last_data_idx}")

        del out
            
    return None

if __name__ == "__main__":
    main()
