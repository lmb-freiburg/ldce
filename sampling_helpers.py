import time
from itertools import islice

import PIL
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch import distributions as torchd
from torch.nn import functional as F
import random

from ldm.util import instantiate_from_config
from huggingface_hub import hf_hub_download

try:
    from groundingdino.util import box_ops
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.models import build_model
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util.inference import predict
except:
    print("Warning install grounding dino!")


# sys.path.append(".")
# sys.path.append('./taming-transformers')

def load_model_hf(repo_id, filename, dir, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=dir)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model.to(device)


# detect object using grounding DINO
def detect(image, text_prompt, model, box_threshold=0.4, text_threshold=0.4, image_source=None):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    sorted_indices = torch.argsort(logits, descending=True)
    sorted_boxes = boxes[sorted_indices]
    return sorted_boxes


def segment(image, sam_model, boxes):
    sam_model.set_image(image)
    H, W, _ = image.shape
    if boxes.shape[0] == 0:
        boxes_xyxy = torch.tensor([[0, 0, W, H]]).to(sam_model.device)
    else:
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(sam_model.device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    if boxes.shape[0] == 0:
        masks = ~masks
    return masks.to(sam_model.device)


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# @title loading utils
def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    #model#.to(device)  # .cuda()
    model.eval()
    return model


def get_model(cfg_path="configs/latent-diffusion/cin256-v2.yaml", ckpt_path="models/ldm/cin256-v2/model.ckpt"):
    config = OmegaConf.load(cfg_path)
    model = load_model_from_config(config, ckpt_path)
    return model


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((256, 256), resample=PIL.Image.LANCZOS)  # ((w, h), resample=PIL.Image.LANCZOS)
    pil_iamge = image
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image, pil_iamge


def compute_lp_dist(x, y, p: int):
    diff = x - y 
    diff_abs_flat = diff.abs().view(diff.shape[0], -1)
    if p == 1.0:
        lp_dist = torch.sum(diff_abs_flat, dim=1)
    else:
        lp_dist = torch.sum(diff_abs_flat ** p, dim=1)
    return lp_dist


def compute_lp_gradient(diff, p, small_const=1e-12):
    if p < 1:
        grad_temp = (p * (diff.abs() + small_const) ** (

                p - 1)) * diff.sign()
    else:
        grad_temp = (p * diff.abs() ** (p - 1)) * diff.sign()
    return grad_temp


def _renormalize_gradient(grad, eps, small_const=1e-22):
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    grad_norm = torch.where(grad_norm < small_const, grad_norm + small_const, grad_norm)
    grad /= grad_norm
    grad *= eps.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    return grad, grad_norm


def renormalize(a, b, small_const=1e-22):
    # changes(removed detach and restored where)
    a_norm = a.view(a.shape[0], -1).norm(p=2, dim=1).view(b.shape[0], 1, 1, 1)
    a_norm_new = torch.where(a_norm < small_const, a_norm + small_const,
                             a_norm)  # torch.clamp(a_norm, min=small_const) #.detach() #torch.where(a_norm < small_const, a_norm + small_const, a_norm)
    a /= a_norm_new
    a *= b.view(a.shape[0], -1).norm(p=2, dim=1).view(a.shape[0], 1, 1, 1)
    return a, a_norm_new


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):

    def __init__(self, logits=None, probs=None, validate_args=None):
        super().__init__(logits=logits, probs=probs, validate_args=validate_args)

    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


def cone_project(grad_temp_1, grad_temp_2, deg, orig_shp):
    """
    grad_temp_1: gradient of the loss w.r.t. the robust/classifier free
    grad_temp_2: gradient of the loss w.r.t. the non-robust
    projecting the robust/CF onto the non-robust
    """
    angles_before = torch.acos(
        (grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_1.norm(p=2, dim=1) * grad_temp_2.norm(p=2, dim=1)))
    grad_temp_2 /= grad_temp_2.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
    grad_temp_1 = grad_temp_1 - ((grad_temp_1 * grad_temp_2).sum(1) / (grad_temp_2.norm(p=2, dim=1) ** 2)).view(
        grad_temp_1.shape[0], -1) * grad_temp_2
    grad_temp_1 /= grad_temp_1.norm(p=2, dim=1).view(grad_temp_1.shape[0], -1)
    radians = torch.tensor([deg], device=grad_temp_1.device).deg2rad()
    cone_projection = grad_temp_1 * torch.tan(radians) + grad_temp_2

    # second classifier is a non-robust one -
    # unless we are less than 45 degrees away - don't cone project
    #print(" ratio of dimensions that are cone projected: ", (angles_before > radians).float().mean())
    #print("angle before", angles_before.mean(), angles_before.std(), angles_before.min(), angles_before.max())
    #print("radians", radians)
    grad_temp = grad_temp_2.clone()
    loop_projecting = time.time()
    grad_temp[angles_before > radians] = cone_projection[angles_before > radians]

    return grad_temp


def cone_project_chuncked(grad_temp_1, grad_temp_2, deg, orig_shp, chunk_size = 1):
    """
    grad_temp_1: gradient of the loss w.r.t. the robust/classifier free
    grad_temp_2: gradient of the loss w.r.t. the non-robust
    projecting the robust/CF onto the non-robust
    """
    grad_temp_1_chuncked = grad_temp_1.view(*orig_shp) \
    .unfold(2, chunk_size, chunk_size) \
    .unfold(3, chunk_size, chunk_size) \
    .permute(0, 1, 4, 5, 2, 3) \
    .reshape(orig_shp[0], -1, orig_shp[-2]//chunk_size, orig_shp[-1]//chunk_size) \
    .permute(0, 2, 3, 1)
    
    grad_temp_2_chuncked = grad_temp_2.view(*orig_shp) \
    .unfold(2, chunk_size, chunk_size) \
    .unfold(3, chunk_size, chunk_size) \
    .permute(0, 1, 4, 5, 2, 3) \
    .reshape(orig_shp[0], -1, orig_shp[-2]//chunk_size, orig_shp[-1]//chunk_size) \
    .permute(0, 2, 3, 1)
   
    angles_before_chuncked = torch.acos((grad_temp_1_chuncked * grad_temp_2_chuncked).sum(-1) / (grad_temp_1_chuncked.norm(p=2, dim=-1) * grad_temp_2_chuncked.norm(p=2, dim=-1)))
    #print('angle before', angles_before_chuncked)
    grad_temp_2_chuncked_norm = grad_temp_2_chuncked / grad_temp_2_chuncked.norm(p=2, dim=-1).view(grad_temp_1_chuncked.shape[0], grad_temp_1_chuncked.shape[1], grad_temp_1_chuncked.shape[1], -1)
    #print(f" norm {grad_temp_2_chuncked_norm.norm(p=2, dim=-1) ** 2}")
    grad_temp_1_chuncked = grad_temp_1_chuncked - ((grad_temp_1_chuncked * grad_temp_2_chuncked_norm).sum(-1) / (grad_temp_2_chuncked_norm.norm(p=2, dim=-1) ** 2)).view(
         grad_temp_1_chuncked.shape[0], grad_temp_1_chuncked.shape[1], grad_temp_1_chuncked.shape[1], -1) * grad_temp_2_chuncked_norm

    grad_temp_1_chuncked_norm = grad_temp_1_chuncked / grad_temp_1_chuncked.norm(p=2, dim=-1).view(grad_temp_1_chuncked.shape[0], grad_temp_1_chuncked.shape[1], grad_temp_1_chuncked.shape[1], -1)
    radians = torch.tensor([deg], device=grad_temp_1_chuncked.device).deg2rad()
    cone_projection = grad_temp_2_chuncked.norm(p=2, dim=-1).unsqueeze(-1) * grad_temp_1_chuncked_norm * torch.tan(radians) + grad_temp_2_chuncked

    # second classifier is a non-robust one -
    # unless we are less than 45 degrees away - don't cone project
    #print(" ratio of dimensions that are cone projected: ", (angles_before > radians).float().mean())
    #print("angle before", angles_before.mean(), angles_before.std(), angles_before.min(), angles_before.max())
    #print("radians", radians)
    #print(angles_before_chuncked > radians, "angles_before")
    #print("False region", (angles_before_chuncked > radians).float().mean())
    # get the indices of the false region
    #print(torch.where(angles_before_chuncked < radians))

    grad_temp_chuncked = grad_temp_2_chuncked.clone().detach()
    grad_temp_chuncked[angles_before_chuncked > radians] = cone_projection[angles_before_chuncked > radians] #grad_temp_1_chuncked[angles_before_chuncked > radians] #cone_projection[angles_before_chuncked > radians]

    grad_temp = grad_temp_chuncked.permute(0, 3, 1, 2) \
    .reshape(orig_shp[0], orig_shp[1], 
    chunk_size, chunk_size,
    grad_temp_1_chuncked.shape[1], grad_temp_1_chuncked
    .shape[2]) \
    .permute(0, 1, 4, 2, 5, 3) \
    .reshape(*(orig_shp))

    
    return grad_temp, ~(angles_before_chuncked > radians)


def cone_project_chuncked_zero(grad_temp_1, grad_temp_2, deg, orig_shp, chunk_size = 1):
    """
    grad_temp_1: gradient of the loss w.r.t. the robust/classifier free
    grad_temp_2: gradient of the loss w.r.t. the non-robust
    projecting the robust/CF onto the non-robust
    """
    grad_temp_1_chuncked = grad_temp_1.view(*orig_shp) \
    .unfold(2, chunk_size, chunk_size) \
    .unfold(3, chunk_size, chunk_size) \
    .permute(0, 1, 4, 5, 2, 3) \
    .reshape(orig_shp[0], -1, orig_shp[-2]//chunk_size, orig_shp[-1]//chunk_size) \
    .permute(0, 2, 3, 1)
    
    grad_temp_2_chuncked = grad_temp_2.view(*orig_shp) \
    .unfold(2, chunk_size, chunk_size) \
    .unfold(3, chunk_size, chunk_size) \
    .permute(0, 1, 4, 5, 2, 3) \
    .reshape(orig_shp[0], -1, orig_shp[-2]//chunk_size, orig_shp[-1]//chunk_size) \
    .permute(0, 2, 3, 1)
   
    angles_before_chuncked = torch.acos((grad_temp_1_chuncked * grad_temp_2_chuncked).sum(-1) / (grad_temp_1_chuncked.norm(p=2, dim=-1) * grad_temp_2_chuncked.norm(p=2, dim=-1)))
    radians = torch.tensor([deg], device=grad_temp_1_chuncked.device).deg2rad()
    

    grad_temp_chuncked = grad_temp_2_chuncked.clone().detach()
    grad_temp_chuncked[angles_before_chuncked > radians] = 0.

    grad_temp = grad_temp_chuncked.permute(0, 3, 1, 2) \
    .reshape(orig_shp[0], orig_shp[1], 
    chunk_size, chunk_size,
    grad_temp_1_chuncked.shape[1], grad_temp_1_chuncked
    .shape[2]) \
    .permute(0, 1, 4, 2, 5, 3) \
    .reshape(*(orig_shp))

    
    return grad_temp, ~(angles_before_chuncked > radians)



def normalize(x):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    x = x - torch.tensor(mean).to(x.device)[None, :, None, None]
    x = x / torch.tensor(std).to(x.device)[None, :, None, None]
    return x


def denormalize(x):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    x = x * torch.tensor(std).to(x.device)[None, :, None, None]
    x = x + torch.tensor(mean).to(x.device)[None, :, None, None]
    return x


def _map_img(x):
    """
    from -1 to 1 to 0 to 1
    """
    return 0.5 * (x + 1)


def _unmap_img(x, from_image_net_dist=False):
    """
    from 0 to 1 to -1 to 1
    """

    return 2. * x - 1


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self



def generate_samples(
        model, 
        sampler, 
        target_y, 
        ddim_steps, 
        scale, 
        init_image=None, 
        t_enc=None,
        init_latent=None, 
        ccdddim=False, 
        ddim_eta=0., 
        latent_t_0=True, 
        prompts: list = None,
        seed: int = 0
):
    torch.cuda.empty_cache()
    
    all_samples = []
    all_probs = []
    all_videos = []
    all_masks = []
    all_cgs = []

    with torch.no_grad():
        with model.ema_scope():
            tic = time.time()
            print(f"rendering target classes '{target_y}' in {len(sampler.ddim_timesteps)} or {ddim_steps}  steps and using s={scale:.2f}.")
            batch_size = target_y.shape[0]
            if "class_label" == model.cond_stage_key: # class-conditional
                uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(batch_size * [1000]).to(model.device)})
                c = model.get_learned_conditioning({model.cond_stage_key: target_y.to(model.device)})
            elif "txt" == model.cond_stage_key: # text-conditional
                uc = model.get_learned_conditioning(batch_size * [""])
                if prompts is None:
                    raise ValueError("Prompts are not defined!")
                c = model.get_learned_conditioning(prompts)
            else:
                raise NotImplementedError
                
            if init_latent is not None:
                if seed!=-1:
                    noises_per_batch = []
                    for b in range(batch_size):
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        random.seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        noises_per_batch.append(torch.randn_like(init_latent[b]))
                    noise = torch.stack(noises_per_batch, dim=0)
                else:
                    noise = None
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * (batch_size)).to(
                    init_latent.device), noise=noise) if not latent_t_0 else init_latent

                if seed!=-1:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.cuda.manual_seed_all(seed)

                # decode it
                if ccdddim:
                    out = sampler.decode(
                        z_enc, 
                        c, 
                        t_enc, 
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc, 
                        y=target_y.to(model.device), 
                        latent_t_0=latent_t_0,
                    )
                    samples = out["x_dec"]
                    prob = out["prob"]
                    vid = out["video"]
                    mask = out["mask"]
                    cg = out["concensus_regions"]

                else:
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc)

                x_samples = model.decode_first_stage(samples)
                x_samples_ddim = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                cat_samples = x_samples_ddim #torch.cat([init_image[:1], x_samples_ddim], dim=0)
            else:

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                                min=0.0, max=1.0)
                cat_samples = x_samples_ddim

            all_samples.append(cat_samples)
            all_probs.append(prob) if ccdddim and prob is not None else None
            all_videos.append(vid) if ccdddim and vid is not None else None
            all_masks.append(mask) if ccdddim and mask is not None else None
            all_cgs.append(cg) if ccdddim and cg is not None else None
        tac = time.time()

    out = {}
    out["samples"] = all_samples
    out["probs"] = all_probs if len(all_probs) > 0 else None
    out["videos"] = all_videos if len(all_videos) > 0 else None
    out["masks"] = all_masks if len(all_masks) > 0 else None
    out["cgs"] = all_cgs if len(all_cgs) > 0 else None
    
    return out


def generate_samples_one_to_many(model, sampler, classes, n_samples_per_class, ddim_steps, scale, init_image=None, t_enc=None,
                     init_latent=None, ccdddim=False, ddim_eta=0., latent_t_0=True):
    all_samples = []
    all_probs = []
    all_videos = []
    all_masks = []

    with torch.no_grad():
        with model.ema_scope():
            tic = time.time()
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)})

            for class_label in classes:
                print(
                    f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                if init_latent is not None:
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * (n_samples_per_class)).to(
                        init_latent.device)) if not latent_t_0 else init_latent

                    # decode it
                    if ccdddim:
                        out = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc, y=xc.to(model.device), latent_t_0=latent_t_0)
                        samples = out["x_dec"]
                        prob = out["prob"]
                        vid = out["video"]
                        mask = out["mask"]

                    else:
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc)

                    x_samples = model.decode_first_stage(samples)
                    x_samples_ddim = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    cat_samples = torch.cat([init_image[:1].cpu(), x_samples_ddim.cpu()], dim=0)
                else:

                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=n_samples_per_class,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                                min=0.0, max=1.0)
                    cat_samples = x_samples_ddim

                all_samples.append(cat_samples)
                all_probs.append(prob) if ccdddim and prob is not None else None
                all_videos.append(vid) if ccdddim and vid is not None else None
                all_masks.append(mask) if ccdddim and mask is not None else None

            tac = time.time()


    out = {}
    out["samples"] = all_samples
    out["probs"] = all_probs if len(all_probs) > 0 else None
    out["videos"] = all_videos if len(all_videos) > 0 else None
    out["masks"] = all_masks if len(all_masks) > 0 else None
    
    return out
