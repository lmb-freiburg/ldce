# Latent Diffusion Counterfactual Explanations

This is the official code of the paper [Latent Diffusion Counterfactual Explanations](https://arxiv.org/abs/2310.06668).

If this work is useful to you, please consider citing our paper:

```
@misc{farid2023latent,
    title={Latent Diffusion Counterfactual Explanations}, 
    author={Karim Farid and Simon Schrodi and Max Argus and Thomas Brox},
    year={2023},
    eprint={2310.06668},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

Download the following model weights:
* LDM on ImageNet from [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
* miniSD from [https://huggingface.co/justinpinkney/miniSD/tree/main](https://huggingface.co/justinpinkney/miniSD/tree/main)
* CelebA HQ DenseNet-121 classifier from [https://github.com/valeoai/STEEX/releases](https://github.com/valeoai/STEEX/releases)

## Counterfactual generation with LDCE
Before generating counterfactuals, you need to configure the config file in `configs/ldce/*.yaml`, e.g., set the paths to the dataset etc.

Below we provide the commands to reproduce the results from our paper.

### ImageNet

#### All classes (Table 1)

For class-conditional diffusion model:
```
python -m scripts.ldce --config-name=v1_wider \
    data.batch_size=5 \
    strength=0.382 \
    data.start_sample=$id data.end_sample=$((id+1)) > logs/imagenet_sd_${id}.log 
```

For text-conditional diffusion model:
```
python -m scripts.ldce --config-name=v1_stable_diffusion \
    data.batch_size=4 \
    sampler.classifier_lambda=3.95 \
    sampler.dist_lambda=1.2 \
    sampler.deg_cone_projection=50. \
    data.start_sample=$id data.end_sample=$((id+1)) > logs/imagenet_sd_${id}.log 
```

#### Only pairs (Table 2; here exemplary for zebra-sorrel)

For class-conditional diffusion model:
```
python -m scripts.ldce --config-name=v1_zs \
    data.batch_size=4 > logs/zs_cls.log 
```

For text-conditional diffusion model:
```
python -m scripts.ldce --config-name=v1_zs \
    data.batch_size=4 \
    strength=0.382 \
    sampler.classifier_lambda=3.95 \
    sampler.dist_lambda=1.2 \
    sampler.deg_cone_projection=50. \
    diffusion_model.cfg_path="configs/stable-diffusion/v1-inference.yaml" \
    diffusion_model.ckpt_path="/path/to/miniSD.ckpt" > logs/zs_sd.log 
```

### CelebA HQ (Table 6)

```
python -m scripts.ldce --config-name=v1_celebAHQ \
    data.batch_size=4 \
    sampler.classifier_lambda=4.0 \
    sampler.dist_lambda=3.3 \
    data.num_shards=7 \
    sampler.deg_cone_projection=55. \
    data.shard=$id \
    strength=$strength > logs/celeb_smile.log 
```

### Flowers 102

```
python -m scripts.ldce --config-name=v1_flowers\
    data.batch_size=4 \
    strength=0.5 \
    sampler.classifier_lambda=3.4 \
    sampler.dist_lambda=1.2 \
    output_dir=results/flowers \
    data.num_shards=7 \
    data.shard=${id} \
     > logs/flowers_${id}.log 
```

### Oxford-IIIT Pets

```
python -m scripts.ldce --config-name=v1_pets\
    data.batch_size=4 \
    sampler.classifier_lambda=4.2 \
    sampler.dist_lambda=2.4 \
    data.num_shards=7 \
    data.shard=$id \
     > logs/pets_${id}.log 
```

## Acknowledgements

We thank the following GitHub users/researchers/groups:

* Rombach et al. for open sourcing the code and providing pretrained models for latent diffusion models/stable diffusion: [https://github.com/CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion).
* Justin Pinkney for sharing weights of the fine-tuned stable diffusion variant for 256x256 images: [https://huggingface.co/justinpinkney/miniSD/tree/main](https://huggingface.co/justinpinkney/miniSD/tree/main).
* Paul Jacob et al. for sharing the weights of the trained DenseNet-121 on CelebA-HQ: [https://github.com/valeoai/STEEX](https://github.com/valeoai/STEEX).