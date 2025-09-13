<div align="center">
  <h1>Latent Diffusion Core</h1>
</div>

The repository contains a minimal implementation of the latent diffusion model (mini Stable Diffusion) on PyTorch.

## Project Description

This project demonstrates:
- Training Variational Autoencoder (VAE) to compress images into a compact latent space.
- Training a U-Net denoizer in this latent space using a diffusion process.
- Using a text encoder (CLIP) and cross-attention to generate images based on text prompts.
- Support for fast samplers (DDIM/PNDM) and classifier-free guidance.

The final goal is to generate 256×256 images based on a text query with a quality approaching Stable Diffusion.

## Installation
To run this project, you'll need to set up a Python environment and install the necessary dependencies.

### Prerequisites
Make sure you have Python 3.11 installed.

1. Clone the repository:
```bash
git clone https://github.com/vlvink/latent_diffusion_core.git
cd latent_diffusion_core
```

2. Install the requirements
```bash
poetry install
```

3. Setting the poetry environment
```bash
poetry shell
```

### Dataset downloading
```bash
curl -L -o ./data/coco-2017-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/awsaf49/coco-2017-dataset

unzip -o ./data/coco-2017-dataset.zip -d ./data
mv ./data/coco2017/* ./data/
rmdir ./data/coco2017

rm ./data/coco-2017-dataset.zip
```

## Running the Code
### Training VAE
```bash
python train_vae.py \
  --epochs 50 \
  --batch-size 64
```

### Training Diffusion Model
```bash
python train_diffusion.py \
  --epochs 200 \
  --batch-size 32 \
  --text-encoder clip-vit
```

### Image Generation
```bash
python sampling.py \
  --prompt "A futuristic cityscape at sunset" \
  --steps 50 \
  --guidance-scale 7.5 \
  --output out.png
```


