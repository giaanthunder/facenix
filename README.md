# Facenix project
> Facenix is a face attribute manipulating application. This app is deployed by using Django web framework with STGAN and StyleGAN models as backend.

[![INSERT YOUR GRAPHIC HERE](x.png)]()

## Preparation
### Hardware (minimum)
CPU: core i3
RAM: 8GB
SSD: 120GB
GPU: Nvidia P106-100
### Software
OS      : Ubuntu 16.04 or above 
CUDA    : 10.1
Anaconda: 3

Run folling command for creating python environment
conda create -n facenix python=3.6
conda activate facenix
pip install --upgrade pip
 
### Application materials
Clone source
git clone
pip install -r requirements.txt
Download pretrained models, extract it. Copy all content of Facenix_data to Facenix directory.
   + Pretrained models

## Testing application
Please make sure you have done all preparation steps.
cd <Facenix working dir>/web_app
./runserver.sh
Open web browser, go to <link>. You should see below GUI. Choose 1 between 2 methods and upload a photo. Face region will be cropped automatically.
[![INSERT YOUR GRAPHIC HERE](x.png)]()

## Re-training models
If you would like to re-train all models from scratch please follow steps below:
- Download training data. Extract it.
   + CelebA:
   + CelebA-HQ: 
- Change data location
- Train STGAN:
   + cd <Facenix working dir>/stgan
   + python train.py
- Train StyleGAN:
   + cd <Facenix working dir>/classifier
   + 
   + cd <Facenix working dir>/boundary_finder
   +
   + cd <Facenix working dir>/stylegan
   + python train.py

## Known issues
- Attribute editing is not guarantee successful.
- When editing with StyleGAN, the result can be different due to random initialization of encode vector






