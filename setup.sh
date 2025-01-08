#!/bin/sh

sudo apt-get install -y g++ build-essential libgl1
cd diffusion_policy
conda init
source ~/.bashrc
conda env create -f conda_environment.yaml
conda activate robodiff

