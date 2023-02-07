# Probabilistic Attention based on Gaussian Processes for Deep Multiple Instance Learning
Authors: Arne Schmidt, Pablo Morales-Álvarez, Rafael Molina
## Description
This repository contains the code of the above mentioned article.
The proposed model for Multiple Instance Learning (MIL) uses Gaussian Processes to estimate the attention weights for the instances.
It can be trained end-to-end with deep learning feature extractors and is based on tensorflow and tensorflow-probabilty.
For the MNIST example, there is a seperate repository: https://github.com/arneschmidt/attention_gp_mnist
## Installation and Usage
To make this code run on your linux machine you need to:
* Install miniconda (or anaconda): https://docs.anaconda.com/anaconda/install/linux/ 
* Navigate to this base folder (attention_gp)
* Set up a conda environment and activate it:
    * `conda env create --file environment.yaml`
    * `conda activate attention_gp`
* Edit the configuration:
    * `./config.yaml` for general settings
    * `./dataset_dependent/sicapv2/dataset_config.yaml` for dataset dependent settings
    * NOTE: The default configuration represents the proposed model as described in the article
* Run the program:
    * `python ./src/main.py`
## Structure
* src: contains all related source code files
* dataset_dependent: contains dataset configurations and possible experiment outputs
* input: contains input tables which are 128-dimensional extracted features