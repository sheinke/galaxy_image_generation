# Galaxy Image Generation 

This repository contains the semester project for the Computational Intelligence Lab 2020 at ETHZ.  

This project was done in a group of four. Group members were Philippe Blatter, Lucas Brunner, Alicja Chaszczewicz and me.


### Project description
The first part of this project is to build a generative model that is able to produce realistic cosmology images.
The provided dataset contains 10800 gray-scales images with 1000x1000 pixels resolution.
The second objective is to repurpose the aforementioned generative model and use it to predict so-called "score-values": A function that measures the similarity of a cosmologic image to a "prototypically ideal" image is defined on a large subset of the data.
Finally, a set of unseen query images needs to be scored and submitted to the Kaggle competition.

### Repository structure
Our initial exploration of the dataset can be found in `./data_exploration/`. There, we have summarized some basic observations and statistics about the given images.

The VAE, DCGAN, WGAN and StyleGAN2 models described in the project paper can be found in `./cvae/, ./dcgan/,  ./wgan/, ./stylegan2/`.

The second part of the project (learning the similarity function) can be found in `./score_prediction/`. All feature-based experiments are located in `./score_prediction/feature_baseline/`.

Most of our code makes use of the `./utils/` folder, where we have collected useful functions for handling input and output operations of images, as well as data augmentation etc.

### Requirements
Our code requires the following modules to be run on the Leonhard cluster:

`module load gcc/6.3.0 python_gpu/3.7.4 hdf5 eth_proxy`   

For a list of packages that should be installed in your Python environment, please see `requirements.yml`.
