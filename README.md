# Getting started
Create the conda environment for this repo by running
```
conda env create -f euler/environment.yml
```
this will create the environment ```fco2diffusion```. Activate the environment the usual way with
```
conda activate fco2diffusion
```


# Repo Overview
Short descsription of the relevant python scripts and jupyter notebooks

## Training, Evaluation and Inference
All python scripts for running and evaluating both the diffusion models and MLP ensembles are in the euler directory:

- [newsub.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/newsub.py) runs the diffusion model training. At the top of the script you can select which model mentioned in the write-up to train
- [eval.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/euler.py) runs the evaluation on the SOCAT test set. Which model to evaluate is selected the top of the script by typing the directory where it was saved
- [expo.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/euler.py) generates cruise tracks using random offsets to generate smooth samples 
- [hpfast.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/hpfast.py) generates samples foe the entire ocean surface using the HEALPix projection. 
- [hpface.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/hpface.py) generates time-conditioned samples for a given period of time and patch

- [sota.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/sota.py) trains a baseline MLP ensemble with 50 members
- [hpmlp.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/hpmlp.py) generates samples for the whole ocean using a trained MLP ensemble
- [mlp_expo.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/mlp_expo.py) generates cruise tracks with the MLP ensemble

- [utils.py](https://github.com/ebrusoni/fco2diffusion/blob/main/euler/utils.py) contains a single function to add a logger and add a path to the src folder. The path to the location of the dataset directory can also be specified here.

## SOCAT Preprocessing and data collocation
The preprocessing functions are in the src/fco2dataset folder:

- [ucollocate.py](https://github.com/ebrusoni/fco2diffusion/tree/main/src/fco2dataset/ucollocate.py) downloads remote sensing data as zarr files and provides functions to collocate the data to given coordinates
- [ucruise.py](https://github.com/ebrusoni/fco2diffusion/tree/main/src/fco2dataset/ucruise.py)  contains the functions to bin the raw socat measurements in 5 km wide bins and to interpolate missing ship positions

These functions were used in the [clean/add_bin_info.ipynb](https://github.com/ebrusoni/fco2diffusion/blob/main/clean/add_bin_info.ipynb) and [notebooks/track_augmentation.ipynb](https://github.com/ebrusoni/fco2diffusion/blob/main/notebooks/track_augmentation.ipynb) files to generate the 
data/training_data/SOCAT_1982_2021_grouped_colloc_augm_bin.pq collocated, binned dataset with interpolated positions

## Helper Scripts
src/fco2models contains other python scripts for training and preprocessing:

- [models.py](https://github.com/ebrusoni/fco2diffusion/blob/main/src/fco2models/models.py) contains the different model classes used
- [time_models.py](https://github.com/ebrusoni/fco2diffusion/blob/main/src/fco2models/time_models.py) contains transformer models for time-series (not used anymore)
- [utraining.py](https://github.com/ebrusoni/fco2diffusion/blob/main/src/fco2models/utraining.py) contains code for training the diffusion model, inferring single samples and preprocessing on binned and collcated SOCAT data
- [ueval.py](https://github.com/ebrusoni/fco2diffusion/blob/main/src/fco2models/ueval.py) contains functions helping with evaluation of diffusion models
- [umeanest.py](https://github.com/ebrusoni/fco2diffusion/blob/main/src/fco2models/unmeanest.py) contains functions helping with MLP training and evaluation

## Modified Packages
src/mydiffusers contains two script modified from the diffusers library
- [ddim_scheduling.py](https://github.com/ebrusoni/fco2diffusion/blob/main/src/mydiffusers/scheduling_ddim.py) modified the inference routine to allow arbitrary inference schedules
- [unet_1d.py](https://github.com/ebrusoni/fco2diffusion/blob/main/src/mydiffusers/unet_1d.py) modified some output shapes otherwise code breaks


