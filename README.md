## Getting started

Install pip packages with requirements.txt file in euler folder
```
pip install -r euler/requirements.txt
```

## Repo Overview

# Training, Evaluation and Inference
All python scripts for running and evaluating both the diffusion models and MLP ensembles are in the euler directory:

- newsub.py runs the diffusion model training. At the top of the script you can select which model mentioned in the write-up to train
- eval.py runs the evaluation on the SOCAT test set. Which model to evaluate is selected the top of the script by typing the directory where it was saved
- expo.py generates cruise tracks using random offsets to generate smooth samples 
- hpfast.py generates samples foe the entire ocean surface using the HEALPix projection. 
- hpface.py generates time-conditioned samples for a given period of time and patch

- sota.py trains a baseline MLP ensemble with 50 members
- hpmlp.py generates samples for the whole ocean using a trained MLP ensemble
- mlp_expo.py generates cruise tracks with the MLP ensemble

- utils.py contains a single function to add a logger and add a path to the src folder

# SOCAT Preprocessing and data collocation
The preprocessing functions are in the src/fco2dataset folder:

- ucollocate.py downloads remote sensing data as zarr files and provides functions to collocate the data to given coordinates
- ucruise.py contains the functions to bin the raw socat measurements in 5 km wide bins and to interpolate missing ship positions

These functions were used in the clean/add_bin_info.ipynb and notebooks/track_augmentation.ipynb files to generate the 
data/training_data/SOCAT_1982_2021_grouped_colloc_augm_bin.pq collocated, binned dataset with interpolated positions

# Helper Scripts
src/fco2models contains other python scripts for training and preprocessing:

- models.py contains the different model classes used
- time_models.py contains transformer models for time-series (not used anymore)
- utraining.py contains code for training the diffusion model, inferring single samples and preprocessing on binned and collcated SOCAT data
- ueval.py contains functions helping with evaluation of diffusion models
- umeanest.py contains functions helping with MLP training and evaluation

# Modified Packages
src/mydiffusers contains two script modified from the diffusers library
- ddim_scheduling.py modified the inference routine to allow arbitrary inference schedules
- unet_1d.py modified some output shapes otherwise code breaks


