# DLESyM Teleconnections: Context
Exploring remote teleconnections in the Deep Learning Earth System Model (DLESyM) presented in Cresswell-Clay et al. 2024.

In the study cited above, DLESyM has demonstrated remarkable fidelity in its recreation of internal variability. One notable example explored here is the southern annular mode (SAM). In particular I'm interested in the well-documented relationship between tropical SSTs and SAM (Ding et al. 2012).

Also noted in study presenting DLESyM is its underestimation the ENSO cycle. Despite this definceincy it remains of interest whether a atmospheric model trained to optimize RMSE over 24 hours is capable of propocating a much longer slower evolving tropical signal into polar regions.

To sidestep this, we will run the atmospheric compoenet of DLESyM in forced mode: coupled with observed SSTs rather than coupled to the dynamic ocean component. Our questions is thus: Can a deep learning weather model realistically propogate forcings from tropical perturbations onto polar modes like the Southern Annular mode.

# The Repo

This repo was made as part of an assignment. Each script corresponds to a set of analyses and figures that were used to explore ENSO-SAM teleconnections in DLESyM. Github's file limits mean I can't upload the data used, but I'm including all processing scripts. Generally in these analyses, raw DLESyM output is processed with the script in `data/` to create reduced dimensionality caches which are used in the evaluation routines. Here are brief descriptions of the contained files:  


`environment.yaml`: requirements file for the environmnet used to run and develop.  
`eofs.py`: computes EOFs and plots their spatial variability and the temporal explained variance.  
`sst_sam_regression.py`: calculates and plots the regression of SST anomalies on SAM phase.  
`data/monthly_sst.py`: takes a DLESyM forecast and calculates monthly averaged SST anomalies.   
`data/monthly_z250.py`: takes a DLESyM forecast and calculates the monthly averaged z250. anomalies.   
`plots/*.png`: output plots of the analsis routines. These can be used as a reference. 

<!-- `plotting_olr`: function to plot forecast output  -->

# Setting up the environment

To get started you'll need to create a viable virtual environment. I've used [Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux) to manage my environment during development (my vesion is 4.13.0). Once you've downloaded and initialized conda, use environment.yaml to download the exact versions of packages that were used during development. 

```
$ conda env create -f environment.yaml
```

Next we'll need the ```DLESyM``` repository for some data processing utilities

```
git clone https://github.com/AtmosSci-DLESM/DLESyM.git
```

# Data

For access to the full forecast output/observations used, please contact me directly (nacc@uw.edu). 

# Running analyses 

To run these analysis, activate your newly created conda environment: 

```
$ conda activate dlesym-tele
```

Then add the `DLESyM` module to your python path. We'll need to use some data processing utils for our analysis. 

```
$ export PYTHONPATH=/path/to/DLESyM
```

And run the routine, 

```
$ python eofs.py
```