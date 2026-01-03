# 3DVar data assimilation with FourCastNet

This paper present three types of results: (1) long-term assimilation errors, (2) forecasting errors based on various initial conditions, and (3) typhoon forecasting errors based on various initial conditions.

## 0. Data prepocessing

In a terminal, change the working directory to ```FCN-3DVar/data```. Preprocess the 2023 ERA5 data from one .h5 file to one file per timepoint with the following command.

```
python create_single_files.py
```

Preprocess the 1979-2015 ERA5 data to create a climatology dataset and store one file per timepoint with the following command.

```
python create_single_files_clim.py
```

## 1. Long-term assimilation errors

Code to produce of long term assimilation errors is contained in the ```filtering``` folder along with a README.md file containing instructions of how to run the code.

## 2. Forecasting errors based on various initial conditions

Code to produce forecast errors given various initial conditions is contained in the ```forecasting``` folder along with a README.md file containing instructions of how to run the code.

## 3. Typhoon forecasting errors based on various initial conditions

Code to reproduce the Typhoon Mawar results is contained in the ```extreme_events``` folder along with a README.md file containing instructions of how to run the code.

## 4. Misc.

The ```figures``` folder contains the .png files of the images in the paper.

The ```torch3dvar``` folder contains our implementation of 3DVar. This code was adapted from AD-EnKF (https://github.com/ymchen0/torchEnKF).