# Forecasting given various initial conditions

This folder contains code to compute FourCastNet forecast errors with various initial conditions over the span of a year: 1) ERA5, 2) climatology, 3) interpolated observations, and 4) 3DVar+FCN analyses. 


## 1. Forcasting with ERA5 data

To compute the forecast errors when initializing with ERA5 data, run the following command in a terminal:

```
sbatch forecasting_given_clean_init.sh
```

## 2. Forecasting with climatology data

To compute the forecast errors when initializing with climatology, run the following command in a terminal:

```
sbatch forecasting_given_clim_init.sh
```

This scripts assumes that you have already computed climatology initial conditions (average of ERA5 data for each pixel from the years 1979-2015).

## 3. Forecasting with interpolated observations

To compute the forecast errors when initializing with interpolated observations (4.5 degrees), run the following command in a terminal:

```
sbatch forecasting_given_upsc_obs_init18.sh
```

This scripts assumes that you have already computed interpolated observation initial conditions. See the ```README.md``` in the ```FCN-3DVar/3DVar/filtering``` folder.


## 4. Forecasting with a 3DVar+FCN analysis

To compute the forecast errors when initializing with 3DVar+FCN analyses (4.5 degrees), run the following command in a terminal:

```
sbatch forecasting_given_analysis_estim18.sh
```

This scripts assumes that you have already computed the 3DVar+FCN (4.5 degrees) analyses.

## Results

The results are visualized in the ```forecasting.ipynb``` notebook.