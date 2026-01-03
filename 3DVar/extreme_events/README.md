# Forecasting Typhoon Mawar 2023

This folder contains the code to generate FourCastNet forecasts of Typhoon Mawar given three initial conditions: (1) interpolated 4.5 $^\circ$ observations, (2) our FCN+3DVar analysis (4.5 $^\circ$ ), and (3) "ground truth" ERA5 data. We additionally compare against IBTrACS data and IFS-HRES forecasts.

These commands assume the interpolated observations and FCN+3DVar analyses have been computed (see ```FCN-3DVar/3DVar/filtering```).

## Computing forecasts

To compute FourCast forecasts, run the following command in a terminal.

```
sbatch typhoon18.sh
```

## Identifying the predicted eye of the typhoon

To identify the predicted eye of the typhoon for the generated forecasts, run the following commands in a terminal.

```
sbatch get_eye_analysis.sh

sbatch get_eye_clean.sh

sbatch get_eye_obs.sh
```

## Visualizing results

Visualizations of the results are contained in the notebook ```typhoon.ipynb```.