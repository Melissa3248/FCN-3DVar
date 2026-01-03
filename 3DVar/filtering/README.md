# Long-term Errors

This folder contains the main files used to compute and visualize long-term errors for various methods. Note that the preamble of the .sh files will need to be edited to be compatible with the user's particular computing cluster.

## Computing errors from 3DVar analyses

To first compute the 3DVar analyses, run the following commands in a terminal.

```
sbatch fcn-3dvar-GLOBAL-obs8.sh

sbatch fcn-3dvar-GLOBAL-obs10.sh

sbatch fcn-3dvar-GLOBAL-obs18.sh

sbatch fcn-3dvar-GLOBAL-obs20.sh
```

Running the following command in a terminal computes the errors of the 3DVar analyses for every 6 hours across a year.

```
sbatch filtering.sh
```

## Computing errors from interpolated observations

To first compute the interpolated observations, run the following command in a terminal. This code assumes the 3DVar analyses have already been computed (and therefore perturbed observation files have been created).

```
sbatch upscale_obs.sh
```

Running the following command in a terminal computes the errors of the interpolated observations for every 6 hours across a year.

```
sbatch compute_upscaled_obs_error.sh
```
