# Data Assimilation with Machine Learning Surrogate Models: A Case Study with FourCastNet


### Abstract


Modern data-driven surrogate models for weather forecasting provide accurate short-term predictions but inaccurate and nonphysical long-term forecasts. This paper investigates online weather prediction using machine learning surrogates supplemented with partial and noisy observations. We empirically demonstrate and theoretically justify that, despite the long-time instability of the surrogates and the sparsity of the observations, filtering estimates can remain accurate in the long-time horizon. As a case study, we integrate FourCastNet, a weather surrogate model, within a variational data assimilation framework using partial, noisy ERA5 data. Our results show that filtering estimates remain accurate over a year-long assimilation window and provide effective initial conditions for forecasting tasks, including extreme event prediction.


### Repository organization

This repository is organized as follows:

```3DVar``` folder contains code for our experiments. Visit the README.md files in this folder for detailed information in running each experiment.

```config``` folder contains the configuration file to run FourCastNet and data assimilation experiments. The .yaml file can be modified to point to the appropriate filepaths.

```data``` folder contains the structure of the folders for how data is stored and accessed by the code.

```networks``` folder contains files for the network structure of FourCastNet.

```results``` folder where results are stored.

```stats_v0``` folder contains ```.npy``` files needed to run FourCastNet.

```utils_fcn``` folder contains useful functions for code in the ```3DVar``` folder.

```weights``` folder stores the FourCastNet model weights to read in during evaluation.


### References:

This repository uses/modifies code from the following two sources:

1. FourCastNet (https://github.com/NVlabs/FourCastNet)
```
@article{pathak2022fourcastnet,
  title={Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar and Hassanzadeh, Pedram and Kashinath, Karthik and Anandkumar, Animashree},
  journal={arXiv preprint arXiv:2202.11214},
  year={2022}
}
```

2. AD-EnKF (https://github.com/ymchen0/torchEnKF)
```
@article{doi:10.1137/21M1434477,
author = {Chen, Yuming and Sanz-Alonso, Daniel and Willett, Rebecca},
title = {Autodifferentiable Ensemble Kalman Filters},
journal = {SIAM Journal on Mathematics of Data Science},
volume = {4},
number = {2},
pages = {801-833},
year = {2022},
doi = {10.1137/21M1434477},
URL = {https://doi.org/10.1137/21M1434477},
eprint = {https://doi.org/10.1137/21M1434477}
}
```

### Data Sources:

ERA5 data was downloaded from the Copernicus Climate Change Service (C3S) Climate Data Store.

```
Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2018): ERA5 hourly data on pressure levels from 1959 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). , 10.24381/cds.bd0915c6

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2018): ERA5 hourly data on single levels from 1959 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). , 10.24381/cds.adbb2d47
```

ECMWF's IFS-HRES forecasts were downloaded from the TIGGE Data Retrieval portal.

```
P. Bougeault, Z. Toth, C. Bishop, B. Brown, D. Burridge, D. H. Chen, B. Ebert, M. Fuentes, T. M. Hamill, K. Mylne, J. Nicolau, T. Paccagnella, Y.-Y. Park, D. Parsons, B. Raoult, D. Schuster, P. S. Dias, R. Swinbank, Y. Takeuchi, W. Tennant, L. Wilson, and S. Worley. The THORPEX interactive grand global ensemble. Bulletin of the American Meteorological Society, 91(8):1059 – 1072, 2010. doi:10.1175/2010BAMS2853.1. URL https://journals.ametsoc.org/view/journals/bams/91/8/2010bams2853_1.xml.
```

NOAA’s International Best Track Archive for Climate Stewardship (IBTrACS) data 

```
K. R. Knapp, M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann. The international best track archive for climate stewardship (ibtracs): Unifying tropical cyclone data. Bulletin of the American Meteorological Society, 91(3):363 – 376, 2010. doi:10.1175/2009BAMS2755.1. URL https://journals. ametsoc.org/view/journals/bams/91/3/2009bams2755_1.xml.

J. Gahtan, K. R. Knapp, C. J. Schreck, H. J. Diamond, J. P. Kossin, and M. C. Kruk. International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01. IBTrACS.last3years.v04r01. URL https://www.ncei.noaa.gov/products/international-best-track-archive.
```

### Citing this work

```
@article {Adrian2025,
      author = "Melissa Adrian and Daniel Sanz-Alonso and Rebecca Willett",
      title = "Data Assimilation with Machine Learning Surrogate Models: A Case Study with FourCastNet",
      journal = "Artificial Intelligence for the Earth Systems",
      year = "2025",
      publisher = "American Meteorological Society",
      address = "Boston MA, USA",
      volume = "4",
      number = "3",
      doi = "10.1175/AIES-D-24-0050.1",
      pages=      "e240050",
      url = "https://journals.ametsoc.org/view/journals/aies/4/3/AIES-D-24-0050.1.xml"
}
```

