# Fisher Forecasting for CMB-HD

This repository contains software to produce Fisher forecasts for CMB-HD. Please cite [MacInnis, Sehgal, and Rothermel (2023)](https://arxiv.org/abs/2309.03021). The option to calculate the theory with CLASS instead of CAMB was added for [Cheslog et. al. (2026)](https://arxiv.org/abs/XXXX.XXXXX); please also cite that work if you use it.


# Installation

## Installation requirements

To use this software, you must have Python (version >=3) installed, along with the following Python packages:
- [hdMockData](https://github.com/CMB-HD/hdMockData) (__new__ as of May 2024)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [CAMB](https://camb.readthedocs.io/en/latest/)
- [CLASS](https://github.com/lesgourg/class_public) and its Python wrapper `classy` (__optional__: only required to calculate the theory with CLASS instead of CAMB; see the "Calculating Fisher matrices with CLASS" section below.)
- [matplotlib](https://matplotlib.org/) (this is only required to run the Jupyter notebooks)
- [getdist](https://getdist.readthedocs.io/en/latest/intro.html) (this is only required to run the Jupyter notebooks)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (__optional__: the calculation of the derivatives used in the Fisher matrices can be parallelized, but this is not required.) 


## Installation instructions

Simply clone this repository and install with `pip`:

```
git clone https://github.com/CMB-HD/hdfisher.git
cd hdfisher
pip install . --user
```

To uninstall the code, use the command `pip uninstall hdfisher`. (Note that you may have to navigate away from the `hdfisher` directory, i.e. the directory containing this README, before running this command).


# CMB and BAO mock signal and covariance matrices

The `hdMockData` [repository](https://github.com/CMB-HD/hdMockData) contains the latest version of the CMB-HD mock data, including the signal and noise lensed and delensed $TT/TE/EE/BB$ and lensing $\kappa\kappa$ power spectra and their associated covariance matrix. By default, the Fisher forecasting code in this repository will use the latest version of the CMB-HD data, which can be accessed by using the `Data` class in `hdfisher/dataconfig.py`.

The forecasts in MacInnis et. al. (2023) used version `'v1.0'` of the CMB-HD mock data. In this repository, we provide the version `'v1.0'` of the CMB-HD mock data described above from multipoles $30$ to $20,000$.  We also provide a mock DESI BAO signal and covariance matrix. These files are in the sub-directories of the `hdfisher/data` directory, and can also be accessed by using the `Data` class in `hdfisher/dataconfig.py`; see the provided notebook `forecast_plots.ipynb` (described in the "Reproducing plots and tables" section below) for an example.


# Reproducing plots and tables in MacInnis et. al. 2023

We provide some pre-computed Fisher matrices in the `hdfisher/data/fisher_matrices` directory. These can also be accessed with the `Data` class in `hdfisher/dataconfig.py`. __Note__ that we have applied a prior on $\tau$ of $\sigma(\tau) = 0.007$ to all pre-computed Fisher matrices (except for those that only include mock BAO data), and that these Fisher matrices used the original `'v1.0'` of the CMB-HD [mock data](https://github.com/CMB-HD/hdMockData).

We have provided a Jupyter notebook, `forecast_plots.ipynb`, which will reproduce most of the plots and tables. This can also be used as an example of how to access the CMB-HD mock data and other files that are provided with `hdfisher`, and how to obtain parameter uncertainties from a Fisher matrix.


# Calculating new Fisher matrices

The Fisher matrix for a given set of parameters is calculated from a set of  derivatives of the mock signal with respect to each parameter, and a covariance matrix for the mock signal (described above). The calculations are done using the `Fisher` class of `hdfisher/fisher.py`. We include additional functions in `hdfisher/fisher.py` to add Fisher matrices, apply a Gaussian prior on a parameter, and remove (a) parameter(s) from a Fisher matrix.

We provide a Jupyter notebook, `example_calculate_fisher_matrices.ipynb`, as an example of the Fisher matrix calculation (including the calculation of the derivates with respect to parameters) for the baseline CMB-HD forecasts. We also provide an example script, `example_calculate_fisher_derivatives.py`, that can be used to calculate the derivatives in parallel with MPI. In the example notebook, we compare the results to a pre-computed Fisher matrix provided with `hdfisher`.

In the example notebook, we use the latest version of the CMB-HD [mock data](https://github.com/CMB-HD/hdMockData) with the fiducial cosmological parameters, parameter step sizes, and CAMB accuracy settings described in MacInnis et. al. (2023); this is the default behavior. You may also provide your own set of fiducial parameter values by saving them in a YAML file, and passing the file name as the `param_file` to the `Fisher` class. Additionally, you may provide your own step sizes in a second YAML file, and pass its name as the `fisher_steps_file` to the `Fisher` class. See `example_calculate_fisher_matrices.ipynb` for more details and additional options. The parameter names must be names that can be passed to the CAMB function `camb.set_params()` (or, if `use_class=True`, names that CLASS accepts; see below).

# Calculating Fisher matrices with CLASS

The `Fisher` class can also calculate the CMB and BAO theory with [CLASS](https://github.com/lesgourg/class_public) instead of CAMB, by passing `use_class=True` when initializing it. This is how the CLASS Fisher forecasts in Cheslog et. al. (2026) were calculated.

When `use_class=True`, the `param_file` and `fisher_steps_file` must use CLASS parameter names (e.g. `omega_b`, `omega_cdm`, `tau_reio`, `ln_A_s_1e10`, `n_s`), and every entry in the `param_file` is passed to CLASS as-is, including the CLASS accuracy settings. The same names must be used in the `params` and `priors` arguments of `get_fisher` (e.g. `priors={'tau_reio': 0.007}`), and are used for the parameters of the returned Fisher matrix. We provide a fiducial parameter file, `hdfisher/data/fiducial_params/class_fiducial_params.yaml`, and step size file, `hdfisher/data/fisher_step_sizes/class_fiducial_step_sizes.yaml`, holding the CLASS settings described in Appendix A of Cheslog et. al. (2026); these are used by default when `use_class=True`. We also provide their CAMB counterparts, `camb_fiducial_params.yaml` and `camb_fiducial_step_sizes.yaml`, which hold the CAMB settings used in that work. (The default `fiducial_params.yaml` and `fiducial_step_sizes.yaml` used when `use_class=False` are unchanged from MacInnis et. al. (2023).) See `example_calculate_fisher_matrices.ipynb` for a description of the CLASS parameter file, including the few entries that `hdfisher` converts before passing them to CLASS.

__Note__ the following before using CLASS:
- __The CLASS source code must be modified__ as described in Appendix A of Cheslog et. al. (2026), or the calculation will fail. In `source/lensing.c`, the variables `num_mu` and `index_mu` must be changed from `int` to `long long` (and `icount` to `unsigned long long`), so that the lensing calculation does not overflow at the multipoles used for CMB-HD. In `source/input.c`, the check that rejects a negative `N_ur` must be commented out, so that $N_\mathrm{eff}$ can be varied below its standard value with three massive neutrinos. These are at lines 124 and 2470 of CLASS version 3.3.4; the forecasts in Cheslog et. al. (2026) used version 3.3.0. A warning is issued as a reminder whenever `use_class=True`.
- CLASS reads the BBN table named by the `sBBN file` entry of the parameter file itself, so it must be an absolute path. We provide the table in `hdfisher/data/class_inputs/PRIMAT21_class_format.dat`; replace the `/path/to/hdfisher` placeholder in `class_fiducial_params.yaml` with the path to your copy of this repository.
- Delensed spectra are not calculated with CLASS, so `cmb_type='delensed'` cannot be used with `use_class=True`; use `cmb_type='lensed'` instead. The `feedback` option is also not available with CLASS.
