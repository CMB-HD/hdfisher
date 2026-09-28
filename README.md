# Fisher Forecasting for CMB-HD

This repository contains software to produce Fisher forecasts for CMB-HD. If you use this code, please cite:
- [MacInnis, Sehgal, and Rothermel (2023)](https://arxiv.org/abs/2309.03021)
- If you use CLASS, please also cite [Cheslog, Finson, MacInnis, Sehgal, Afshordi, Nerval, and Hložek (2026)](https://arxiv.org/abs/XXXX.XXXXX)
- If you use the CMB-HD mock data, please also cite the appropriate references given in the [`hdMockData` repository](https://github.com/CMB-HD/hdMockData#forecasting-data-for-cmb-hd) for the mock data version used (the latest version by default)


# Installation

## Installation requirements

To use this software, you must have Python (version >=3) installed, along with the following Python packages:
- [hdMockData](https://github.com/CMB-HD/hdMockData) (latest version)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [CAMB](https://camb.readthedocs.io/en/latest/)
- [CLASS](https://github.com/lesgourg/class_public) (this is only required if you would like to use CLASS; if so, please **read [this note](#class-modifications-required-to-use-the-defaults-in-hdfisher)**  below)
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

The forecasts in MacInnis et. al. (2023) used version `'v1.0'` of the CMB-HD mock data. In this repository, we provide the version `'v1.0'` of the CMB-HD mock data described above from multipoles $30$ to $20,000$.  We also provide a mock DESI BAO signal and covariance matrix. These files are in the sub-directories of the `hdfisher/data` directory, and can also be accessed by using the `Data` class in `hdfisher/dataconfig.py`; see the provided notebook `forecast_plots.ipynb` (described in the "[Reproducing plots and tables" section](#reproducing-plots-and-tables-in-macinnis-et-al-2023) below) for an example.


# Reproducing plots and tables in MacInnis et. al. 2023

We provide some pre-computed Fisher matrices in the `hdfisher/data/fisher_matrices` directory. These can also be accessed with the `Data` class in `hdfisher/dataconfig.py`. __Note__ that we have applied a prior on $\tau$ of $\sigma(\tau) = 0.007$ to all pre-computed Fisher matrices (except for those that only include mock BAO data), and that these Fisher matrices used the original `'v1.0'` of the CMB-HD [mock data](https://github.com/CMB-HD/hdMockData).

We have provided a Jupyter notebook, `forecast_plots.ipynb`, which will reproduce most of the plots and tables. This can also be used as an example of how to access the CMB-HD mock data and other files that are provided with `hdfisher`, and how to obtain parameter uncertainties from a Fisher matrix.


# Calculating new Fisher matrices

The Fisher matrix for a given set of parameters is calculated from a set of  derivatives of the mock signal with respect to each parameter, and a covariance matrix for the mock signal (obtained from `hdMockData` by default). These calculations are done with either either CAMB (the default) or CLASS in the `Fisher` class of `hdfisher/fisher.py`. We include additional functions in `hdfisher/fisher.py` to add Fisher matrices, apply a Gaussian prior on a parameter, and remove (a) parameter(s) from a Fisher matrix.

We provide a Jupyter notebook, `example_calculate_fisher_matrices.ipynb`, as an example of the Fisher matrix calculation (including the calculation of the derivates with respect to parameters) for the baseline CMB-HD forecasts. We also provide an example script, `example_calculate_fisher_derivatives.py`, that can be used to calculate the derivatives in parallel with MPI. In the example notebook, we compare the results to a pre-computed Fisher matrix provided with `hdfisher`.

In the example notebook, we use the latest version of the CMB-HD [mock data](https://github.com/CMB-HD/hdMockData) with the fiducial cosmological parameters, parameter step sizes, and CAMB accuracy settings described in the work(s) cited for the latest mock data version; this is the default behavior. You may use CLASS instead of CAMB by passing `use_class=True`. You may also provide your own set of fiducial parameter values by either passing them as a dictionary to `fiducial_params` in the `Fisher` class, or saving them in a YAML file and passing the file name to `fiducial_params`. Additionally, you may provide your own step sizes as a dictionary or in a second YAML file, and pass the dictionary or YAML file name to `step_sizes` in the `Fisher` class. See `example_calculate_fisher_matrices.ipynb` for more details and additional options. The parameter names must be valid names that can be passed to  `camb.set_params()` if you are using CAMB, or to `classy.Class().set()` if you are using CLASS.


## Note about CAMB versions

The fiducial set of CAMB parameters used in `hdfisher` includes a parameter named `lens_output_margin` starting in CAMB version 2.0.0, or named `lens_margin` in previous versions. If you use this default set of fiducial parameters, we attempt to import CAMB in order to determine which CAMB version you are using, and use the correct parameter name for that version. If CAMB cannot be imported, we default to the newer `lens_output_margin` name.


## CLASS modifications required to use the defaults in `hdfisher`

**CLASS must be modified** in order to use the default CMB-HD fiducial accuracy settings, and in order to vary the effective number of relativistic species in the Fisher forecasts. These modifications must be made *before* compiling and installing CLASS (e.g., via `make` or `pip install`). 

The instructions are given in Appendix A of Cheslog et. al. (2026) and repeated below. We will use `$CLASS_DIR` to denote the path to the cloned CLASS repository (named `class_public` by default); e.g., the directory that contains the CLASS "readme" and `explanatory.ini` files.

- In `$CLASS_DIR/source/lensing.c` ([line 124](https://github.com/lesgourg/class_public/blob/v3.3.4/source/lensing.c#L124) in version 3.3.4, or [lines 130-131](https://github.com/lesgourg/class_public/blob/v3.4.0/source/lensing.c#L130) in version 3.4.0), you must update the type of `num_mu` and `index_mu` to be `long long`, and update the type of `icount` to be `unsigned long long`. After these modifications, the relevant lines in the file should be:
    ```c
    
    unsigned long long icount;
    long long num_mu , index_mu;
    
    
    ```
  This is necessary in order to use `l_max_scalars` above approximately 14,000 when `accurate_lensing=1`.

- In `$CLASS_DIR/source/input.c`, you must comment out the following line ([line 2470](https://github.com/lesgourg/class_public/blob/v3.3.4/source/input.c#L2470) in version 3.3.4 or [line 2548](https://github.com/lesgourg/class_public/blob/v3.4.0/source/input.c#L2548) in version 3.4.0):
    ```c
    
    /*
    class_test(pba->Omega0_ur<0,errmsg,"You cannot set the density of ultra-relativistic relics (dark radiation/neutrinos) to negative values. You might have input a total Neff smaller than what your massive neutrinos require minimally (around 1.02 * N_ncdm * deg_ncdm).");
    */
    
    
    ```
  This is necessary in order to vary `Neff` below approximately 3.0396 (or, equivalently, the `N_ur` parameter below zero) with three massive neutrinos (`N_ncdm=3`). 
  
- In order to use the default `sBBN file` used by `hdfisher` (available [here](https://github.com/CMB-HD/hdMockData/blob/main/hd_mock_data/data/theory/PRIMAT_Yp_DH_ErrorMC_2021_CLASS.dat), which is a copy of the corresponding file provided by [CAMB](https://github.com/cmbant/CAMB/blob/master/camb/PRIMAT_Yp_DH_ErrorMC_2021.dat)that has been formatted for CLASS), you *must* place a copy of that file in the `$CLASS_DIR/external/bbn/` directory. You may copy it over yourself, or follow these instructions:

  First, make sure that `hd_mock_data` has been correctly installed by running the command `python -c "from hd_mock_data import hd_data` . If nothing happens, you may proceed; otherwise, follow the instructions above to install `hd_mock_data`. 
  
  Then, run the following two commands from within your `$CLASS_DIR`:
  
  ```bash
  CLASS_SBBN_FILE=$(python -c "from hd_mock_data import hd_data; print(hd_data.class_sbbn_file())")
  
  ```
  
  ```bash
  cp $CLASS_SBBN_FILE external/bbn/
  ```
  
