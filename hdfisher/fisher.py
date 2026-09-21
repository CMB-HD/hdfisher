"""functions used to calculate a Fisher matrix"""
import os
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
import yaml
from . import utils, theory, config, dataconfig, mpi

# when using MPI, only issue a `UserWarning` once:
if mpi.rank > 0:
    warnings.simplefilter('ignore', category=UserWarning)



# functions for derivatives of theory spectra with respect to params:

def get_param_info(param_file, fisher_steps_file):
    """Dictionaries of the fiducial cosmological parameters and their
    step sizes  used to calculate the derivatives of the theory with
    respect to each varied parameter.

    If there is a parameter name in the `fisher_steps_file` without a
    corresponding fiducial value in the `param_file`, that parameter will
    not be included in the returned dictionaries.

    Parameters
    ----------
    param_file : str
        Path to a YAML file that contains the varied parameter names and
        values.
    fisher_steps_file : str
        Path to a YAML file holding the parameter step sizes.

    Returns
    -------
    fids : dict of float
        A dictionary with the parameter names as the keys holding their
        fiducial value.
    step_sizes : dict of float
        A dictionary containing a subset of the parameters in `fids`,
        holding their absolute step size used to calculate the
        derivatives.

    Warns
    ------
    If there is a step size provided for a parameter without a fiducial
    value.

    Note
    ----
    The fiducial parameters are loaded from the YAML `param_file`, with
    entries in the format `param_name: fiducial_value`. The step sizes
    for a subset of the fiducial parameters are loaded from the YAML
    `fisher_steps_file`. Each parameter (with a  `param_name`
    corresponding to a fiducial `param_name`) gets its own block, with
    two entries: `step_size`, with  a (float) value, and `step_type`,
    with a (str) value of either `'abs'`  for 'absolute' or `'rel'` for
    'relative'. If the step type is relative, the step size is assumed to
    be a fraction of the fiducial value; e.g., a relative step size of
    0.01 means increase/decrease the parameter value by 1% of its
    fiducial value.
    """
    # load fiducial params
    fids = theory.get_params(param_file=param_file)
    # load step sizes for some or all of them
    with open(fisher_steps_file, 'r') as f:
        step_info = yaml.safe_load(f)
    # create a dict of absolute step sizes
    step_sizes = {}
    for param in step_info.keys():
        # make sure we have a fiducial value for that param
        if param not in fids.keys():
            msg = (f"The parameter '{param}' in the `fisher_steps_file` will be "
                   "ignored because there is no fiducial value in the `param_file`.")
            warnings.warn(msg)
        else:
            step = step_info[param]['step_size']
            if 'rel' in step_info[param]['step_type'].lower():
                step *= fids[param]
            step_sizes[param] = step
    return fids, step_sizes


def get_varied_param_values(fids, step_sizes, include_fid=True):
    """Returns a list of tuples containing the names of parameters that
    are varied when calculating a Fisher matrix, and their values when
    they are varied away from their fiducial value. 

    Parameters
    ----------
    fids : dict of float
        A dictionary with the parameter names and their fiducial values.
    step_sizes : dict of float
        A dictionary with a subset of the parameters in `fids` and the
        absolute step sizes to take when varying the parameter values
        up or down.
    include_fid : bool, default=True
        If `True`, include a tuple `(None, None)` in the list,
        corresponding to the fiducial case.

    Returns
    -------
    param_values : list of tuple of str, float
        A list containing two tuples for each varied parameter (i.e. for
        each key in `step_sizes`). Each tuple has the form 
        `(param_name, param_value)`, where 
        `param_value = fids[param_name] +/- step_sizes[param_name]`. 
        If `include_fid = True`, there is also a tuple `(None, None)` 
        corresponding to the fiducial set of parameters.
    """
    param_values = []
    if include_fid:
        param_values.append((None, None))
    for param in step_sizes.keys():
        param_values.append((param, fids[param] - step_sizes[param]))
        param_values.append((param, fids[param] + step_sizes[param]))
    return param_values


# functions to calculate the fisher matrix:

def get_available_cmb_fisher_derivs(derivs_dir, use_H0=False):
    """Returns the available `cmb_types` and `params` for the derivatives 
    of the lensed, unlensed, or delensed theory spectra with respect to
    cosmological parameters.

    Parameters
    ----------
    derivs_dir : str
        The directory containing the derivatives.
    use_H0 : bool, default=False
        If `True`, look for derivatives that were calculated by passing `'H0'`
        to CAMB when varying the other parameters, as opposed to passing
        `'cosmomc_theta'`.

    Returns
    -------
    cmb_types : list of str
        The available types of spectra: may include `'lensed'`, `'unlensed'`, 
        and `'delensed'`, if there are files containing those names.
    params : list of str
        The available parameters, set in the Fisher parameteer YAML file
        used to calculate the derivatives.

    Note
    ----
    This function assumes that the file names of the derivatives are in the 
    format returned by `hdfisher.config.fisher_cmb_deriv_fname`, and that the 
    list of parameters will be the same for each CMB type.
    """
    # get the file name template 
    full_derivs_fname = config.fisher_cmb_deriv_fname(derivs_dir, 'cmbtype', 'param', use_H0=use_H0)
    derivs_fname = os.path.basename(full_derivs_fname) # remove abs path
    derivs_fname_root, derivs_fname_ext = os.path.splitext(derivs_fname)   # remove extension
    # filename has info about cmb_type and param; find where each is located
    derivs_info = derivs_fname_root.split('_')
    cidx = derivs_info.index('cmbtype')
    pidx = derivs_info.index('param')
    # loop through the files in `derivs_dir` and find the ones that match the
    # format above; keep track of the different CMB types and parameters.
    cmb_types = []
    params = []
    for fname in os.listdir(derivs_dir):
        if ((not use_H0) and ('useH0' in fname)) or ((use_H0) and ('useH0' not in fname)):
            pass
        else:
            root, ext = os.path.splitext(fname)
            info = root.split('_')
            if (ext == derivs_fname_ext) and ('cls_deriv' in root):
                cmb_type = info[cidx]
                if cmb_type not in cmb_types:
                    cmb_types.append(cmb_type)
                if use_H0:
                    param = '_'.join(info[pidx:-1])
                else:
                    param = '_'.join(info[pidx:])
                if param not in params:
                    params.append(param)
    return cmb_types, params


def get_available_bao_fisher_derivs(derivs_dir, use_H0=False):
    """Returns the available `params` for the derivatives of the BAO theory 
    with respect to cosmological parameters.

    Parameters
    ----------
    derivs_dir : str
        The directory containing the derivatives.
    use_H0 : bool, default=False
        If `True`, look for derivatives that were calculated by passing `'H0'`
        to CAMB when varying the other parameters, as opposed to passing
        `'cosmomc_theta'`.

    Returns
    -------
    params : list of str
        The available parameters, set in the Fisher parameter YAML file
        used to calculate the derivatives.

    Note
    ----
    This function assumes that the file names of the derivatives are in the
    format returned by `hdfisher.config.fisher_bao_deriv_fname`.
    """
    # get the file name template 
    full_derivs_fname = config.fisher_bao_deriv_fname(derivs_dir, 'param', use_H0=use_H0)
    derivs_fname = os.path.basename(full_derivs_fname) # remove abs path
    derivs_fname_root, derivs_fname_ext = os.path.splitext(derivs_fname)   # remove extension
    # filename has info about cmb_type and param; find where each is located
    derivs_info = derivs_fname_root.split('_')
    pidx = derivs_info.index('param')
    # loop through the files in `derivs_dir` and find the ones that match the
    # format above; keep track of the different parameters.
    params = []
    for fname in os.listdir(derivs_dir):
        if ((not use_H0) and ('useH0' in fname)) or ((use_H0) and ('useH0' not in fname)):
            pass
        else:
            root, ext = os.path.splitext(fname)
            info = root.split('_')
            if (ext == derivs_fname_ext) and ('bao' in root):
                if use_H0:
                    param = '_'.join(info[pidx:-1])
                else:
                    param = '_'.join(info[pidx:])
                if param not in params:
                    params.append(param)
    return params


def load_cmb_fisher_derivs(derivs_dir, cmb_types=None, params=None,
                           spectra=['tt', 'te', 'ee', 'bb', 'kk'], use_H0=False,
                           lmin=None, lmax=None, bin_edges=None, ell_ranges=None):
    """Returns a dict containing the derivatives of the theory `spectra`
    for each CMB type in `cmb_types` with respect to each parameter in `params`.

    Parameters
    ----------
    derivs_dir : str
        The directory to load the files from.
    cmb_types : list of str, default=None
        A list of types of theory spectra: can include `'lensed'`, `'unlensed'`,
        `'delensed'`. If `None`, uses all available CMB types.
    params : list of str, default=None
        A list of parameter names. If `None`, uses all available parameters.
    spectra : list of str, default=['tt', 'te', 'ee', 'bb', 'kk']
        A list of spectra to return for each combination of CMB type and parameter.
    use_H0 : bool, default=False
        If `True`, look for derivatives that were calculated by passing `'H0'`
        to CAMB when varying the other parameters, as opposed to passing
        `'cosmomc_theta'`.
    lmin, lmax : int, default=None
        If not `None`, return the derivatives in the multipole range from `lmin`
        to `lmax`. Otherwise use the full multipole range.
    bin_edges : array_like of float or int, default=None
        If not `None`, use the bin edges to bin the derivatives between `lmin`
        and `lmax`. Otherwise the unbinned derivatives are returned.
    ell_ranges : dict of list or tuple of int, default=None
        A dictionary with keys `'tt'`, `'kk'`, etc., whose values are a list or
        tuple giving the minimum and maximum multipole values for each spectrum
        in `spectra`.


    Returns
    -------
    ells : array_like of float
        The multipoles at which the derivatives are calculated.
    derivs : nested dict of array_like of float
        A nested dictionary of the form `derivs[cmb_type][param][spec]` for each
        `cmb_type` in `cmb_types`, `param` in `params`, and `spec` in `spectra`,
        which holds the derivative of that spectrum with respect to the
        parameter, d(C_ell^XY) / d(param).

    Raises
    ------
    ValueError
        If any `cmb_type` in `cmb_types` is not `'lensed'`, `'unlensed'`, or
        `'delensed'`.
    
    Note
    ----
    This function assumes that the file names of the derivatives are in the 
    format returned by `hdfisher.config.fisher_cmb_deriv_fname`, and that the 
    list of parameters will be the same for each CMB type.
    """
    valid_cmb_types = ['lensed', 'unlensed', 'delensed']
    cols = config.theo_cols # all columns in derivs file
    all_cmb_types, all_params = get_available_cmb_fisher_derivs(derivs_dir, use_H0=use_H0)
    if cmb_types is None:
        cmb_types = all_cmb_types
    if params is None:
        params = all_params
    derivs = {}
    for cmb_type in cmb_types:
        cmb_type = cmb_type.lower()
        if cmb_type not in valid_cmb_types:
            err_msg = f"You passed an unknown CMB type `'{cmb_type}'` in `cmb_types`. The options are {valid_cmb_types}."
            raise ValueError(err_msg)
        derivs[cmb_type] = {}
        for param in params:
            derivs[cmb_type][param] = {}
            derivs_fname = config.fisher_cmb_deriv_fname(derivs_dir, cmb_type, param, use_H0=use_H0)
            if not os.path.exists(derivs_fname):
                err_msg = f"The file '{derivs_fname}' does not exist. Make sure that you have calculated Fisher derivatives before calculating a new Fisher matrix."
                raise FileNotFoundError(err_msg)
            derivs_dict = utils.load_from_file(derivs_fname, cols)
            ells = derivs_dict['ells']
            # trim the multipole range
            ell_min = ells[0] if (lmin is None) else lmin
            ell_max = ells[-1] if (lmax is None) else lmax
            loc = np.where((ells >= ell_min) & (ells <= ell_max))
            ells = ells[loc]
            for s in spectra:
                derivs[cmb_type][param][s] = derivs_dict[s.lower()][loc].copy()
            if bin_edges is not None:
                ells, derivs[cmb_type][param] = utils.bin_theo_dict(ells, derivs[cmb_type][param], bin_edges, lmin=lmin, lmax=lmax, ell_ranges=ell_ranges)
    return ells, derivs


def load_bao_fisher_derivs(derivs_dir, params=None, use_H0=False):
    """Returns a dict containing the derivatives of the BAO theory with
    respect to each parameter in `params`.

    Parameters
    ----------
    derivs_dir : str
        The directory to load the files from.
    params : list of str, default=None
        A list of parameter names. If `None`, uses all available parameters.
    use_H0 : bool, default=False
        If `True`, look for derivatives that were calculated by passing `'H0'`
        to CAMB when varying the other parameters, as opposed to passing
        `'cosmomc_theta'`.


    Returns
    -------
    z : array_like of float
        The redshifts at which the derivatives were calculated.
    derivs : dict of array_like of float
        A dictionary whose keys are the parameter names (same as the elements
        of `params`, if it was provided), holding the derivative of the BAO
        theory with respect to the each parameter.

    Note
    ----
    This function assumes that the file names of the derivatives are in the
    format returned by `hdfisher.config.fisher_bao_deriv_fname`, and that all
    derivatives were calculated for the same redshifts, returned in `z`.
    """
    if params is None:
        params = get_available_bao_fisher_derivs(derivs_dir, use_H0=use_H0)
    derivs = {}
    for param in params:
        derivs_fname = config.fisher_bao_deriv_fname(derivs_dir, param, use_H0=use_H0)
        z, derivs[param] = np.loadtxt(derivs_fname, unpack=True)
    return z, derivs


def calc_cmb_fisher(cmb_cov, derivs, params, spectra=['tt', 'te', 'ee', 'bb', 'kk'], priors=None, fname=None):
    """Calculate the Fisher matrix for the given set of `params` using the
    covariance matrix `cmb_cov`, and the derivatives (`derivs`) of the theory
    `spectra` with respect to each parameter.

    Parameters
    ----------
    cmb_cov : array_like of float
        The two-dimensional covariance matrix of the CMB and CMB lensing
        power spectra, with blocks in the order given by `spectra`.
    derivs: dict of dict of array_like of float
        The derivatives of the power spectra with respect to each parameter.
        The first set of keys should be the same as the parameter names in
        `params`, and the second set should be the same as the names in
        `spectra`. Each array should be the same length as one block of 
        `cmb_cov`, i.e. the derivatives should be binned in the same way 
        as the covariance matrix.
    params : list of str
        The names of the parameters to use.
    spectra : list of str
        The names of the spectra to use.
    priors : dict of float, default=None
        If not `None`, provide any Gaussian priors with the parameter name
        as the key and the prior as the value, e.g. `{'tau': 0.007}`.
    fname : str, default=None
        If not `None`, save the Fisher matrix to the file name (including the
        absolute path) given by `fname`, which is assumed to be a `.txt` file.

    Returns
    -------
    fisher_matrix : array_like of float
        The Fisher matrix, with elements in the order given by `params`.
    """
    invcov = np.linalg.inv(cmb_cov)
    priors = {} if (priors is None) else priors
    # put derivatives into a single vector for each parameter
    dcls = {}
    for param in params:
        dcls[param] = np.hstack([derivs[param][s] for s in spectra])
    # calculate the fisher matrix
    nparams = len(params)
    fisher_matrix = np.zeros((nparams, nparams))
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            if j >= i:
                tmp = invcov @ dcls[p2]
                fisher_matrix[i, j] = dcls[p1] @ tmp
                if j > i:
                    fisher_matrix[j, i] = fisher_matrix[i, j]
                if (i == j) and (p1 in priors.keys()):
                    fisher_matrix[i, i] += (1. / priors[p1]**2)
    if fname is not None:
        utils.save_fisher_matrix(fname, fisher_matrix, params)
    return fisher_matrix


def calc_bao_fisher(bao_cov, derivs, params, priors=None, fname=None):
    """Calculate the Fisher matrix for the given set of `params` using the
    covariance matrix `bao_cov`, and the derivatives (`derivs`) of the BAO
    theory with respect to each parameter.

    Parameters
    ----------
    bao_cov : array_like of float
        The two-dimensional covariance matrix for the mock BAO data.
    derivs: dict of array_like of float
        The derivatives of the BAO theory with respect to each parameter.
        The keys are the parameter names, and each name in `params` should 
        appear as a key in `derivs`. Each array should be the same length 
        as the `bao_cov`, i.e. they must be calculated at the same redshifts.
    params : list of str
        The names of the parameters to use.
    priors : dict of float, default=None
        If not `None`, provide any Gaussian priors with the parameter name
        as the key and the prior as the value, e.g. `{'tau': 0.007}`.
    fname : str, default=None
        If not `None`, save the Fisher matrix to the file name (including the
        absolute path) given by `fname`, which is assumed to be a `.txt` file.

    Returns
    -------
    fisher_matrix : array_like of float
        The Fisher matrix, with elements in the order given by `params`.
    """
    invcov = np.linalg.inv(bao_cov)
    priors = {} if (priors is None) else priors
    # calculate the fisher matrix
    nparams = len(params)
    fisher_matrix = np.zeros((nparams, nparams))
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            if j >= i:
                tmp = invcov @ derivs[p2].copy()
                fisher_matrix[i, j] = derivs[p1].copy() @ tmp
                if j > i:
                    fisher_matrix[j, i] = fisher_matrix[i, j]
                if (i == j) and (p1 in priors.keys()):
                    fisher_matrix[i, i] += (1. / priors[p1]**2)
    if fname is not None:
        utils.save_fisher_matrix(fname, fisher_matrix, params)
    return fisher_matrix



# other useful functions:

def save_fisher_matrix(fname, fisher_matrix, params):
    """Save the Fisher matrix. 

    See Also
    --------
    utils.save_fisher_matrix

    Notes
    -----
    This function is defined here for backwards compatibility.
    """
    utils.save_fisher_matrix(fname, fisher_matrix, params)


def load_fisher_matrix(fname):
    """Load a Fisher matrix and a list of corresponding parameter names.

    See Also
    --------
    utils.load_fisher_matrix

    Notes
    -----
    This function is defined here for backwards compatibility.
    """
    fisher_matrix, params = utils.load_fisher_matrix(fname)
    return fisher_matrix, params


def get_fisher_errors(fisher_matrix, params, fname=None):
    """Returns the 1-sigma error bars on the parameters from the Fisher matrix.
    
    Parameters
    ----------
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    params : list of str
        A list of parameter names, in the same order as in the Fisher matrix.
    fname : str, default=None
        If not `None`, save the Fisher errors to the file name given by `fname`,
        assumed to be a YAML file.

    Returns
    -------
    errors : dict of float
        A dictionary with the keys given by the names in `params` and values
        holding the error bar for that parameter.
    """
    cov = np.linalg.inv(fisher_matrix.copy())
    errs = np.sqrt(np.diag(cov))
    errors = {}
    for i, param in enumerate(params):
        errors[param] = float(errs[i]) # convert from numpy dtype, for yaml saving
    if fname is not None:
        with open(fname, 'w') as f:
            yaml.dump(errors, f)
    return errors


def add_fishers(fisher_matrix1, fisher_params1, fisher_matrix2, fisher_params2,
        priors=None):
    """Returns the sum of two Fisher matrices.
    
    Parameters
    ----------
    fisher_matrix1, fisher_matrix2 : array_like of float
        The two-dimensional arrays holding the two Fisher matrices to add.
    fisher_params1, fisher_params2 : list of str
        Lists of parameter names for the parameters in `fisher_matrix1` and
        `fisher_matrix2`, respectively, in the same order that they appear
        in the rows/columns of the Fisher matrix.
    priors : None or dict of float, default=None
        If not `None`, provide any Gaussian priors with the parameter name
        as the key and the prior as the value, e.g. `{'tau': 0.007}`, to 
        be applied to the sum of the two Fisher matrices.

    Returns
    -------
    fisher_matrix : array_like of float
        A two-dimensional array holding the new Fisher matrix.
    fisher_params : list of str
        A list of parameter names for the parameters contained in the
        `fisher_matrix`, in the correct order.
    """
    # convert each Fisher matrix to a `pandas.DataFrame` instance to
    # deal with missing params, different ordering, etc.:
    df1 = pd.DataFrame(fisher_matrix1.copy(), columns=fisher_params1.copy(), index=fisher_params1.copy(), copy=True)
    df2 = pd.DataFrame(fisher_matrix2.copy(), columns=fisher_params2.copy(), index=fisher_params2.copy(), copy=True)
    df_sum = df1.add(df2, fill_value=0)
    # add any priors:
    if priors is None: 
        fisher_matrix = df_sum.values.copy()
        fisher_params = df_sum.columns.tolist().copy()
    else:
        params_with_prior = list(priors.keys())
        n_priors = len(params_with_prior)
        prior_matrix = np.zeros((n_priors, n_priors))
        for i, p in enumerate(params_with_prior):
            prior_matrix[i, i] += 1 / priors[p]**2
        df_prior = pd.DataFrame(prior_matrix.copy(), columns=params_with_prior.copy(), index=params_with_prior.copy())
        df_final = df_sum.add(df_prior, fill_value=0)
        fisher_matrix = df_final.values.copy()
        fisher_params = df_final.columns.tolist().copy()
    return fisher_matrix, fisher_params


def add_priors(fisher_matrix, fisher_params, priors):
    """Apply (a) Gaussian prior(s) to a Fisher matrix.
    
    Parameters
    ----------
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    fisher_params : list of str
        A list of parameter names, in the same order as their rows/columns
        in the Fisher matrix.
    priors : dict of float
        A dictionary containing the Gaussian prior(s) with the parameter name
        as the key and the prior as the value, e.g. `{'tau': 0.007}`.  

    Returns
    -------
    fisher_with_prior : array_like of float
        A two-dimensional array holding the Fisher matrix with the prior(s)
        applied and the parameter order unchanged.
    """
    fisher_with_prior = fisher_matrix.copy()
    for i, param in enumerate(fisher_params):
        if param in priors.keys():
            fisher_with_prior[i, i] += 1 / priors[param]**2
    return fisher_with_prior


def remove_params(fisher_matrix, fisher_params, params_to_remove):
    """Returns a Fisher matrix without the rows and columns corresponding
    to each parameter in the list `params_to_remove`.

    Parameters
    ----------
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    fisher_params : list of str
        A list of parameter names, in the same order as their rows/columns
        in the Fisher matrix.
    params_to_remove : list of str
        A list of names of parameters that will be removed from the Fisher
        matrix.

    Returns
    -------
    new_fisher_matrix : array_like of float
        A two-dimensional array holding the new Fisher matrix.
    new_fisher_params : list of str
        A list of parameter names for the parameters contained in the
        `fisher_matrix`, in the correct order.
    """
    df = pd.DataFrame(fisher_matrix.copy(), columns=fisher_params.copy(), index=fisher_params.copy(), copy=True)
    df.drop(index=params_to_remove, columns=params_to_remove, inplace=True)
    new_fisher_matrix = df.values.copy()
    new_fisher_params = df.columns.tolist().copy()
    return new_fisher_matrix, new_fisher_params


def remove_zeros(fisher_matrix, fisher_params):
    """Returns a Fisher matrix where all rows and columns that contain only
    zeros have been removed.

    Parameters
    ----------
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    fisher_params : list of str
        A list of parameter names, in the same order as their rows/columns
        in the Fisher matrix.

    Returns
    -------
    new_fisher_matrix : array_like of float
        A two-dimensional array holding the new Fisher matrix.
    new_fisher_params : list of str
        A list of parameter names for the parameters contained in the
        `fisher_matrix`, in the correct order.
    """
    nparams = len(fisher_params)
    params_to_remove = []
    for i, param in enumerate(fisher_params):
        if all(np.isclose(np.zeros(nparams), fisher_matrix[i])):
            params_to_remove.append(param)
    new_fisher_matrix, new_fisher_params = remove_params(fisher_matrix, fisher_params, params_to_remove)
    return new_fisher_matrix, new_fisher_params
 

def get_common_params(list_of_dicts):
    """Returns a list of parameter names that appear as keys for each 
    dictionary in the `list_of_dicts`.
    
    Parameters
    ----------
    list_of_dicts : list of dict
        A list of dictionaries whose keys are parameter names.

    Returns
    -------
    params : list of str
        A list of parameter names that the dictionaries share in common.
    """
    all_params = []
    for d in list_of_dicts:
        all_params += list(d.keys())
    params = []
    for param in set(all_params):
        if all([param in d.keys() for d in list_of_dicts]):
            params.append(param)
    return params



class FisherData:
    """Data needed to calculate Fisher matrices."""
    cov_spectra = ['tt', 'te', 'ee', 'bb', 'kk'] # in covmats
    
    def __init__(self, exp='hd', hd_data_version='latest', 
                 pol_only_lensing=False, hd_lmax=None, include_fg=True):
        """Initialization for a given experimental configuration.
        
        Parameters
        ----------
        exp : str, default='hd'
            The name of a CMB experiment, used to load its covariance
            matrix and set the multipole ranges used to calculate the
            Fisher matrix. Must be either `'HD'`, `'SO'`, or `'S4'`. The
            name is case-insensitive.
        hd_data_version : str, default='latest'
            The CMB-HD mock data version to use. This determines which
            CMB-HD covariance matrix, noise spectra, etc. is used. By
            default, the latest version is used. To reproduce the results
            in MacInnis et. al. (2023), use `hd_data_version='v1.0'`.
            See the `hdMockData` repository for a list of versions.
            
        Other Parameters
        ----------------
        pol_only_lensing : bool, default=False
            If `True`, the CMB-HD lensing reconstruction noise is
            calculated with only the EE and EB estimators. By default,
            the TT, TE, TB, EE, and EB estimators are used. Only
            available for `hd_data_version` >= `'v1.2'` (the default).
            Ignored if `exp` is not `'HD'`.
        include_fg : bool, default=True
            If `True`, the CMB-HD mock data was calculated including the
            effects of residual extragalactic foregrounds in temperature.
            If `False`, the data was calculated by neglecting these
            effects; only available for v1.0 CMB-HD mock data with 
            `cmb_type='lensed'` and `hd_lmax=None` or `hd_lmax=20100`.
            Ignored if `exp` is not `'HD'`.
        hd_lmax : int, default=None
            Used for CMB-HD mock data that was calculated with a lower
            maximum (CMB and lensing) multipole than the default; the
            non-default options are `1000`, `3000`, `5000`, or `10000`.
            Only available for `'v1.0'` of the CMB-HD mock data when 
            `cmb_type='delensed'` and `include_fg=True`. Ignored if `exp`
            is not `'HD'`.
        
        Raises
        ------
        ValueError
            If a set of invalid experimental parameters were passed.

        Warns
        -----
        If any of the arguments will be ignored, and if we do not have
        covariance matrices for both lensed and delensed mock spectra,
        which limits the kind (lensed or delensed) of spectra that can be
        used to calculate a Fisher matrix.
        """
        self.data = dataconfig.Data(hd_data_version=hd_data_version)
        self.hd_data_version = self.data.hd_data_version
        self.exp = self.data._check_cmb_exp(exp)
        
        self.cmb_types = ['lensed', 'delensed', 'unlensed'] # for theory spectra
        self.cov_cmb_types = self.data.cov_cmb_types[self.exp].copy() # for covmats
        
        # multipole ranges and binning:
        self.ell_ranges = deepcopy(self.data.ell_ranges[self.exp])
        self.lmin = self.data.lmins[self.exp]
        self.Lmin = self.lmin
        self.lmax = self.data.lmaxs[self.exp]
        self.Lmax = self.data.Lmaxs[self.exp]
        self.theo_lmax = self.data.theo_lmaxs[self.exp]
        self.bin_edges = self.data.load_bin_edges()
        # for CMB-HD:
        if self.exp == 'hd':
            self.data._check_hd_fgs(include_fg=include_fg)
            if not include_fg:
                self.cov_cmb_types = ['lensed']
                warnings.warn(f"There is no delensed CMB-HD covariance matrix"
                              " available for `include_fg=False`.")
            if hd_lmax is not None:
                hd_lmax = self.data._check_hd_lmax(hd_lmax, include_fg=include_fg)
                if hd_lmax < self.lmax:
                    warnings.warn("There is no lensed CMB-HD covariance matrix"
                                  f"available for `{hd_lmax = }`.")
                    self.cov_cmb_types = ['delensed']
                    self.lmax = hd_lmax
                    self.Lmax = hd_lmax
                    self.theo_lmax = hd_lmax
                    for key in self.ell_ranges:
                        self.ell_ranges[key][1] = hd_lmax
        else:
            warnings.warn("Ignoring the `hd_lmax`, `include_fg`, and "
                          f"`pol_only_lensing` arguments for {exp = }.")
        self.include_fg = include_fg
        self.hd_lmax = hd_lmax
        self.pol_only_lensing = pol_only_lensing
        # lensing noise (used to calculate delensed spectra)
        _, self.nlkk = self.data.load_cmb_lensing_noise_spectrum(self.exp, hd_Lmax=self.hd_lmax, 
                                                                 include_fg=self.include_fg, 
                                                                 pol_only_lensing=self.pol_only_lensing)
        
        if len(self.cov_cmb_types) == 1:
            cmb_type = self.cov_cmb_types[0]
            warnings.warn(f"There is only a {cmb_type} covariance matrix "
                          f"available for {exp = } and {hd_data_version = }.")
        
        
    def covmat(self, cmb_type='delensed'):
        cov = self.data.load_cmb_covmat(self.exp, cmb_type=cmb_type, 
                                        pol_only_lensing=self.pol_only_lensing,
                                        include_fg=self.include_fg, 
                                        hd_lmax=self.hd_lmax)
        return cov


class Fisher(FisherData):
    """Calculate new Fisher derivatives and matrices from the mock CMB
    and BAO covariance matrices provided with `hdfisher`.
    """

    def __init__(self, fisher_dir, param_file=None, fisher_steps_file=None,
                 fisher_params=None, use_H0=False, feedback=False,
                 use_class=False, #TODO
                 overwrite=False, **kwargs):
        """Initialization for a given experimental configuration and set
        of parameters to be included in the Fisher matrix.

        Parameters
        ----------
        fisher_dir : str
            The absolute path to a directory in which the theory and its
            derivatives used to calculate the Fisher matrix, and the
            calculated Fisher matrices, will be saved. The directory will
            be created if it does not exist.
        param_file : str or None, default=None
            The name (including the full path) of a YAML file containing
            the names and fiducial values of cosmological parameters used
            in the theory calculation, and any other names that can be
            passed to the CAMB function `camb.set_params()` (e.g.,
            accuracy parameters). Each entry in the file should have its
            own line in the format `param_name: value`.
            If `param_file=None`, the default file included with
            `hdfisher` is used.
        fisher_steps_file : str or None, default=None
            The name (including the full path) of a YAML file containing
            the step sizes to use when calculating the derivatives of the
            theory with respect to each parameter included in the Fisher
            matrix. Each parameter should have its own block with two
            entries: a `step_size` (float), giving the step size to use
            for the numerical derivatives when varying the parameter up
            or down; and the `step_type` (str), which should be
            `'absolute'`, or `'relative'` if the `step_size` was given as
            a fraction of the fiduical parameter value (e.g., a relative
            step size of 0.01 corresponds to an absolute step size of 1%
            of the fiducial value). Each parameter in the
            `fisher_steps_file` must have a fiducial value specified in
            the `param_file`.
        fisher_params : None or list of str, default=None
            An optional list of parameter names to use in the
            calculation, which must be a sub-set of the parameters in the
            `fisher_steps_file`. If `None`, all parameters in the
            `fisher_steps_file` are used.
        use_H0 : bool, default=False
            Used when the `fisher_steps_file` (or the `fisher_params`
            list) contains both the Hubble constant `'H0'` and the
            cosmoMC approximation to the angular scale of the sound
            horizon at last scattering, `cosmomc_theta` (or `theta`,
            which is defined in `hdfisher` as `100 * cosmomc_theta`).
            Only one of these parameters can be used in the theory
            calculation. If `use_H0=True`, `'H0'` is used; otherwise,
            `'cosmomc_theta'` is used.
        feedback : bool, default=False
            Used if `param_file=None` and/or `fisher_steps_file=None`. If
            `feedback=True`, the default `param_file` specifies the
            HMCode2020 + feedback non-linear model to use for the theory
            calculation, and includes a fiducial value for its feedback
            parameter, with a step size given in the default
            `fisher_steps_file`. If `False`, the  CDM-only model with
            HMCode2016 is used.
        overwrite : bool, default=False
            If `False`, any Fisher derivatives or matrices that are saved
            in the `fisher_dir` will be loaded. Otherwise, if `True`, the
            results will be re-calculated, and their files will be
            over-written by the new calculations.
        **kwargs : dict
            Keyword arguments accepted by the `FisherData` class for the
            experimental configuration, including the `exp` name and the
            `hd_data_version`.

        Raises
        ------
        ValueError
            If a set of invalid experimental parameters were passed, or
            if the `fisher_steps_file` contains a parameter that wasn't
            given in the `param_file`.

        Notes
        -----
        Copies of the `param_file` and `fisher_steps_file` will be saved
        in the `fisher_dir`.

        If you would like the option to switch between using `'H0'` and
        `'cosmomc_theta'` in your Fisher matrices, you will need to
        calculate two sets of Fisher derivatives: one that contains
        derivatives of the theory with respect to `'cosmomc_theta'` and
        fixes it when varying the other parameters, and a second that
        uses `'H0'` instead. You should (re-)initialize the `Fisher`
        class twice, with `use_H0=False` (to do the first calculation)
        and then again with `use_H0=True` (to do the second calculation),
        with all other arguments unchanged. Then, you may use either
        parameter by setting the `use_H0` flag in the `get_fisher`
        method, which will override the value set here.

        See Also
        --------
        config.fiducial_param_file : Default `param_file`
        config.fiducial_fisher_steps_file : Default `fisher_steps_file`
        """
        super().__init__(**kwargs)
        self.fisher_dir = fisher_dir
        self.overwrite = overwrite
        self.use_H0 = use_H0

        # get the fiducial parameter values and step sizes:
        self.param_file = param_file
        if self.param_file is None:
            self.param_file = self.get_param_file(feedback=feedback)
        if fisher_steps_file is None:
            fisher_steps_file = self.get_fisher_steps_file(feedback=feedback)
        self.fid_params, self.step_sizes = get_param_info(self.param_file, fisher_steps_file)

        # create the output directories, and save a copy of the
        # input param and step sizes files:
        self.theo_dir = os.path.join(self.fisher_dir, 'theory')
        self.derivs_dir = os.path.join(self.fisher_dir, 'derivs')
        self.fmat_dir = os.path.join(self.fisher_dir, 'fisher_matrices')
        # make sure no one tries to load files (above) while rank 0 saves them:
        mpi.comm.barrier()
        if mpi.rank == 0:
            utils.set_dir(self.fisher_dir)
            utils.set_dir(self.theo_dir)
            utils.set_dir(self.derivs_dir)
            utils.set_dir(self.fmat_dir)
            self.save_param_steps_values()
        mpi.comm.barrier()

        # varied parameters:
        self._has_H0 = ('H0' in self.step_sizes)
        self._H0_name = 'H0'
        if 'theta' in self.step_sizes:
            self._has_theta = True
            self._theta_name = 'theta'
        elif 'cosmomc_theta' in self.step_sizes:
            self._has_theta = True
            self._theta_name = 'cosmomc_theta'
        else:
            self._has_theta = False
            self._theta_name = None
        self.all_varied_params = list(self.step_sizes.keys()) # all
        self.fisher_params, _ = self.check_H0_theta() # default list


    # functions called during initialization:

    def get_param_file(self, feedback=False):
        """Returns the file name of the YAML file that contains the
        fiducial cosmological parameter names and values, and any other
        CAMB settings, if no `param_file` was given during initialization.
        """
        # look for an input param file in the output directory:
        input_param_file = os.path.join(self.fisher_dir, 'fiducial_params.yaml')
        if os.path.exists(input_param_file) and (not self.overwrite):
            param_file = input_param_file
        else:
            param_file = config.fiducial_param_file(feedback=feedback,
                                                    hd_data_version=self.hd_data_version)
        return param_file


    def get_fisher_steps_file(self, feedback=False):
        """Returns the file name of the YAML file that contains the
        parameter step sizes to be used when calculating the derivatives
        of the theory, if no `fisher_steps_file` was given during
        initialization.
        """
        # look for an input param file in the output directory:
        input_steps_file = os.path.join(self.fisher_dir, 'step_sizes.yaml')
        if os.path.exists(input_steps_file) and (not self.overwrite):
            fisher_steps_file = input_steps_file
        else:
            fisher_steps_file = config.fiducial_fisher_steps_file(feedback=feedback)
        return fisher_steps_file


    def save_param_steps_values(self):
        """Saves a copy of the `param_file` and `fisher_steps_file` in 
        the `fisher_dir` set during initialization.
        """
        param_values = self.fid_params.copy()
        param_steps = self.step_sizes.copy()
        # ensure float values are actually `float` (not, e.g., `numpy.float64`):
        for param, value in param_values.items():
            if type(value) not in [str, bool, int]:
                param_values[param] = float(value)
        for param, step_size in param_steps.items():
            # all step sizes should be floating point numbers
            param_steps[param] = {'step_size': float(step_size), 'step_type': 'absolute'}
        # save the files
        if mpi.rank == 0: 
            fid_params_fname = os.path.join(self.fisher_dir, 'fiducial_params.yaml')
            with open(fid_params_fname, 'w') as f:
                yaml.dump(param_values, f,  default_flow_style=False)
            step_sizes_fname = os.path.join(self.fisher_dir, 'step_sizes.yaml')
            with open(step_sizes_fname, 'w') as f:
                yaml.dump(param_steps,  f,  default_flow_style=False)


    def check_H0_theta(self, params=None, use_H0=None):
        """If step sizes were provided for both `'H0'` and `'theta'`
        (or `'cosmomc_theta'`), removes one of them from the set of
        parameters (default is all varied parameters, i.e. those with a
        step size) to be used in the calculations, based on the value of
        `use_H0` (default is value set either initialization).

        Returns list of paramemeter names and step sizes dictionary with
        that only contains one of H0, theta.
        """
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        params_list = self.all_varied_params if (params is None) else params
        if self._has_H0 and self._has_theta: # only keep one of them
            param_to_remove = self._theta_name if use_H0 else self._H0_name
            warnings.warn(f"Removing '{param_to_remove}' from the list"
                          f" of varied parameters, because `{use_H0=}`.")
            params_list = [p for p in params_list if (p != param_to_remove)]
        step_sizes = {p: step for (p, step) in self.step_sizes.items() if (p in params_list)}
        return params_list, step_sizes
            

    # functions to calculate the derivatives of theory with respect to params:

    def get_varied_param_values(self, use_H0=None):
        """Returns a list of tuples containing the names of parameters
        that are varied when calculating the Fisher derivatives, and
        their values when they are varied away from their fiducial value.
        If the derivative of the theory with respect to a given parameter
        has already been saved, it will be excluded from the list, unless
        `overwrite=True` was passed during initialization.

        See also
        --------
        fisher.get_varied_param_values
        """
        # we need to choose between H0 or theta, if both are in `fisher_params`;
        #  the choice is based on the `use_H0` flag:
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        params_list, step_sizes = self.check_H0_theta(use_H0=use_H0)
        all_param_values = get_varied_param_values(self.fid_params, step_sizes)
        if not self.overwrite: # don't re-compute theory and derivatives
            param_values = []
            for param_info in all_param_values:
                param, value = param_info
                if param is None: # fiducial case
                    step_direction = None
                else:
                    step_direction = 'up' if (value > self.fid_params[param]) else 'down'
                # check if CMB and BAO theory is already saved
                bao_fname = config.fisher_bao_theo_fname(self.theo_dir, param,
                                                         step_direction, use_H0=use_H0)
                theo_fnames = [bao_fname]
                for cmb_type in self.cmb_types:
                    fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param,
                                                         step_direction, use_H0=use_H0)
                    theo_fnames.append(fname)
                if not all([os.path.exists(fname) for fname in theo_fnames]):
                    param_values.append((param, value))
        else:
            param_values = all_param_values
        mpi.comm.barrier()
        return param_values

    
    def calculate_fisher_derivs(self, use_H0=None):
        """Calculate and save the derivatives of the theory with respect
        to the parameters. The varied parameters are set during
        initialization. The calculation can be parallelized using MPI.
        """
        # get a list of varied parameter names and values:
        param_values = self.get_varied_param_values(use_H0=use_H0)
        # distribute the theory calculations among the mpi processes, by
        #  getting a list of indices in `varied_param_values` that this MPI
        #  process will use:
        ntasks = len(param_values)
        task_idxs = mpi.distribute(ntasks, mpi.size, mpi.rank)
        # loop through the theory calculations, and save the theory for each:
        for idx in task_idxs:
            param, value = param_values[idx]
            idx_info = f'({idx+1}/{len(task_idxs)})'
            mpi_info = f'[rank {mpi.rank}]'
            # print some information about what the code is doing
            if param is None:
                print(f"{mpi_info} Calculating fiducial theory {idx_info}")
            else:
                step_dir = 'up' if (value > self.fid_params[param]) else 'down'
                print(f"{mpi_info} Calculating theory when varying '{param}' {step_dir} {idx_info}")
            self.calculate_theory_for_deriv(param, value, use_H0=use_H0)
        mpi.comm.barrier()
        # load the theory to calculate and save the derivatives:
        param_names = list(set([pval[0] for pval in param_values if (pval[0] is not None)]))
        if mpi.rank == 0:
            print(f"Calculating derivatives of the theory with respect to the following parameters: {param_names}.")
            for param in param_names:
                self.calculate_deriv(param, use_H0=use_H0)


    def calculate_theory_for_deriv(self, param, value, use_H0=None):
        """Calculate and save the CMB and BAO theory when the `param` has
        the given `value`, with the other parameters fixed to their
        fiducial values.
        """
        # will pass the `cosmo_params` dict to the `theory.Theory` class, to
        #  overide the param value from its fiducial value; we also keep track
        #  of which way the parameter is varied to use in the output file name:
        if param is None: # fiducial case
            cosmo_params = {}
            step_direction = None
        else:
            cosmo_params = {param: value}
            step_direction = 'up' if (value > self.fid_params[param]) else 'down'
            if param == 'logA': # set 'As' to `None`, so 'logA' is actually used
                cosmo_params['As'] = None
        # do the calculation:
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        theolib = theory.Theory(self.theo_lmax, self.theo_dir,
                                param_file=self.param_file, use_H0=use_H0,
                                nlkk=self.nlkk, recon_lmin=self.Lmin,
                                recon_lmax=self.Lmax, **cosmo_params)
        cmb_theo = theolib.get_theory(save=False, output_lmax=self.lmax)
        z = dataconfig.desi_redshifts()
        rs_dv = theolib.get_rs_dv(z, save=False)
        # save the theory
        header_info = f'{param} = {value}\n'
        for cmb_type in self.cmb_types:
            cmb_theo_fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, step_direction, use_H0=use_H0)
            utils.save_to_file(cmb_theo_fname, cmb_theo[cmb_type], keys=config.theo_cols, extra_header_info=header_info)
        bao_theo_fname = config.fisher_bao_theo_fname(self.theo_dir,  param, step_direction, use_H0=use_H0)
        utils.save_to_file(bao_theo_fname, {'z': z, 'rs_dv': rs_dv}, keys=['z', 'rs_dv'],  extra_header_info=header_info)


    def calculate_deriv(self, param, use_H0=None):
        """Calculate and save the derivatives of the CMB and BAO theory
        with respect to the given `param`.
        """
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        delta_param = 2 * self.step_sizes[param]
        header_info = f'{param}: fiducial = {self.fid_params[param]}, step size = {self.step_sizes[param]}\n'
        # BAO:
        bao_theo_up_fname = config.fisher_bao_theo_fname(self.theo_dir, param, 'up', use_H0=use_H0)
        z, rs_dv_up = np.loadtxt(bao_theo_up_fname, unpack=True)
        bao_theo_down_fname = config.fisher_bao_theo_fname(self.theo_dir, param, 'down', use_H0=use_H0)
        z, rs_dv_down = np.loadtxt(bao_theo_down_fname, unpack=True)
        rs_dv_deriv = (rs_dv_up - rs_dv_down) / delta_param
        bao_fname = config.fisher_bao_deriv_fname(self.derivs_dir, param, use_H0=use_H0)
        utils.save_to_file(bao_fname, {'z': z, 'rs_dv': rs_dv_deriv}, keys=['z', 'rs_dv'], extra_header_info=header_info)
        # CMB:
        for cmb_type in self.cmb_types:
            cmb_theo_up_fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, 'up', use_H0=use_H0)
            cmb_theo_up = utils.load_from_file(cmb_theo_up_fname, config.theo_cols)
            cmb_theo_down_fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, 'down', use_H0=use_H0)
            cmb_theo_down = utils.load_from_file(cmb_theo_down_fname, config.theo_cols)
            cmb_derivs = {'ells': cmb_theo_up['ells'].copy()}
            for s in self.cov_spectra:
                cmb_derivs[s] = (cmb_theo_up[s] - cmb_theo_down[s]) / delta_param
            cmb_fname = config.fisher_cmb_deriv_fname(self.derivs_dir, cmb_type, param, use_H0=use_H0)
            utils.save_to_file(cmb_fname, cmb_derivs, keys=config.theo_cols,  extra_header_info=header_info)


    # functions to calculate the Fisher matrix:

    def load_cmb_fisher_derivs(self, cmb_types=None, binned=False, use_H0=None):
        """Returns a dict containing the derivatives of the theory spectra
        for each CMB type in `cmb_types` with respect to each varied
        parameter.

        Parameters
        ----------
        cmb_types : None or list of str, default=None
            A list of types of theory spectra: can include `'lensed'`, 
            `'unlensed'`, `'delensed'`. If `None`, uses all available CMB
            types.
        binned : bool, default=False
            If `True`, bin the spectra using the same binning as the
            covariance matrix for the mock CMB spectra. If `False`, the
            spectra are unbinned.
        use_H0 : bool or None, default=None
            If `True`, look for derivatives that were calculated by
            passing `'H0'` to CAMB when varying the other parameters, as
            opposed to passing `'cosmomc_theta'`; otherwise, look for
            derivatives that were  calculated by passing `'cosmomc_theta'`.
            If `None`, use the value of `use_H0` set during initialization.

        Returns
        -------
        ells : array_like of float
            The multipoles at which the derivatives are calculated.
        derivs : nested dict of array_like of float
            A nested dictionary of the form `derivs[cmb_type][param][spec]` 
            for each `cmb_type` in `cmb_types`, `param` in the list of
            varied parameters, and `spec` in the list of spectra included
            in the covariance matrix (TT, TE, EE, BB, and kappakappa),
            which holds the derivative of that spectrum with respect to
            the parameter, d(C_ell^XY) / d(param).

        Raises
        ------
        ValueError
            If any `cmb_type` in `cmb_types` is not `'lensed'`, 
            `'unlensed'`, or `'delensed'`.
        
        See also
        --------
        fisher.load_cmb_fisher_derivs
        """
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        spectra = self.cov_spectra
        bin_edges = self.bin_edges if binned else None
        ell_ranges = self.ell_ranges if binned else None
        kwargs = {'cmb_types': cmb_types, 'spectra': spectra, 'use_H0': use_H0,
                  'bin_edges': bin_edges, 'ell_ranges': ell_ranges}
        ells, derivs = load_cmb_fisher_derivs(self.derivs_dir, **kwargs)
        return ells, derivs

    
    def load_bao_fisher_derivs(self, use_H0=None):
        """Returns a dict containing the derivatives of the BAO theory
        with respect to each varied parameter.

        Parameters
        ----------
        use_H0 : bool or None, default=None
            If `True`, look for derivatives that were calculated by
            passing `'H0'` to CAMB when varying the other parameters, as
            opposed to passing `'cosmomc_theta'`; otherwise, look for
            derivatives that were  calculated by passing `'cosmomc_theta'`.
            If `None`, use the value of `use_H0` set during initialization.

        Returns
        -------
        z : array_like of float
            The redshifts at which the derivatives were calculated.
        derivs : dict of array_like of float
            A dictionary whose keys are the parameter names, holding the 
            derivative of the BAO theory with respect to the each
            parameter.

        See also
        --------
        fisher.load_bao_fisher_derivs
        """
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        z, derivs = load_bao_fisher_derivs(self.derivs_dir, use_H0=use_H0)
        return z, derivs

    
    def calc_cmb_fisher(self, cmb_type, params=None, priors=None,
                        use_H0=None, save=False, fname=None):
        """Calculate the Fisher matrix for the given set of `params`
        using the covariance matrix for the CMB spectra of the given
        `cmb_type` and the derivatives of the CMB theory spectra with
        respect to each parameter.

        Parameters
        ----------
        cmb_type : str
            The type of CMB spectra to use in the calculation: either
            `'lensed'` or `'delensed'`. Note that, depending on the
            arguments passed during the initialization (e.g. the `exp`
            name), the CMB covariance matrix for the requested `cmb_type`
            may not exist.
        params : None or list of str, default=None
            A list of parameter names to use. Must have already
            calculated derivatives  of the theory with respect to each
            parameter. If `None`, all available parameters are used (with
            the choice between `'H0'` and `'theta'` determined by the
            value of `use_H0`).
        priors : None or dict of float, default=None
            An optional dictionary containing any Gaussian priors to be
            applied to the Fisher matrix, with the parameter name as the
            key and the width of the prior as the value.
        use_H0 : bool or None, default=None
            If `True`, the Fisher matrix is formed with derivatives that
            were calculated by passing `'H0'` to CAMB when varying the
            other parameters, as opposed to passing `'cosmomc_theta'`;
            otherwise, use derivatives that were calculated by passing
            `'cosmomc_theta'`. Note that the derivatives in either case
            must have been calculated already. If `None`, use the value
            of `use_H0` set during initialization.
        save : bool, default=False
            Whether to save the Fisher matrix in the `fisher_dir` set
            during initialization. If `True`, must also pass the `fname`.
        fname : str or None, default=None
            A file name used when saving the Fisher matrix when
            `save=True`. The file name should not contain any (absolute
            or relative) path.

        Returns
        -------
        fisher_matrix : array_like of float
            The two-dimensional Fisher matrix.
        params : list of str
            A list of parameter names, in the same order as their
            rows/columns in the Fisher matrix.

        Raises
        ------
        ValueError
            If `save=True` but `fname=None`; or if the CMB covariance
            matrix for the requested `cmb_type` does not exist, based on
            the experimental configuration set during initialization.

        See also
        --------
        fisher.calc_cmb_fisher
        """
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        params, _ = self.check_H0_theta(params=params, use_H0=use_H0)
        _, derivs = self.load_cmb_fisher_derivs(cmb_types=[cmb_type],
                                                binned=True, use_H0=use_H0)
        cmb_covmat = self.covmat(cmb_type=cmb_type)
        if save and (fname is None):
            err_msg = (f"You set `save=True` but `fname` is None: you "
                       "must provide a file name, `fname`, that will be "
                       "used to save the Fisher matrix in the directory "
                       f"`{self.fmat_dir}`.")
            raise ValueError(err_msg)
        if save and (mpi.rank == 0):
            fisher_fname = os.path.join(self.fmat_dir, fname)
        else:
            fisher_fname = None
        fisher_matrix = calc_cmb_fisher(cmb_covmat, derivs[cmb_type], params,
                                        spectra=self.cov_spectra, priors=priors,
                                        fname=fisher_fname)
        return fisher_matrix, params


    def calc_bao_fisher(self, params=None, priors=None, use_H0=None,
                        save=False, fname=None):
        """Calculate the Fisher matrix for the given set of parameters
        using the covariance matrix for the mock DESI BAO data and the
        derivatives of the BAO theory with respect to each parameter.

        Parameters
        ----------
        params : None or list of str, default=None
            A list of parameter names to use. Must have already
            calculated derivatives of the theory with respect to each
            parameter. If `None`, all available parameters are used (with
            the choice between `'H0'` and `'theta'` determined by the
            value of `use_H0`).
        priors : None or dict of float, default=None
            An optional dictionary containing any Gaussian priors to be
            applied to the Fisher matrix, with the parameter name as the
            key and the width of the prior as the value.
        use_H0 : bool or None, default=None
            If `True`, the Fisher matrix is formed with derivatives that
            were calculated by passing `'H0'` to CAMB when varying the
            other parameters, as opposed to passing `'cosmomc_theta'`;
            otherwise, use derivatives that were calculated by passing
            `'cosmomc_theta'`. Note that the derivatives in either case
            must have been calculated already. If `None`, use the value
            of `use_H0` set during initialization.
        save : bool, default=False
            Whether to save the Fisher matrix in the `fisher_dir` set
            during initialization. If `True`, must also pass the `fname`.
        fname : str or None, default=None
            A file name used when saving the Fisher matrix when
            `save=True`. The file name should not contain any (absolute
            or relative) path.

        Returns
        -------
        fisher_matrix : array_like of float
            The two-dimensional Fisher matrix.
        params : list of str
            A list of parameter names, in the same order as their
            rows/columns in the Fisher matrix.

        Raises
        ------
        ValueError
            If `save=True` but `fname=None`.

        See also
        --------
        fisher.calc_bao_fisher
        """
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        params, _ = self.check_H0_theta(params=params, use_H0=use_H0)
        _, derivs = self.load_bao_fisher_derivs(use_H0=use_H0)
        bao_covmat = dataconfig.load_desi_covmat()
        if save and (fname is None):
            err_msg = (f"You set `save=True` but `fname` is None: you "
                       "must provide a file name, `fname`, that will be "
                       "used to save the Fisher matrix in the directory "
                       f"`{self.fmat_dir}`.")
            raise ValueError(err_msg)
        if save and (mpi.rank == 0):
            fisher_fname = os.path.join(self.fmat_dir, fname)
        else:
            fisher_fname = None
        fisher_matrix = calc_bao_fisher(bao_covmat, derivs, params,
                                        priors=priors, fname=fisher_fname)
        return fisher_matrix, params
        
    
    def get_fisher(self, cmb_type='delensed', params=None, priors=None, 
                   use_H0=None, with_desi=False, save=False, fname=None):
        """Calculate the Fisher matrix for the given set of `params` using the
        covariance matrix for the CMB spectra of the given `cmb_type` and the
        derivatives of the CMB theory spectra with respect to each parameter,
        and optionally combine it with a Fisher matrix for the mock DESI BAO 
        data.

        Parameters
        ----------
        cmb_type : str
            The type of CMB spectra to use in the calculation: either 
            `'lensed'` or `'delensed'`. Note that, depending on the arguments
            passed during the initialization (e.g. the `exp` name), the 
            mock CMB covariance matrix for the requested `cmb_type` may not 
            exist.
        params : None or list of str, default=None
            A list of parameter names to use. Must have already calculated 
            derivatives  of the theory with respect to each parameter. If
            `None`, all available parameters are used (with the choice between
            `'H0'` and `'cosmomc_theta'` determined by the value of `use_H0`).
        priors : None or dict of float, default=None 
            An optional dictionary containing any Gaussian priors to be 
            applied to the Fisher matrix, with the parameter name as the key
            and the width of the prior as the value.
        use_H0 : bool or None, default=None
            If `True`, the Fisher matrix is formed with derivatives that were 
            calculated by passing `'H0'` to CAMB when varying the other 
            parameters, as opposed to passing `'cosmomc_theta'`; otherwise, 
            use derivatives that were calculated by passing `'cosmomc_theta'`. 
            Note that the derivatives in either case must have been calculated 
            already. If `None`, use the value of `use_H0` set during 
            initialization.
        with_desi : bool, default=False
            If `True`, combine the CMB Fisher matrix with the mock DESI BAO
            Fisher matrix by adding the two. Otherwise, return the CMB-only
            Fisher matrix.
        save : bool, default=False
            Whether to save the Fisher matrix in the `fisher_dir` set during
            initialization. If `True`, must also pass the `fname`.
        fname : str or None, default=None
            A file name used when saving the Fisher matrix when `save=True`.
            The file name should not contain any absolute or relative path.

        Returns
        -------
        fisher_matrix : array_like of float
            The two-dimensional Fisher matrix.
        fisher_params : list of str
            A list of parameter names, in the same order as their rows/columns
            in the Fisher matrix.

        Raises
        ------
        ValueError
            If `save=True` but `fname=None`; or if `params` was passed and 
            the list contains both `'H0'` and `'cosmomc_theta'` or `'theta'`;
            or if the mock CMB covariance matrix for the requested `cmb_type` 
            does not exist, based on the experimental configuration set
            during initialization.

        See also
        --------
        fisher.calc_cmb_fisher, fisher.Fisher.calc_cmb_fisher
        fisher.calc_bao_fisher, fisher.Fisher.calc_bao_fisher
        fisher.Fisher.get_fisher_errors
        """
        if use_H0 is None:
            use_H0 = self.use_H0
        if save and (fname is None):
            err_msg = f"You set `save=True` but `fname` is None: you must provide a file name, `fname`, that will be used to save the Fisher matrix in the directory `{self.fmat_dir}`."
            raise ValueError(err_msg)
        # check if we need to calculate a new Fisher matrix, or if we can load it:
        calc_fisher = True
        if fname is not None:
            fisher_fname = os.path.join(self.fmat_dir, fname)
            if os.path.exists(fisher_fname) and (not self.overwrite):
                fisher_matrix, fisher_params = utils.load_fisher_matrix(fisher_fname)
                calc_fisher = False
        if calc_fisher:
            if with_desi:
                cmb_fisher_matrix, cmb_fisher_params = self.calc_cmb_fisher(cmb_type, params=params, use_H0=use_H0, save=False)
                bao_fisher_matrix, bao_fisher_params = self.calc_bao_fisher(params=params, use_H0=use_H0, save=False)
                fisher_matrix, fisher_params = add_fishers(cmb_fisher_matrix, cmb_fisher_params, bao_fisher_matrix, bao_fisher_params, priors=priors) 
            else:
                fisher_matrix, fisher_params = self.calc_cmb_fisher(cmb_type, params=params, priors=priors, use_H0=use_H0, save=False)
            if save and (mpi.rank == 0):
                fisher_fname = os.path.join(self.fmat_dir, fname)
                utils.save_fisher_matrix(fisher_fname, fisher_matrix, fisher_params)
        return fisher_matrix, fisher_params


    def get_fisher_errors(self, cmb_type='delensed', params=None, priors=None, 
                          use_H0=None, with_desi=False, save=False, fname=None):
        """Returns the parameter uncertainties obtained from the Fisher matrix 
        calculated for the given set of `params` using the covariance matrix 
        for the CMB spectra of the given `cmb_type` and the derivatives of the 
        CMB theory spectra with respect to each parameter, optionally 
        combined with a Fisher matrix for the mock DESI BAO data.

        Parameters
        ----------
        cmb_type : str
            The type of CMB spectra to use in the calculation: either 
            `'lensed'` or `'delensed'`. Note that, depending on the arguments
            passed during the initialization (e.g. the `exp` name), the 
            mock CMB covariance matrix for the requested `cmb_type` may not 
            exist.
        params : None or list of str, default=None
            A list of parameter names to use. Must have already calculated 
            derivatives  of the theory with respect to each parameter, and/or
            calculated and saved a Fisher matrix using the file name given by 
            `fname`. If `params=None`, all available parameters are used (with 
            the choice between `'H0'` and `'cosmomc_theta'` determined by the 
            value of `use_H0`).
        priors : None or dict of float, default=None 
            An optional dictionary containing any Gaussian priors to be 
            applied to the Fisher matrix, with the parameter name as the key
            and the width of the prior as the value.
        use_H0 : bool or None, default=None
            If `True`, the Fisher matrix is formed with derivatives that were 
            calculated by passing `'H0'` to CAMB when varying the other 
            parameters, as opposed to passing `'cosmomc_theta'`; otherwise, 
            use derivatives that were calculated by passing `'cosmomc_theta'`. 
            Note that the derivatives in either case must have been calculated 
            already. If `None`, use the value of `use_H0` set during 
            initialization.
        with_desi : bool, default=False
            If `True`, combine the CMB Fisher matrix with the mock DESI BAO
            Fisher matrix by adding the two. Otherwise, return the CMB-only
            Fisher matrix.
        save : bool, default=False
            Whether to save the Fisher matrix in the `fisher_dir` set during
            initialization. If `True`, must also pass the `fname`.
        fname : str or None, default=None
            A file name used when saving the Fisher matrix when `save=True`.
            The file name should not contain any absolute or relative path.

        Returns
        -------
        fisher_errors : dict of float
            A dictionary with the parameter names as the keys and their
            uncertainties as the values.

        Raises
        ------
        ValueError
            If `save=True` but `fname=None`; or if `params` was passed and 
            the list contains both `'H0'` and `'cosmomc_theta'` or `'theta'`;
            or if the mock CMB covariance matrix for the requested `cmb_type` 
            does not exist, based on the experimental configuration set
            during initialization.

        See also
        --------
        fisher.calc_cmb_fisher, fisher.Fisher.calc_cmb_fisher
        fisher.calc_bao_fisher, fisher.Fisher.calc_bao_fisher
        fisher.Fisher.get_fisher
        """
        if use_H0 is None:
            use_H0 = self.use_H0
        fisher_matrix, fisher_params = self.get_fisher(cmb_type=cmb_type, params=params, priors=priors, use_H0=use_H0, with_desi=with_desi, save=save, fname=fname)
        fisher_errors = get_fisher_errors(fisher_matrix, fisher_params)
        return fisher_errors


    def load_example_hd_fisher(self, cmb_type='delensed', use_H0=False, with_desi=False):
        """Returns an example CMB-HD Fisher matrix that was calculated with
        the correct `hd_data_version`, and a list of the parameters it 
        contains. 

        See Also
        --------
        dataconfig.Data.load_example_hd_fisher
        """
        fisher_matrix, fisher_params = self.data.load_example_hd_fisher(cmb_type=cmb_type, 
                                                                        use_H0=use_H0, 
                                                                        with_desi=with_desi)
        return fisher_matrix, fisher_params

