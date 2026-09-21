import os
import warnings
import numpy as np
from hd_mock_data import hd_data
from . import utils

# columns in power spectra files:
theo_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk']
noise_cols = ['ells', 'tt', 'te', 'ee', 'bb'] # for CMB noise


def data_path(relative_path):
    """Absolute path to a file provided with the `hdfisher` package
    (i.e., a file in the `hdfisher/data/` directory of the `hdfisher`
    repository), or a directory containing such files.

    Parameters
    ----------
    relative_path : str
        The path to a file or directory, relative to the `data`
        directory. E.g., the `relative_path` of the "readme" file in the
        `data` directory is `'README.md'`.

    Returns
    -------
    absolute_path : str
        The absolute path to the file or directory.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
    absolute_path = os.path.join(data_dir, relative_path)
    return absolute_path


def fiducial_param_file(feedback=False, hd_data_version='latest'):
    """Absolute path to the YAML file holding the fiduical parameters
    (including cosmological and accuracy parameters) passed to CAMB when
    calculating the CMB and BAO theory.

    Parameters
    ----------
    feedback : bool, default=False
        If `True`, the parameter file sets the CAMB `halofit_version`
        to `mead2020_feedback`, i.e. uses the HMCode 2020 + baryonic
        feedback non-linear model. Otherwise, the HMCode 2016 CDM-only
        model is used by setting `halofit_version` to `mead2016`.
    hd_data_version : str, default='latest'
        The CMB-HD data version to use.  By default, the latest version is
        used. To reproduce the results in  MacInnis et. al. (2023), use 
        `hd_data_version='v1.0'`. See the `hdMockData` repository for a
        list of versions.

    Returns
    -------
    fname : str
        The absolute path to the file.

    Notes
    -----
    The parameter file does not contain `lmax`.
    """
    fid_params_dir = data_path('fiducial_params')
    feedback_info = '_feedback' if feedback else ''
    if hd_data_version in ['v1.0', 'v1.1']:
        v = 'v1.0'
    else:
        hd_datalib = hd_data.HDMockData(version=hd_data_version)
        v = hd_datalib.get_compatible_version(hd_datalib.camb_theo_versions, 
                                              'CAMB parameters')
    fname = os.path.join(fid_params_dir, f'fiducial_params{feedback_info}_{v}.yaml')
    return fname


def fiducial_fisher_steps_file(feedback=False):
    """Absolute path to the YAML file holding the fiduical parameter step
    sizes used to calculate the Fisher matrices.

    Parameters
    ----------
    feedback : bool, default=False
        If `True`, the file includes a step size for the HMCode 2020
        baryonic feedback parameter, `HMCode_logT_AGN`. Otherwise this
        parameter is excluded.

    Returns
    -------
    fname : str
        The absolute path to the file.
    """
    fid_steps_dir = data_path('fisher_step_sizes')
    feedback_info = '_feedback' if feedback else ''
    fname = os.path.join(fid_steps_dir, f'fiducial_step_sizes{feedback_info}.yaml')
    return fname


def camb_theo_fnames(theo_dir, theo_root=None, mkdir=False):
    """Returns a dictionary containing file names used to save the CAMB theory.
    
    Parameters
    ----------
    theo_dir : str
        The absolute path to the directory in which the CAMB output will 
        be saved.
    theo_root : str or None, default=None
        If provided, all file names will begin with the `theo_root`.
    mkdir : bool, default=False
        If the `theo_dir` does not exist, and `mkdir=True`, the directory
        will be created; otherwise, the user is only warned that the
        directory does not exist.

    Returns
    -------
    fnames : dict of str
        A dictionary with a key `'params'` holding the file name used
        to save information about the `camb.CAMBParams` instance used to
        calculate the theory; keys `'lensed'`, `'unlensed'`, and `'delensed'`
        holding the file name used when saving the corresponding CMB theory;
        a key `'bao'` holding the file name used to save the BAO theory; and
        a key `'clkk_res'` holding the file name used to save the residual
        CMB lensing power.
    """
    # if `theo_dir` does not exist, make it or warn the user:
    if not os.path.exists(theo_dir):
        if mkdir:
            utils.set_dir(theo_dir)
        else:
            msg = f"The `theo_dir` '{theo_dir}' does not exist. Pass `mkdir=True` to automatically create it."
            warnings.warn(msg)
    root = '' if theo_root is None else f'{theo_root}_'
    theo_path = lambda x: os.path.join(theo_dir, x)
    file_types = {'params': 'camb_params', 'bao': 'bao_rs_dv', 'clkk_res': 'clkk_res'}
    for cmb_type in ['lensed', 'unlensed', 'delensed']:
        file_types[cmb_type] = f'{cmb_type}_cmb_cls'
    fnames = {fname: theo_path(f'{root}{ftype}.txt') for fname, ftype in file_types.items()}
    return fnames

   
def fisher_cmb_theo_fname(fisher_theo_dir, cmb_type, param, step_direction, use_H0=False):
    """Returns the file name for the file containing the theory spectra when
    the given `param` is varied, or for the fiducial spectra (if `param = None`).

    Parameters
    ----------
    fisher_theo_dir : str
        The absolute path to the directory where the theory is saved.
    cmb_type : str
        The type of theory spectra: `'lensed'`, `'unlensed'`, or `'delensed'`.
    param : str or None
        The name of the parameter that was varied from its fiducial value.
        If `None`, get the file name for the fiducial spectra.
    step_direction : str or None
        The direction in which the `param` was varied from its fiducial
        value: `'up'` or `'down'`. If `None`, `param` must also be `None`;
        this corresponds to the fiducial theory (i.e., no parameters have 
        been varied away from their fiduical value).
    use_H0 : bool, default=False
        Whether the Hubble constant is fixed instead of cosmoMC theta when 
        varying the other parameters in the Fisher derivatives calculation.

    Returns
    -------
    fname : str
        The absolute path to the file holding the theory spectra.

    Raises
    ------
    ValueError
        If `step_direction is None` but `param is not None`.
    """
    if param is None:
        fname = f'{cmb_type}_cls_fiducial'
    elif step_direction is None:
        err_msg = f"You must pass 'up' or 'down' as the `step_direction` argument for `param = '{param}'`."
        raise ValueError(err_msg)
    else:
        fname = f'{cmb_type}_cls_{param}_{step_direction}'
    if use_H0:
        fname = f'{fname}_useH0'
    fname = os.path.join(fisher_theo_dir, f'{fname}.txt')
    return fname


def fisher_cmb_deriv_fname(fisher_derivs_dir, cmb_type, param, use_H0=False):
    """Returns the name of the file containing the derivatives of the
    CMB theory power spectra with respect to the given `param`.

    Parameters
    ----------
    fisher_derivs_dir : str
        The absolute path to the directory where the derivatives are saved.
    cmb_type : str
        The type of theory spectra: `'lensed'`, `'unlensed'`, or `'delensed'`.
    param : str
        The name of the parameter that was varied.
    use_H0 : bool, default=False
        Whether the Hubble constant is fixed instead of cosmoMC theta when 
        varying the other parameters in the Fisher derivatives calculation.

    Returns
    -------
    fname : str
        The absolute path to the file holding the theory derivatives.
    """
    fname = f'{cmb_type}_cls_deriv_{param}'
    if use_H0:
        fname = f'{fname}_useH0'
    fname = os.path.join(fisher_derivs_dir, f'{fname}.txt')
    return fname

   
def fisher_bao_theo_fname(fisher_theo_dir, param, step_direction, use_H0=False):
    """Returns the file name for the file containing the BAO theory when
    the given `param` is varied, or for the fiducial theory (if `param = None`).

    Parameters
    ----------
    fisher_theo_dir : str
        The absolute path to the directory where the theory is saved.
    param : str or None
        The name of the parameter that was varied from its fiducial value.
        If `None`, get the file name for the fiducial spectra.
    step_direction : str or None
        The direction in which the `param` was varied from its fiducial
        value: `'up'` or `'down'`. If `None`, `param` must also be `None`;
        this corresponds to the fiducial theory (i.e., no parameters have 
        been varied away from their fiduical value).
    use_H0 : bool, default=False
        Whether the Hubble constant is fixed instead of cosmoMC theta when 
        varying the other parameters in the Fisher derivatives calculation.

    Returns
    -------
    fname : str
        The absolute path to the file holding the theory.

    Raises
    ------
    ValueError
        If `step_direction is None` but `param is not None`.
    """
    if param is None:
        fname = f'bao_rs_dv_fiducial'
    elif step_direction is None:
        err_msg = f"You must pass 'up' or 'down' as the `step_direction` argument for `param = '{param}'`."
        raise ValueError(err_msg)
    else:
        fname = f'bao_rs_dv_{param}_{step_direction}'
    if use_H0:
        fname = f'{fname}_useH0'
    fname = os.path.join(fisher_theo_dir, f'{fname}.txt')
    return fname


def fisher_bao_deriv_fname(fisher_derivs_dir, param, use_H0=False):
    """Returns the file name for the file containing the derivatives of the
    BAO theory with respect to the given `param`.

    Parameters
    ----------
    fisher_derivs_dir : str
        The absolute path to the directory where the derivatives are saved.
    param : str or None
        The name of the parameter that was varied.
    use_H0 : bool, default=False
        Whether the Hubble constant is fixed instead of cosmoMC theta when 
        varying the other parameters in the Fisher derivatives calculation.

    Returns
    -------
    fname : str
        The absolute path to the file holding the theory derivatives.
    """
    fname = f'bao_rs_dv_deriv_{param}'
    if use_H0:
        fname = f'{fname}_useH0'
    fname = os.path.join(fisher_derivs_dir, f'{fname}.txt')
    return fname
