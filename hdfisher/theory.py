"""Calculate CMB and BAO theory to make a Fisher matrix"""
import os
import warnings
from copy import deepcopy
import numpy as np
import yaml
import camb
from . import mpi, utils, config


# ----- theory parameters: -----

def get_param_dict(param_dict_or_file=None, use_fiducial=True,
                   param_aliases=True, use_class=False,
                   param_file=None):
    """Get a dictionary of parameter names and values.

    Parameters
    ----------
    param_dict_or_file : str or dict or None, default=None
        Either a dictionary of parameter names and values, or the path to
        a YAML file containing the parameter names and values.
    use_fiducial : bool, default=True
        Whether to use the fiducial set of parameters when no parameter
        dictionary or file is provided.
    param_aliases : bool, default=True
        Whether to add the `hdfisher` parameter "aliases" to the dictionary.
    use_class : bool, default=False
        Whether to use CLASS instead of CAMB.

    Returns
    -------
    param_dict : dict or None
        Dictionary of CAMB or CLASS parameter names and values, and any
        `hdfisher` parameter "aliases" if `param_aliases=True`.
        Will return `None` if `param_dict_or_file=None`,
        `param_file=None`, and `use_fiducial=False`.

    Other Parameters
    ----------------
    param_file : str or None, default=None
        A path to a YAML file containing the parameter names and values.
        Available for backwards compatibility. Only used if
        `param_dict_or_file=None`.

    See Also
    --------
    config.fiducial_params
    config.add_param_aliases
    """
    param_dict = None
    if isinstance(param_dict_or_file, dict):
        param_dict = param_dict_or_file.copy()
    elif param_dict_or_file is not None: # assume it's a file name
        param_dict = utils.load_yaml(param_dict_or_file)
    elif param_file is not None:
        param_dict = utils.load_yaml(param_file)
    elif use_fiducial:
        param_dict = config.fiducial_params(use_class=use_class)
    if param_dict is not None:
        if param_aliases:
            param_dict = utils.add_param_aliases(param_dict, use_class=use_class, replace=False)
        else:
            param_dict = utils.remove_param_aliases(param_dict, use_class=use_class)
    return param_dict


def get_params(param_file=None):
    """Returns a dictionary of cosmological parameter names and values, 
    obtained from the given `param_file`, or from the fiducial parameter
    file if no `param_file` is provided.
    
    Parameters
    ----------
    param_file : str, default=None
        The file name, including the absolute path, of a YAML file that
        contains the cosmological parameter names and values. If not
        provided, the default/fiducial values are used.

    Returns
    -------
    params : dict of float
        A dictionary with the parameter names as keys holding their values.

    Notes
    -----
    Included for backwards compatibility.
    """
    if param_file is None:
        params = config.fiducial_params()
    else:
        params = utils.load_yaml(param_file)
    return params


def set_cosmo_params(params=None, use_H0=False, use_class=False,
                     param_file=None, **cosmo_params):
    """Load the parameter values saved in the `param_file` and return a
    dictionary that can be passed to either CAMB or CLASS (see the "Notes"
    section below).

    Parameters
    ----------
    params : str or dict or None, default=None
        Either a dictionary of parameter names and values, or the path to
        a YAML file containing the parameter names and values. If not
        provided (and a `param_file` is also not provided), the default
        set of fiducial parameters is used.
    use_H0 : bool, default=False
        Whether to use the Hubble constant instead of `cosmomc_theta`
        (for CAMB) or `theta_s_100` (for CLASS).
    use_class : bool, default=False
        Whether to use CLASS instead of CAMB.
    **cosmo_params : dict of float
        An optional dictionary of parameter names and values to override
        the values passed, or add additional parameters.

    Returns
    -------
    p : dict
        A dictionary of the parameter names and values.

    Other Parameters
    ----------------
    param_file : str or None, default=None
        Path to a YAML file that contains the parameter names and values.
        Available for backwards compatibility; use `params` instead.

    Notes
    -----
    The returned dictionary does not contain a key `lmax`/`l_max_scalars`
    for the maximum multipole for the calculation; this is added in the
    function `set_camb_params` or `set_class_params`.
    """
    params_dict = get_param_dict(param_dict_or_file=params, 
                                 use_class=use_class,
                                 param_file=param_file)
    p = {**params_dict, **cosmo_params}
    p = utils.remove_param_aliases(p, use_class=use_class)
    # if both H0 and theta are specified, can only provide one
    if use_H0:
        p.pop('cosmomc_theta', None)
        p.pop('theta_s_100', None)
    else:
        p.pop('H0', None)
        p.pop('h', None)
    return p


def set_camb_params(lmax, params=None, use_H0=False, param_file=None,
                    **cosmo_params):
    """Returns a `CAMBparams` instance with the requested cosmological
    and accuracy parameters.

    Parameters
    ----------
    lmax : int
        The maximum multipole for the theory calculation.
    params : str or dict or None, default=None
        Either a dictionary of parameter names and values, or the path to
        a YAML file containing the parameter names and values. If not
        provided (and a `param_file` is also not provided), the default
        set of fiducial parameters is used.
    use_H0 : bool, default=False
        Pass the Hubble constant instead of `cosmomc_theta` to CAMB, if
        both are provided.
    **cosmo_params : dict of float
        An optional dictionary of parameter names and values to override
        the values passed, or add additional parameters.

    Returns
    -------
    pars : camb.model.CAMBparams
        A `CAMBparams` instance.

    Other Parameters
    ----------------
    param_file : str or None, default=None
        Path to a YAML file that contains the parameter names and values.
        Available for backwards compatibility; use `params` instead.
    """
    input_params = set_cosmo_params(params=params, use_H0=use_H0,
                                    param_file=param_file, **cosmo_params)
    input_params['lmax'] = int(lmax + 500)
    pars = camb.set_params(**input_params)
    return pars


def set_class_params(lmax, params=None, use_H0=False, **cosmo_params):
    """Returns a dictionary of cosmological, accuracy, and any additional
    parameters parameters that CLASS accepts.

    Parameters
    ----------
    lmax : int
        The maximum multipole for the theory calculation.
    params : str or dict or None, default=None
        Either a dictionary of parameter names and values, or the path to
        a YAML file containing the parameter names and values. If not
        provided (and a `param_file` is also not provided), the default
        set of fiducial parameters is used.
    use_H0 : bool, default=False
        Pass the Hubble constant instead of `theta_s_100` to CLASS, if
        both are provided.
    **cosmo_params : dict of float
        An optional dictionary of parameter names and values to override
        the values passed, or add additional parameters.

    Returns
    -------
    class_params : dict
        A dictionary of CLASS parameter names and values which can be
        passed to the `classy.Class.set` method.
    """
    class_params = set_cosmo_params(params=params, use_H0=use_H0,
                                    use_class=True, **cosmo_params)
    class_params['l_max_scalars'] = int(lmax + 500)
    # warn the user if these parameters are likely to cause an error
    # if CLASS has not been modified:
    _warn_about_class_lmax(class_params)
    _warn_about_class_Neff(class_params)
    return class_params


def _warn_about_class_lmax(class_params, max_lmax=14000):
    lmax = class_params.get('l_max_scalars', 0)
    accurate_lensing = class_params.get('accurate_lensing', 0)
    if (lmax >= max_lmax) and (accurate_lensing > 0) and (mpi.rank == 0):
        msg = (f"`l_max_scalars = {lmax}`: the CLASS source code must be "
               "modified before using a value of `l_max_scalars` higher than "
               f"approximately {max_lmax} with `accurate_lensing=1`. If you "
               "have already modified CLASS to allow such a calculation, you "
               "may ignore this message. Otherwise, you must follow the "
               "instructions provided in the README file of `hdfisher` and "
               "given in Appendix A of Cheslog et. al. (2026).")
        warnings.warn(msg)


def _warn_about_class_Neff(class_params, min_Neff=3.0396):
    N_ur = class_params.get('N_ur')
    Neff = class_params.get('Neff')
    param_name = None
    param_val = None
    min_val = None
    if (N_ur is not None) and (N_ur < 0):
        param_name = 'N_ur'
        param_val = f'{N_ur = }'
        min_val = 0
    elif (Neff is not None) and (Neff < min_Neff):
        param_name = 'Neff'
        param_val = f'{Neff = }'
        min_val = f'approximately {min_Neff}'
    if param_name is not None:
        msg = (f"{param_val}: the CLASS source code must be modified before "
               f"using a value of {param_name} below {min_val}. If you have "
               "already modified CLASS to allow such a calculation, you may "
               "ignore this message. Otherwise, you must follow the "
               "instructions provided in the README file of `hdfisher` and "
               "given in Appendix A of Cheslog et. al. (2026).")
        warnings.warn(msg)


# ----- CMB and BAO theory -----

def get_camb_bao_rs_dv(camb_params, z, camb_results=None):
    """Returns the theoretical BAO quantity r_s/d_V(z) at the given 
    redshifts calculated by CAMB.

    Parameters
    ----------
    camb_params : camb.model.CAMBparams
        The `camb.model.CAMBparams` instance to be used in the
        calculation.
    z : array_like of float
        An array of redshifts at which to calculate r_s/d_V(z).
    camb_results : camb.results.CAMBdata, default=None
        A `camb.results.CAMBdata` instance that has already been
        initialized. If `None`, `camb.get_background(camb_params)` will
        be called.
    
    Returns
    -------
    rs_dv : array_like of float
        The values of r_s/d_V(z) for each redshift in `z`.
    """
    if camb_results is None:
        camb_results = camb.get_background(camb_params)
    rs_dv = camb_results.get_BAO(z, camb_params)[:,0]
    return rs_dv


def get_class_bao_rs_dv(class_results, z):
    """Returns the theoretical BAO quantity r_s/d_V(z) at the given
    redshifts calculated by CLASS.

    r_s is the comoving sound horizon at baryon drag (r_drag); d_V is the
    volume-averaged distance d_V(z) = [D_M(z)^2 * c*z/H(z)]^(1/3), with
    D_M = (1+z) * D_A, and D_A is the comoving angular-diameter distance.

    Parameters
    ----------
    class_results : classy.Class
        A `Class` instance on which `.compute()` has already been called.
    z : array_like of float
        Redshift(s), all > 0, at which to evaluate r_s/d_V(z).

    Returns
    -------
    rs_dv : array_like of float
        The values of r_s/d_V(z) for each redshift in `z`.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    rs = class_results.rs_drag() # Mpc
    d_a = class_results.angular_distance(z) # Mpc (physical D_A)
    hubble = class_results.Hubble(z) # 1/Mpc, i.e. H(z)/c
    d_m = (1.0 + z) * d_a # Mpc (comoving angular-diameter distance)
    d_v = (d_m**2 * z / hubble)**(1.0 / 3.0) # Mpc
    rs_dv = rs / d_v
    return rs_dv


def get_bao_rs_dv(camb_params, z, camb_results=None):
    """Returns the theoretical BAO quantity r_s/d_V(z) at the given
    redshifts calculated by CAMB.

    See Also
    --------
    get_camb_bao_rs_dv

    Notes
    -----
    This function is defined here for backwards compatibility.
    """
    return get_camb_bao_rs_dv(camb_params, z, camb_results=camb_results)



def get_spectra(params, lmax, results=None, use_class=False, raw_cl=True,
                cmb_types=['lensed', 'unlensed'], **kwargs):
    """Returns the theoretical lensed and/or unlensed CMB TT, EE, BB, and
    TE power spectra and the lensing convergence power spectrum.

    Parameters
    ----------
    params : camb.model.CAMBparams or dict
        An instance of `camb.model.CAMBparams` if `use_class=False`, or
        a dictionary of CLASS parameter names and values that can be
        passed to the `classy.Class.set` method.
    lmax : int
        The maximum multipole of the spectra to be returned. This cannot
        be higher than the value used in the calculation.
    results : camb.results.CAMBdata or classy.Class or None, default=None
        An instance of `camb.results.CAMBdata` if `use_class=False`, or
        an instance of `classy.Class` on which `.compute()` has already
        been called. If `None`, the `params` will be used to calculate
        the `results`.
    use_class : bool, default=False
        Whether to use CLASS instead of CAMB.
    raw_cl : bool, default=True
        If `True`, returns only C_ell instead of multiplying by
        ell * (ell + 1) / 2pi for the CMB power spectra.
    cmb_types : list of str, default=['lensed', 'unlensed']
        The kinds of spectra to return.

    Returns
    -------
    theo : dict of dict of array_like of float
        A nested dictionary with the `cmb_types` as keys. Each holds
        another dictionary with keys `'tt'`, `'te'`, `'ee'`, and `'bb'`
        holding the CMB power spectra, a key 'kk' holding the lensing
        convergence power spectrum C_L^kk = [L(L+1)]^2 C_L^phiphi / 4,
        and a key `'ells'` holding the multipoles, starting from zero.

    Other Parameters
    ----------------
    **kwargs : dict, optional
        Other keyword arguments, allowed for backwards compatibility.
        - If `camb_results` is passed, `use_class=False`, and
          `results=None`, the `camb_results` will be used; ignored if
          `results` is passed.
        Any other keyword arguments will be ignored.

    See Also
    --------
    get_delensed_spectra : Delensed CMB power spectra (CAMB only).
    """
    # backwards compatibility:
    if (results is None) and (not use_class):
        results = kwargs.pop('camb_results', None)
    if len(kwargs) > 0:
        kwargs_list = ', '.join([f'{k}={v}' for (k, v) in kwargs.items()])
        warnings.warn(f"The following keyword arguments will be ignored: {kwargs_list}")
    # calculate the power spectra:
    if use_class:
        theo = get_class_spectra(params, lmax, class_results=results,
                                 raw_cl=raw_cl, cmb_types=cmb_types)
    else:
        theo = get_camb_spectra(params, lmax, camb_results=results,
                                 raw_cl=raw_cl, cmb_types=cmb_types)
    return theo


def get_camb_spectra(camb_params, lmax, camb_results=None, raw_cl=True,
                     CMB_unit='muK', cmb_types=['lensed', 'unlensed']):
    """Returns the theoretical lensed and/or unlensed CMB TT, EE, BB, and
    TE power spectra and the lensing convergence power spectrum computed
    by CAMB.

    Parameters
    ----------
    camb_params : camb.model.CAMBparams
        The `camb.model.CAMBparams` instance to be used in the
        calculation.
    lmax : int
        The maximum multipole of the spectra to be returned. This cannot
        be higher than the value used in the CAMBparams instance, which
        is `camb.model.CAMBparams.max_l`.
    camb_results : camb.results.CAMBdata or None, default=None
        A `camb.results.CAMBdata` instance. If `None`,
        `camb.get_results(camb_params)` will be called.
    raw_cl : bool, default=True
        If `True`, returns only C_ell, instead of CAMB's default
        ell * (ell + 1) * C_ell / 2pi, for the CMB power spectra.
    CMB_unit : str, default='muK'
        The units of the CMB power spectra. Must be a valid `CMB_unit`
        accepted by `camb.results.CAMBdata.get_cmb_power_spectra`.
    cmb_types : list of str, default=['lensed', 'unlensed']
        The kinds of spectra to return.

    Returns
    -------
    theo : dict of dict of array_like of float
        A nested dictionary with the `cmb_types` as keys. Each holds
        another dictionary with keys `'tt'`, `'te'`, `'ee'`, and `'bb'`
        holding the CMB power spectra, a key 'kk' holding the lensing
        convergence power spectrum C_L^kk = [L(L+1)]^2 C_L^phiphi / 4,
        and a key `'ells'` holding the multipoles, starting from zero.

    See Also
    --------
    get_delensed_spectra : Delensed CMB power spectra.
    """
    if camb_results is None:
        camb_results = camb.get_results(camb_params)
        camb_results.calc_power_spectra()
    theo_keys = {'lensed': 'total', 'unlensed': 'unlensed_total'}
    camb_spectra = ['tt', 'ee', 'bb', 'te']
    theo = {}
    ells = np.arange(lmax + 1)
    powers = camb_results.get_cmb_power_spectra(camb_params, CMB_unit=CMB_unit,
                                                raw_cl=raw_cl, lmax=lmax)
    clkk = camb_results.get_lens_potential_cls(lmax=lmax)[:,0] * 2 * np.pi / 4
    for cmb_type in cmb_types:
        theo[cmb_type] = {'ells': ells.copy(), 'kk': clkk.copy()}
        for i, s in enumerate(camb_spectra):
            theo[cmb_type][s] = powers[theo_keys[cmb_type]][:,i].copy()
            theo[cmb_type][s][:2] = 0
    return theo


def get_class_spectra(class_params, lmax, class_results=None, raw_cl=True,
                      cmb_types=['lensed', 'unlensed'], TCMB=2.7255e6):
    """Returns the theoretical lensed and/or unlensed CMB TT, EE, BB, and
    TE power spectra and the lensing convergence power spectrum computed
    by CLASS.

    Parameters
    ----------
    class_params : dict
        A dictionary of CLASS parameter names and values that can be
        passed to the `classy.Class.set` method.
    lmax : int
        The maximum multipole of the spectra to be returned. This cannot
        be higher than the value of `l_max_scalars` passed to CLASS.
    class_results : classy.Class or None, default=None
        An instance of `classy.Class` on which `.compute()` has already
        been called. If `None`, the `class_params` will be used to
        calculate the `class_results`.
    raw_cl : bool, default=True
        If `True`, returns only C_ell instead of multiplying by
        ell * (ell + 1) / 2pi for the CMB power spectra.
    cmb_types : list of str, default=['lensed', 'unlensed']
        The kinds of spectra to return.

    Returns
    -------
    theo : dict of dict of array_like of float
        A nested dictionary with the `cmb_types` as keys. Each holds
        another dictionary with keys `'tt'`, `'te'`, `'ee'`, and `'bb'`
        holding the CMB power spectra (in units of uK^2 without any
        multiplicative factors applied), a key 'kk' holding the lensing
        convergence power spectrum C_L^kk = [L(L+1)]^2 C_L^phiphi / 4,
        and a key `'ells'` holding the multipoles, starting from zero.

    Other Parameters
    ----------------
    TCMB : float, default=2.7255e6
        A default value for the CMB temperature (in uK), which is used to
        return the CMB power spectra in units of uK^2. Only used if
        `'T_cmb'` is not in `class_params`. (Note that CLASS expects
        `'T_cmb'` in units of K, but `default_TCMB` has units of uK)
    """
    # calculate results:
    if class_results is None:
        from classy import Class
        class_results = Class()
        class_results.empty()
        class_results.set(class_params)
        class_results.compute()

    # calculate spectra:
    ells = np.arange(lmax + 1)
    unlensed_cls = class_results.raw_cl(lmax)
    if 'unlensed' in cmb_types:
        class_cls['unlensed'] = unlensed_cls
    if 'lensed' in cmb_types:
        class_cls['lensed'] = class_results.lensed_cl(lmax)
    clpp = unlensed_cls['pp']
    clkk = ells**2 * (ells + 1)**2 * clpp / 4

    theo = {}
    lfact = 1 if raw_cl else ells * (ells + 1) / (2 * np.pi)
    if 'T_cmb' in class_params:
        TCMB = class_params['T_cmb'] * 1e6
    for cmb_type in cmb_types:
        theo[cmb_type] = {'ells': ells.copy(), 'kk': clkk.copy()}
        for s in ['tt', 'ee', 'bb', 'te']:
            theo[cmb_type][s] = class_cls[cmb_type][s].copy() * lfact * TCMB**2
            theo[cmb_type][s][:2] = 0
    return theo


def get_delensed_spectra(camb_params, lmax, lensing_noise, Lmin, Lmax=None, 
        camb_results=None, raw_cl=True, CMB_unit='muK'):
    """Returns the theoretical delensed CMB TT, EE, BB, and TE power spectra 
    and the lensing convergence power spectrum computed by CAMB, given the 
    expected noise on the lensing reconstruction.

    Parameters
    ----------
    camb_params : camb.model.CAMBparams
        The `camb.model.CAMBparams` instance to be used in the calculation.
    lmax : int
        The maximum multipole of the spectra to be returned. This cannot be
        higher than the value used in the CAMBparams instance, which is 
        `camb.model.CAMBparams.max_l`.
    lensing_noise : array_like of float
        The expected lensing reconstruction noise. This should have elements 
        corresponding to lensing multipoles from L = 0 (even if `Lmin` > 0) 
        to L = `Lmax`.  It should be passed as the noise on the lensing
        convergence spectrum, C_L^kk = [L(L+1)]^2 C_L^phiphi / 4.
    Lmin : int
        The minimum multipole that will be used in the lensing reconstruction.
        The lensing noise will be set to infinity below this multipole in the
        calculation.
    Lmax : int, default=None
        The maximum multipole that will be used in the lensing reconstruction.
        The lensing noise will be set to infinity above this multipole in the
        calculation. If `None`, the value of `lmax` will be used.
    camb_results : camb.results.CAMBdata, default=None
        A `camb.results.CAMBdata` instance that has already been initialized.
        If `None`, it will be initialized within this function.
    raw_cl : bool, default=True
        If `True`, returns only C_ell, instead of ell * (ell + 1) * C_ell / 2pi,
        for the CMB power spectra.
    CMB_unit : str, default='muK'
        The units of the CMB power spectra. Must be a valid `CMB_unit` that 
        may be passed to `camb.results.CAMBdata.get_cmb_power_spectra`.

    Returns
    -------
    theo : dict of array_like of float
        A dictionary with keys `'tt'`, `'te'`, `'ee'`, and `'bb'` holding the
        delensed CMB power spectra, a key 'kk' holding the lensing power 
        spectrum as C_L^kk = [L(L+1)]^2 C_L^phiphi / 4, and a key `'ells'` 
        holding the multipoles, starting from zero.
    """
    if Lmax is None:
        Lmax = lmax
    if camb_results is None:
        camb_results = camb.get_results(camb_params)
        camb_results.calc_power_spectra()
    camb_spectra = ['tt', 'ee', 'bb', 'te']
    ells = np.arange(lmax + 1)
    clkk = camb_results.get_lens_potential_cls(lmax=camb_params.max_l)[:,0] * 2 * np.pi / 4
    # get the residual lensing power
    clkk_res = get_residual_lensing(clkk, lensing_noise, Lmin, Lmax, camb_params.max_l) * 4 / (2 * np.pi)
    # use it to get the delensed spectra
    powers = camb_results.get_lensed_cls_with_spectrum(clkk_res, lmax=lmax, CMB_unit=CMB_unit, raw_cl=raw_cl)
    theo = {'ells': ells.copy(), 'kk': clkk[:lmax+1].copy()}
    for i, s in enumerate(camb_spectra):
        theo[s] = powers[:,i].copy()
        theo[s][:2] = 0
    return theo
    


def get_residual_lensing(cl, nl, lmin, lmax, lmax_calc):
    """Returns the residual lensing power from L=0 to L=`lmax_calc`, given 
    the expected lensing reconstruction noise.
    
    Parameters
    ----------
    cl, nl : array_like of float
        The theory lensing convergence spectrum and the expected lensing
        reconstruction noise. `cl` should range from L = 0 to at least
        `lmax_calc`, and `nl` should range from L = 0 to at least `lmax`
        (note that only values of `nl` between `lmin` and `lmax` are used).
    lmin, lmax : int
        The minimum and maximum lensing multipoles to use.
    lmax_calc : int
        The maximum multipole to use for the output residual lensing power.


    Returns
    -------
    cl_res : array_like of float
        The residual lensing power starting at L = 0 and ending at `lmax_calc`,
        in the same convention as the input lensing and noise spectra.
    """
    # set the lensing noise to inf outside of the range `lmin`, `lmax`, 
    # up to the maximum multipole used for the lensing calculation
    noise = np.zeros(int(lmax_calc)+1)
    noise[:int(lmax)+1] = nl[:int(lmax)+1].copy()
    noise[:int(lmin)] = np.inf
    noise[int(lmax)+1:] = np.inf
    # wiener filter the signal with the noise
    filt = cl[:int(lmax_calc)+1] / (cl[:int(lmax_calc)+1] + noise)
    cl_filt = cl[:int(lmax_calc)+1] * filt
    # set the wiener-filtered clkk to zero outside of the range `lmin`, `lmax`
    cl_filt[:int(lmin)] = 0
    cl_filt[int(lmax)+1:] = 0
    # return the residual lensing power
    cl_res = cl[:int(lmax_calc)+1] - cl_filt
    return cl_res 
        



class Theory:
    """Calculate the theoretical CMB + lensing potential power spectra and 
    the theoretical BAO r_s / d_V values.
    """
    cmb_spectra = ['tt', 'ee', 'bb', 'te'] # order output by CAMB

    def __init__(self, lmax, output_dir, output_root=None, params=None,
                 nlkk=None, recon_lmin=None, recon_lmax=None, use_H0=False,
                 use_class=False, param_file=None, **cosmo_params):
        """Initialization of the theory calculation for a specific set of
        cosmological, accuracy, and experimental parameters.

        Parameters
        ----------
        lmax : int
            The  maximum multipole to be used for the theory spectra.
        output_dir: str
            The full path to the directory where the input parameters,
            output theory CMB and lensing power spectra, and the BAO
            theory are saved.
        output_root : str, default=None
            If not `None`, all files saved in the `output_dir` will begin
            with the `output_root`.
        params : dict or str or None, default=None
            A dictionary of CAMB or CLASS parameter names and values, or
            the path to a YAML file with the parameter names and values.
            If `None` (and `param_file=None)`, the default set of fiducial
            parameters will be used.
        nlkk : array_like of float, default=None
            The lensing reconstruction noise, used to calculate the
            delensed spectra if `use_class=False`. This should be the
            noise on the lensing convergence power spectrum, i.e.
            N_L^kappakappa = [L(L+1)]^2 N_L^phiphi / 4, starting from L=0.
        recon_lmin, recon_lmax : int, default=None
            The minimum and maximum multipoles to use for delensing (if
            `use_class=False`), corresponding to those used in the
            lensing reconstruction. If `nlkk` is not None, `recon_lmin`
            must be passed. If `recon_lmax` is `None`, `lmax` will be
            used.
        use_H0 : bool, default=False
            Used when `params` contains both the Hubble constant `'H0'`
            and the angular scale of the sound horizon at last scattering,
            either `cosmomc_theta` (or `theta`, which is defined in
            `hdfisher` as `100 * cosmomc_theta`) in CAMB or `theta_s_100`
            in CLASS. If `use_H0=True`, the Hubble constant will be used;
            otherwise, `cosmomc_theta` or `theta_s_100` is used.
        **cosmo_params : dict of float, optional
            An optional dictionary of parameter names and values to
            override the values in `params` (or the `param_file`).

        Other Parameters
        ----------------
        param_file : str or None, default=None
            Path to a YAML file containing the parameter names and values.
            Allowed for backwards compatibility; use `params` instead.
            Only used if `params=None`.

        Notes
        -----
        For the CMB lensing power spectrum, we use the convention
        C_L^kappakappa = C_L^phiphi * [L * (L + 1)]^2 / 4,
        rather than the CAMB convention,
        [C_L^kappakappa]_CAMB = (2 pi / 4) * C_L^kappakappa.

        All power spectrum arrays begin at a multipole ell = 0. The CMB power
        spectra are in C_ell's (i.e., no factor of ell * (ell + 1) / (2 * pi)
        applied), in units of uK^2.

        CLASS does not calculate delensed power spectra.
        """
        # TODO: add another note to docstring about CLASS modifications

        self.lmax = int(lmax)
        self.ells = np.arange(self.lmax+1) 
        
        self.use_class = use_class
        if self.use_class:
            self.cmb_types = ['lensed', 'unlensed']
        else:
            self.cmb_types = ['lensed', 'unlensed', 'delensed'] 

        # filenames to save/load theory:
        self.output_dir = output_dir
        self.theo_fnames = config.theo_fnames(output_dir, theo_root=output_root, use_class=use_class)

        # set up empty dict to hold theory:
        self.theo = {cmb_type: {} for cmb_type in self.cmb_types}
        # will also save the lensing potential theory and noise to use for delensing
        self.clkk = None
        # get the lensing reconstruction noise, if it was provided:
        self.nlkk = nlkk
        self.Lmin = recon_lmin
        if self.Lmin is not None:
            self.Lmin = int(self.Lmin)
        if recon_lmax is not None:
            self.Lmax = int(recon_lmax)
        else:
            self.Lmax = self.lmax

        # initialize CAMB or CLASS with the given parameters:
        params = params if (params is not None) else param_file # backwards compatibility
        self.camb_params = None
        self.class_params = None
        self._setup_boltzmann_params(params=params, use_H0=use_H0, **cosmo_params)
        self.results = None # only calculate if necessary


    def _setup_boltzmann_params(self, params=None, use_H0=False, **cosmo_params):
        """Initialize CAMB or CLASS with the given parameters."""
        if self.use_class:
            self.class_params = set_class_params(self.lmax, params=params, use_H0=use_H0, **cosmo_params)
        else:
            self.camb_params = set_camb_params(self.lmax, params=params, use_H0=use_H0, **cosmo_params)


    def get_results(self, save=False):
        """Get the `camb.results.CAMBdata` or `classy.Class` instance
        used to calculate the theory.

        See Also
        --------
        get_camb_results, get_class_results
        """
        if self.use_class:
            self.get_class_results(save=save)
        else:
            self.get_camb_results(save=save)
        return self.results


    def get_camb_results(self, save=False):
        """Get the `CAMBdata` instance from the `CAMBparams` (stored in
        the `camb_params` attribute), and store it in the `results`
        attribute.

        Parameters
        ----------
        save : bool, default=False
            If `True`, save the values set in the `CAMBparams` instance
            used to calculate the theory. The file will be saved in the
            `output_dir` passed when initializing the `Theory` class.

        Returns
        -------
        camb.results.CAMBdata
            A `CAMBdata` instance calculated from the input `CAMBparams`
        """
        if self.results is None:
            self.results = camb.get_results(self.camb_params)
            self.results.calc_power_spectra()
        # save info about the camb params that went into the theory
        if save:
            with open(self.theo_fnames['params'], 'w') as f:
                f.write(str(self.camb_params))
        return self.results


    def get_class_results(self, save=False):
        """Compute the CLASS results from the parameter dictionary stored
        in the `class_params` attribute, and store it in the `results`
        attribute.

        Parameters
        ----------
        save : bool, default=False
            If `True`, save the parameters that were passed to CLASS. The
            file is written to the `output_dir` given when initializing
            the `Theory` class.

        Returns
        -------
        classy.Class
            The `Class` instance, after `.compute()` has been called on it.
        """
        if self.results is None:
            from classy import Class
            M = Class()
            M.empty()
            M.set(self.class_params)
            M.compute()
            self.results = M
        if save:
            utils.save_yaml(self.theo_fnames['params'], self.class_params)
        return self.results


    def get_rs_dv(self, zs, overwrite=False, save=False):
        """Returns the BAO theory, i.e. r_s / d_V(z) for each redshift z in `zs`.
        
        Parameters
        ----------
        zs : array_like of float
            An array of redshifts at which the BAO theory is calculated.
        overwrite : bool, default=False
            If False, try to load the theory from disk before calculating it.
        save : bool, default=False
            If True, save the theory to the `output_dir` passed when 
            initializing the `Theory` class.

        Returns
        -------
        rs_dv : array_like of float
            The values of r_s/d_V(z) for each redshift in `zs`.
        """
        rs_dv = None 
        # check if the file exists
        fname = self.theo_fnames['bao']
        if os.path.exists(fname) and (not overwrite): # load it
            print(f'loading BAO theory from {fname}')
            z_vals, rs_dv_vals = np.loadtxt(fname, unpack=True)
            # check if all redshifts in `zs` are in the loaded `z_vals`
            if all([any(np.isclose(z_vals, z)) for z in zs]):
                # get the indices where `z_vals` has each redshift in `zs`
                idxs = [np.where(np.isclose(z_vals, z))[0][0] for z in zs]
                # only return rs_dv at the redshifts in `zs`
                rs_dv = rs_dv_vals[idxs]
        if rs_dv is None: # still need to calculate it
            if self.results is None:
                self.get_results(save=save)
            if self.use_class:
                rs_dv = get_class_rs_dv(self.results, zs)
            else:
                rs_dv = get_camb_bao_rs_dv(self.camb_params, zs, camb_results=self.results)
            if save:
                header = 'z, r_s/d_V(z)'
                np.savetxt(fname, np.column_stack([zs, rs_dv]), header=header)
        return rs_dv


    def get_theory_spectra(self, overwrite=False, save=False):
        """Get the CAMB lensed and unlensed CMB spectra and the lensing potential spectrum.
        
        Parameters
        ----------
        overwrite : bool, default=False
            If `False`, try to load the theory from the `output_dir` passed when
            initializing the `Theory` class, or use the theory spectra stored in 
            memory (in the `Theory.theo` dictionary) if it has already been 
            calculated, instead of re-computing it.
        save : bool, default=False
            If `True`, save the theory to the `output_dir` passed when
            initializing the `Theory` class.

        Returns
        -------
        Theory.theo : nested dict of array_like of float
            A nested dict with keys `'lensed'` and `'unlensed'`, each of which 
            contains another dict with the keys `'ells'`, `'tt'`, `'te'`, 
            `'ee'`, `'bb'`, and `'kk'` holding the theory spectra (C_ell's) in 
            units of uK^2, starting at ell = 0.
        """
        theo = {}
        for cmb_type in ['lensed', 'unlensed']:
            fname = self.theo_fnames[cmb_type]
            # if not overwriting, check if we already have all the spectra, or if its saved
            if all([s in self.theo[cmb_type] for s in self.cmb_spectra + ['kk']]) and (not overwrite):
                theo[cmb_type] = self.theo[cmb_type].copy()
            elif os.path.exists(fname) and (not overwrite): # load it
                print(f"loading {cmb_type} theory from {fname}")
                theo[cmb_type] = utils.load_from_file(fname, config.theo_cols)
            else: # get it from camb
                if self.results is None:
                    self.get_results(save=save)
                params = self.class_params if self.use_class else self.camb_params
                spectra_dict = get_spectra(params, self.lmax, results=self.results, 
                                           use_class=self.use_class, raw_cl=True, 
                                           cmb_types=[cmb_type])
                theo[cmb_type] = spectra_dict[cmb_type] 
            # save it
            if save:
                print(f"saving {cmb_type} theory to {fname}") 
                utils.save_to_file(fname, theo[cmb_type], keys=config.theo_cols)
            self.theo[cmb_type] = theo[cmb_type].copy()
        return theo


    def check_delensing_vars(self):
        """Check if values were passed for the arguments `nlkk` and `recon_lmin`
        when initializing the `Theory` class.
        
        Raises
        ------
        ValueError 
            If `nlkk` and/or `recon_lmin` was not passed when initializing 
            the `Theory` class.
        """
        if self.use_class:
            raise ValueError("`use_class=True`: Cannot calculate delensed spectra with CLASS.")
        info = 'you must pass the lensing reconstruction noise as `nlkk` along with the minimum multipole to use as `recon_lmin` when initializing the `Theory` class in order to calculate the delensed spectra.'
        missing_args = []
        if self.nlkk is None:
            missing_args.append('`nlkk` is `None`')
        if self.Lmin is None:
            missing_args.append('`recon_lmin` is `None`')
        if len(missing_args) > 0:
            err_msg = ' and '.join(missing_args)
            raise ValueError(f"{err_msg}: {info}")


    def get_residual_clkk(self):
        """Returns the residual lensing power, given the expected lensing 
        reconstruction noise.
        
        Raises
        ------
        ValueError 
            If `nlkk` and/or `recon_lmin` was not passed when initializing 
            the `Theory` class.

        Returns
        -------
        clkk_res : array_like of float
            The residual lensing power as 
            C_L^kappakappa = C_L^phiphi * [L * (L + 1)]^2 / 4,
            starting at ell = 0 and ending at `Theory.camb_params.max_l`,
            the maximum multipole used for the lensing calculation, which
            may be larger than `Theory.lmax`.
        """
        self.check_delensing_vars()
        if self.clkk is None: 
            if self.results is None:
                self.get_camb_results(save=False)
            self.clkk = self.results.get_lens_potential_cls(lmax=self.camb_params.max_l)[:,0] * 2. * np.pi / 4.
        clkk_res = get_residual_lensing(self.clkk, self.nlkk, self.Lmin, self.Lmax, self.camb_params.max_l)
        return clkk_res 
        


    def get_delensed_spectra(self, save=False, overwrite=False):
        """Returns the delensed theory CMB spectra given the expected 
        lensing reconstruction noise.

        Parameters
        ----------
        overwrite : bool, default=False
            If `False`, try to load the theory from the `output_dir` passed when
            initializing the `Theory` class, or use the theory spectra stored in 
            memory (in the `Theory.theo` dictionary) if it has already been 
            calculated, instead of re-computing it.
        save : bool, default=False
            If `True`, save the theory to the `output_dir` passed when
            initializing the `Theory` class.
        
        Raises
        ------
        ValueError 
            If `nlkk` was not passed when initializing the `Theory` class.

        Returns
        -------
        delensed_theo : dict of array_like of float
            A dict with keys `'tt'`, `'te'`, `'ee'`, `'bb'` holding the
            delensed CMB theory spectra (C_ell's)  in units of uK^2, starting at
            ell = 0. 
        """
        #  get the filename for saved delensed theory
        fname = self.theo_fnames['delensed']
        # if not overwriting, check if we already have all the spectra, or if its saved
        if all([s in self.theo['delensed'] for s in self.cmb_spectra + ['kk']]) and (not overwrite):
            delensed_theo = self.theo['delensed'].copy()
        if os.path.exists(fname) and (not overwrite): # load it
            print(f"loading delensed theory from {fname}")
            delensed_theo = utils.load_from_file(fname, config.theo_cols)
        else: # calculate it
            self.check_delensing_vars()
            if self.results is None:
                self.get_camb_results(save=save)
            delensed_theo = get_delensed_spectra(self.camb_params, self.lmax, self.nlkk, self.Lmin, Lmax=self.Lmax, camb_results=self.results, raw_cl=True, CMB_unit='muK')
        if save:
            print(f"saving delensed theory to {fname}") 
            utils.save_to_file(fname, delensed_theo, keys=config.theo_cols)
        self.theo['delensed'] = delensed_theo.copy()
        return delensed_theo
   

    def get_theory(self, cmb_types=None, output_lmax=None, save=False, overwrite=False):
        """Returns the lensed, unlensed, and/or delensed CMB and lensing 
        potential theory spectra.

        Parameters
        ----------
        cmb_types : str or list of str, default=None
            The type of CMB spectra to return. Must be 'lensed', 'unlensed', 
            and/or 'delensed'. Returns all three by default.
        output_lmax : int or None, default=None
            If provided and `output_lmax` is lower than the `lmax` 
            attribute, cut the spectra at a maximum multipole given by the
            `output_lmax` value.
        overwrite : bool, default=False
            If `False`, try to load the theory from the `output_dir` 
            passed when initializing the `Theory` class, or use the theory
            spectra stored in memory (in the `Theory.theo` dictionary) if
            it has already been  calculated, instead of re-computing it.
        save : bool, default=False
            If `True`, save the theory to the `output_dir` passed when
            initializing the `Theory` class.

        Returns
        -------
        theo : dict of dict of array_like of float
            A dictionary with key(s) given by the `cmb_types`. Each holds
            another dict containing one-dimensional arrays for the
            spectra, with keys `'ells'` for the multipoles, `'tt'`,
            `'te'`, `'ee'`, `'bb'` for the CMB spectra (C_ell's in units
            of uK^2), and `'kk'` for the lensing potential spectrum
            (C_L^kappakappa = C_L^phiphi * [L * (L + 1)]^2 / 4).
            Everything begins at ell = 0.

        Raises
        ------
        ValueError
            If any of the `cmb_types` are not a recognized option.
        """
        # check what `cmb_types` were passed before any calculation
        if cmb_types is None:
            cmb_types = self.cmb_types
        elif type(cmb_types) is str:
            cmb_types = [cmb_types]
        for cmb_type in cmb_types:
            if cmb_type.lower() not in self.cmb_types:
                err_msg = (f"You passed `{cmb_types = }`, but "
                           f"`'{cmb_type}'` is not a valid option. "
                           f"The valid `cmb_types` are `{self.cmb_types}`.")
                raise ValueError(err_msg)
        # get theory for each cmb_type
        theo = {}
        for cmb_type in cmb_types:
            if len(list(self.theo[cmb_type].keys())) < len(config.theo_cols):
                if 'delens' in cmb_type.lower():
                    theo_spectra = self.get_delensed_spectra(save=save, overwrite=overwrite)
                else:
                    theo_spectra = self.get_theory_spectra(save=save, overwrite=overwrite)[cmb_type]
                theo[cmb_type] = theo_spectra.copy()
            else:
                theo[cmb_type] = self.theo[cmb_type].copy()
            if output_lmax is not None:
                output_lmax = int(output_lmax)
                if output_lmax < self.lmax:
                    for key in theo[cmb_type]:
                        theo[cmb_type][key] = theo[cmb_type][key][:output_lmax+1]
        return theo
