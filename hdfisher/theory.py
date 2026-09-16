"""Calculate CMB and BAO theory to make a Fisher matrix"""
import os
import warnings
from copy import deepcopy
import numpy as np
import yaml
import camb
from . import mpi, utils, config
# CLASS import goes here, but this breaks on the seawulf login node for some reason (outdated linux package?), so I manually remove it when working on it.
try:
    from classy import Class
    _CLASS_IMPORT_ERROR = None
except Exception as _err:
    Class = None
    _CLASS_IMPORT_ERROR = _err
# ----- multipole and neutrino conventions -----

# Multipoles that CAMB/CLASS compute above the multipole to which the
# theory spectra are saved, so the lensing calculation has headroom.
LMAX_BUFFER = 500

# Contribution to N_eff from each massive (ncdm) species, used to derive
# `N_ur` from a given `Neff`:  N_ur = Neff - N_ncdm * NUR_PER_MASSIVE_NCDM.
NUR_PER_MASSIVE_NCDM = 1.0131966


# CLASS itself must be modified before it can be used for these forecasts.
# The changes are described in Appendix A of Cheslog et. al. (2026), and
# `Fisher` issues this message as a warning whenever `use_class=True`:
CLASS_MODIFICATION_WARNING = (
    "`use_class=True`: the CLASS source code must be modified before it can "
    "be used for these forecasts, as described in Appendix A of Cheslog et. "
    "al. (2026), or the calculation will fail. (1) In `source/lensing.c`, "
    "change the variables `num_mu` and `index_mu` from `int` to `long long`, "
    "and `icount` to `unsigned long long`, so that the lensing calculation "
    "does not overflow at the high multipoles used here (line 124 in CLASS "
    "v3.3.4). (2) In `source/input.c`, comment out the `class_test` that "
    "rejects a negative `N_ur` (line 2470 in CLASS v3.3.4), so that N_eff "
    "can be varied below its standard value with three massive neutrinos. "
    "The `sBBN file` entry of the CLASS parameter file must also be an "
    "absolute path to the BBN table provided in `hdfisher/data/class_inputs`; "
    "see the README."
)


def get_theory_lmax(lmax, param_file=None, use_class=False, **cosmo_params):
    """Returns the maximum multipole to which the theory spectra are
    calculated and saved.

    The value in the parameter file takes precedence over the `lmax`
    argument. The relevant key is `'lmax'` for CAMB and `'l_max_scalars'`
    for CLASS. The Boltzmann code itself is asked for this value plus
    `LMAX_BUFFER`.

    Parameters
    ----------
    lmax : int
        The maximum multipole to use if the parameter file does not
        provide one.
    param_file : str, default=None
        The file name, including the absolute path, of the YAML file
        holding the parameter names and values. If `None`, the
        default/fiducial file is used.
    use_class : bool, default=False
        If `True`, look for the CLASS key `'l_max_scalars'`; otherwise look
        for the CAMB key `'lmax'`.
    **cosmo_params : dict
        Optional overrides of the values loaded from the file.

    Returns
    -------
    theory_lmax : int
        The maximum multipole of the saved theory spectra.
    """
    params = {**get_params(param_file=param_file), **cosmo_params}
    key = 'l_max_scalars' if use_class else 'lmax'
    file_lmax = params.get(key)
    return int(file_lmax) if file_lmax is not None else int(lmax)

# ----- cosmological parameters: -----

def get_valid_param_files(param_file_dir):
    """Returns a list of file names within the `param_file_dir` that may
    contain sets of cosmological parameter names and values. 
    
    Parameters
    ----------
    param_file_dir : str
        The absolute path to the directory holding the parameter files.

    Returns
    -------
    param_files : list of str
        A list of names of different yaml files found in the `param_file_dir`.
    """
    files = os.listdir(param_file_dir)
    param_files = []
    for f in files:
        name, ext = os.path.splitext(f)
        if ('yaml' in ext.lower()) or ('yml' in ext.lower()):
            param_files.append(f)
    # warn the user if we didn't find anything
    if len(param_files) < 1:
        msg = f"Couldn't find any valid parameter YAML files in {param_file_dir}."
        warnings.warn(msg)
    return param_files


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
    """
    if param_file is None:
        # get the fiducial parameter file
        # (imported here, rather than at module level, to avoid a circular
        #  import: `dataconfig` imports `theory`)
        from . import dataconfig
        data = dataconfig.Data()
        param_file = data.fiducial_param_file()
    if not os.path.exists(param_file):
        param_set_dir, param_set_fname = os.path.split(param_file)
        if len(param_set_dir) > 1:
            param_set_name = os.path.splitext(param_set_fname)[0]
            # existing options
            param_files = get_valid_param_files(param_set_dir)
            # check if argument matches
            if param_set_fname not in param_files:
                err_msg = f"You passed `param_file = '{param_file}'`, but the only files found in the directory `{param_set_dir}` are: {param_files}."
                raise FileNotFoundError(err_msg)
        else:
            err_msg = f"Cannot find the `param_file` '{param_file}.'"
            raise FileNotFoundError(err_msg)
    # now get the params
    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def set_cosmo_params(param_file=None, use_H0=False, **cosmo_params):
    """Load the parameter values saved in the `param_file` and return a 
    dictionary that can be passed to `camb.set_params()`.
    
    Parameters
    ----------
    param_file : str, default=None
        The file name, including the absolute path, of a YAML file that
        contains the parameter names and values that can be passed to the 
        CAMB `camb.set_params()` function. If not provided, the 
        default/fiducial values are used (including accuracy parameters).
    use_H0 : bool, default=False
        Pass the Hubble constant instead of CosmoMC theta to CAMB, if both are 
        present in the parameter file.
    **cosmo_params : dict of float
        An optional dictionary of parameter names and values to override 
        the values loaded from the YAML file, or add additional parameters.
   
    Returns
    -------
    p : dict
        A dictionary of the parameter names and values.

    Note
    ----
    The returned dictionary does not contain a key `lmax` for the maximum
    multipole for CAMB to use; this is added in the function 
    `theory.set_camb_params`.
    """
    params_dict = get_params(param_file=param_file).copy()
    p = {**params_dict, **cosmo_params}
    # if both H0 and theta are specified, can only provide one
    has_hubble = False
    has_theta = False
    if 'H0' in p.keys():
        has_hubble = (p['H0'] is not None)
    if 'theta' in p.keys():
        theta_key = 'theta'
        has_theta = (p[theta_key] is not None)
    elif 'cosmomc_theta' in p.keys():
        theta_key = 'cosmomc_theta'
        has_theta = (p[theta_key] is not None)
    if (not has_hubble) and (not has_theta):
        err_msg = f"You must provide a value for either `'H0'` or `'cosmomc_theta'` to CAMB; neither was found in the `param_file` '{param_file}'."
        raise ValueError(err_msg)
    elif (use_H0 or not has_theta) and has_hubble:
        p['cosmomc_theta'] = None
    else:
        p['H0'] = None
        p['cosmomc_theta'] = p[theta_key]
        if p['cosmomc_theta'] > 0.1: # need theta, not 100 * theta
            p['cosmomc_theta'] /= 100
        if use_H0:
            msg = f"You passed `use_H0 = True`, but could not find a value for H0 in the `param_file` '{param_file}'. Passing CosmoMC theta to CAMB instead of H0."
            warnings.warn(msg)
    if 'theta' in p.keys():
        p.pop('theta', None)
    # camb wants As instead of logA
    if 'As' not in p.keys():
        p['As'] = None
    if ('logA' in p.keys()) and (p['As'] is None):
        p['As'] = np.exp(p['logA'])/1.e10
    if 'logA' in p.keys():
        p.pop('logA', None)
    # baryonic feedback
    if ('hmcode_version' not in p) and ('halofit_version' not in p):
        p['hmcode_version'] = 'mead2016'
    else:
        if 'hmcode_version' in p:
            p['halofit_version'] = p['hmcode_version']
            p.pop('hmcode_version', None)
    if 'HMCode_A_baryon' not in p:
        p['HMCode_A_baryon'] = 3.13
    if 'HMCode_eta_baryon' not in p:
        p['HMCode_eta_baryon'] = 0.603
    if 'HMCode_logT_AGN' not in p:
        if 'logTagn' in p:
            p['HMCode_logT_AGN'] = p['logTagn']
            p.pop('logTagn', None)
        else:
            p['HMCode_logT_AGN'] = 7.8
    return p


def set_camb_params(lmax, param_file=None, use_H0=False, **cosmo_params):
    """Returns a `CAMBparams` instance with the requested cosmological and accuracy parameters.
    
    Parameters
    ----------
    lmax : int
        The maximum multipole for the theory spectra.
    param_file : str, default=None
        The file name, including the absolute path, of a YAML file that
        contains the parameter names and values that can be passed to the 
        CAMB `camb.set_params()` function. If not provided, the 
        default/fiducial values are used (including accuracy parameters).
    use_H0 : bool, default=False
        Pass the Hubble constant instead of CosmoMC theta to CAMB, if both are 
        present in the parameter file.
    **cosmo_params : dict of float
        An optional dictionary of parameter names and values to override 
        the values loaded from the YAML file. 

    Notes
    -----
    If both 'H0' and 'theta' are provided, only 'theta' is used, unless `use_H0` 
    is `True`. If both 'logA' and 'As' are provided, only 'As' is used. 

    The returned `CAMBparams` instance carries three extra private
    attributes, `_fid_As`, `_fid_ns`, and `_fid_kpivot`, which record the
    fiducial primordial power-law parameters used to set it up. CAMB does
    not use these; they exist for downstream packages (e.g. hdInitPk) that
    replace the primordial power spectrum and need the fiducial power law.
    """
    input_params = set_cosmo_params(param_file=param_file, use_H0=use_H0, **cosmo_params).copy()
    # the parameter file's `lmax` (if present) sets the multipole to which
    # the theory is saved; CAMB computes `LMAX_BUFFER` multipoles beyond it
    file_lmax = input_params.get('lmax')
    theory_lmax = int(file_lmax) if file_lmax is not None else int(lmax)
    input_params['lmax'] = theory_lmax + LMAX_BUFFER
    # only pass `redshifts` to CAMB if the parameter file provided them;
    # otherwise drop the key so CAMB uses its own default
    if input_params.get('redshifts') is None:
        input_params.pop('redshifts', None)
    pars = camb.set_params(**input_params)

    pars_out = pars.copy()
    # Record the fiducial primordial power-law parameters on the returned
    # `CAMBparams` instance as private attributes. These are NOT used by
    # CAMB itself; they are provided so that downstream packages (e.g.
    # hdInitPk) that replace the primordial power spectrum can recover the
    # fiducial power law without re-parsing the parameter file.
    pars_out._fid_As     = input_params.get('As')
    pars_out._fid_ns     = input_params.get('ns')
    pars_out._fid_kpivot = input_params.get('pivot_scalar', 0.05)
    return pars_out



def set_class_params(lmax, param_file=None, use_H0=False, **cosmo_params):
    """Load the parameter values saved in the `param_file` and return a
    dictionary that can be passed to `classy.Class.set()`.

    The parameter file is read as native CLASS input: every key is passed
    through to CLASS unchanged, apart from the handling described in the
    Notes. There is no translation from CAMB parameter names.

    Parameters
    ----------
    lmax : int
        Used only if the parameter file does not provide
        `'l_max_scalars'`.
    param_file : str, default=None
        The file name, including the absolute path, of a YAML file holding
        the CLASS parameter names and values. If not provided, the
        default/fiducial values are used.
    use_H0 : bool, default=False
        Pass the Hubble constant instead of the acoustic scale to CLASS,
        if both are present in the parameter file.
    **cosmo_params : dict
        An optional dictionary of parameter names and values to override
        the values loaded from the YAML file, or add additional
        parameters.

    Returns
    -------
    class_params : dict
        A dictionary of the parameter names and values to pass to CLASS.

    Raises
    ------
    ValueError
        If neither `'H0'` nor `'theta_s_100'` is provided; if both
        `'N_ur'` and `'Neff'` are provided; if `'Neff'` is provided
        without `'N_ncdm'`; if both `'sum_m_ncdm'` and `'m_ncdm'` are
        provided; if `'sum_m_ncdm'` is provided without `'N_ncdm'`; if the
        number of entries in `'deg_ncdm'` does not match `'N_ncdm'`; or if
        the number of masses listed in `'m_ncdm'` does not match
        `'N_ncdm'`.

    Notes
    -----
    Four things are applied on top of the raw file contents:

    1. `'l_max_scalars'` is increased by `LMAX_BUFFER`, so that CLASS
       computes beyond the multipole to which the theory is saved.
    2. If both `'H0'` and `'theta_s_100'` are given, one is dropped
       according to `use_H0`.
    3. If `'Neff'` is given instead of `'N_ur'`, the number of
       ultra-relativistic species is derived as
       `N_ur = Neff - (N_ncdm * deg_ncdm) * NUR_PER_MASSIVE_NCDM`,
       where `deg_ncdm` defaults to 1 and may be given either as a
       single number or as one value per ncdm species.
    4. If `'sum_m_ncdm'` is given instead of `'m_ncdm'`, the summed
       neutrino mass is split evenly over the massive species and written
       into `'m_ncdm'` in the form CLASS expects. This exists so that the
       total mass can be varied by the Fisher machinery, which needs a
       single float rather than a comma-separated string.
    5. Any key whose value is `None` is dropped, so that CLASS falls back
       to its own default.

    Give either `'sum_m_ncdm'` (the total mass, which can be varied) or
    `'m_ncdm'` (the CLASS-native per-species form, which cannot), never
    both. ``sum_m_ncdm: 0.06`` with ``N_ncdm: 3`` produces exactly the
    string ``'0.02,0.02,0.02'`` that the old CAMB-to-CLASS translation
    built from ``mnu: 0.06``.
    """
    class_params = {**get_params(param_file=param_file), **cosmo_params}

    # --- maximum multipole ---
    file_lmax = class_params.get('l_max_scalars')
    theory_lmax = int(file_lmax) if file_lmax is not None else int(lmax)
    class_params['l_max_scalars'] = theory_lmax + LMAX_BUFFER

    # --- H0 vs. the acoustic scale: CLASS takes one or the other ---
    has_H0 = class_params.get('H0') is not None
    has_theta = class_params.get('theta_s_100') is not None
    if (not has_H0) and (not has_theta):
        err_msg = f"You must provide a value for either `'H0'` or `'theta_s_100'` to CLASS; neither was found in the `param_file` '{param_file}'."
        raise ValueError(err_msg)
    if has_H0 and has_theta:
        class_params.pop('theta_s_100' if use_H0 else 'H0')
    elif use_H0 and (not has_H0):
        msg = f"You passed `use_H0 = True`, but could not find a value for H0 in the `param_file` '{param_file}'. Passing 'theta_s_100' to CLASS instead of H0."
        warnings.warn(msg)

    # --- number of massive neutrino species ---
    # `deg_ncdm` gives the degeneracy of each ncdm species, so the total number
    #  of massive species is the sum of the degeneracies, not `N_ncdm` itself.
    #  This makes `N_ncdm: 1, deg_ncdm: 3` and `N_ncdm: 3` equivalent. CLASS
    #  accepts `deg_ncdm` as a single number applying to every species, or as
    #  one value per species:
    n_ncdm = class_params.get('N_ncdm')
    n_massive = None
    if n_ncdm is not None:
        deg_ncdm = class_params.get('deg_ncdm', 1)
        if isinstance(deg_ncdm, str):
            degeneracies = [float(d) for d in deg_ncdm.split(',')]
        elif np.isscalar(deg_ncdm):
            degeneracies = [float(deg_ncdm)] * int(n_ncdm)
        else:
            degeneracies = [float(d) for d in deg_ncdm]
        if len(degeneracies) != int(n_ncdm):
            err_msg = f"`'deg_ncdm'` has {len(degeneracies)} entries, but `'N_ncdm'` is {n_ncdm}; give either one degeneracy for all species, or one per species."
            raise ValueError(err_msg)
        n_massive = sum(degeneracies)

    # --- effective number of neutrino species ---
    has_nur = class_params.get('N_ur') is not None
    has_neff = class_params.get('Neff') is not None
    if has_nur and has_neff:
        err_msg = f"Both `'N_ur'` and `'Neff'` were provided in the `param_file` '{param_file}'; you must provide only one."
        raise ValueError(err_msg)
    if has_neff:
        if n_massive is None:
            err_msg = f"`'Neff'` was provided without `'N_ncdm'` in the `param_file` '{param_file}'; both are needed to derive `'N_ur'`."
            raise ValueError(err_msg)
        class_params['N_ur'] = (float(class_params['Neff'])
                                - n_massive * NUR_PER_MASSIVE_NCDM)
        # An unmodified CLASS installation rejects `N_ur < 0`. With the
        # fiducial Neff = 3.044 and three massive species, the derived
        # fiducial `N_ur` is only ~0.0044, so stepping Neff down by more
        # than that for the Fisher derivatives (e.g. the default HD step
        # size) makes `N_ur` negative. Warn the user that a modified CLASS
        # (with the `N_ur >= 0` check relaxed) is required in that case:
        if class_params['N_ur'] < 0:
            msg = (f"The derived `N_ur = {class_params['N_ur']:.6f}` (from "
                   f"`Neff = {class_params['Neff']}` with {n_massive} massive "
                   "neutrino species) is negative. An unmodified CLASS "
                   "installation will reject this: you must use a CLASS "
                   "build with the `N_ur >= 0` check relaxed, or provide a "
                   "larger `Neff` (or fewer massive species).")
            warnings.warn(msg)
    class_params.pop('Neff', None)

    # --- total neutrino mass ---
    # `sum_m_ncdm` is a convenience key, not a CLASS parameter: it holds the SUM
    #  of the neutrino masses. CLASS wants `m_ncdm` as one mass per species,
    #  which is a comma-separated string and therefore cannot be stepped by the
    #  Fisher machinery (it does float arithmetic on the fiducial value). Giving
    #  the summed mass instead means the Fisher varies a single float, exactly
    #  as it did for CAMB's `mnu`. The sum is split evenly over the massive
    #  species and written into `m_ncdm` in the form CLASS expects:
    has_sum_mnu = class_params.get('sum_m_ncdm') is not None
    has_m_ncdm = class_params.get('m_ncdm') is not None
    if has_sum_mnu and has_m_ncdm:
        err_msg = f"Both `'sum_m_ncdm'` and `'m_ncdm'` were provided in the `param_file` '{param_file}'; you must provide only one."
        raise ValueError(err_msg)
    if has_sum_mnu:
        if n_massive is None:
            err_msg = f"`'sum_m_ncdm'` was provided without `'N_ncdm'` in the `param_file` '{param_file}'; both are needed to derive `'m_ncdm'`."
            raise ValueError(err_msg)
        m_each = float(class_params['sum_m_ncdm']) / n_massive
        class_params['m_ncdm'] = ','.join([str(m_each)] * int(n_ncdm))
    class_params.pop('sum_m_ncdm', None)

    # --- the neutrino masses must match the number of massive species ---
    m_ncdm = class_params.get('m_ncdm')
    if (n_ncdm is not None) and (m_ncdm is not None):
        n_masses = len(str(m_ncdm).split(','))
        if n_masses != int(n_ncdm):
            err_msg = f"`'m_ncdm'` lists {n_masses} mass(es), but `'N_ncdm'` is {n_ncdm}; they must match. Write the individual masses in the `param_file`, e.g. `m_ncdm: '0.02, 0.02, 0.02'` with `N_ncdm: 3`."
            raise ValueError(err_msg)

    # --- drop anything left unset, so CLASS uses its own defaults ---
    class_params = {k: v for k, v in class_params.items() if v is not None}
    return class_params



# ----- CMB and BAO theory -----

def get_bao_rs_dv(camb_params, z, camb_results=None):
    """Returns the theoretical BAO quantity r_s/d_V(z) at the given 
    redshifts calculated by CAMB.

    Parameters
    ----------
    camb_params : camb.model.CAMBparams
        The `camb.model.CAMBparams` instance to be used in the calculation.
    z : array_like of float
        An array of redshifts at which to calculate r_s/d_V(z).
    camb_results : camb.results.CAMBdata, default=None
        A `camb.results.CAMBdata` instance that has already been initialized.
        If `None`, it will be initialized within this function.
    
    Returns
    -------
    rs_dv : array_like of float
        The values of r_s/d_V(z) for each redshift in `z`.
    """
    rs_dv = camb_results.get_BAO(z, camb_params)[:,0]
    return rs_dv

def get_class_rs_dv(class_results, zs):
    """Returns r_s/d_V(z) computed from a CLASS instance, matching the
    convention of CAMB's `get_BAO(...)[:, 0]`.

    r_s is the comoving sound horizon at baryon drag (r_drag); d_V is the
    volume-averaged distance d_V(z) = [D_M(z)^2 * c*z/H(z)]^(1/3), with
    D_M = (1+z) * D_A the comoving angular-diameter distance.

    Parameters
    ----------
    class_results : classy.Class
        A `Class` instance on which `.compute()` has already been called.
    zs : array_like of float
        Redshift(s), all > 0, at which to evaluate r_s/d_V(z).

    Returns
    -------
    rs_dv : numpy.ndarray
        r_drag / d_V(z) at each redshift (dimensionless).
    """
    zs = np.atleast_1d(np.asarray(zs, dtype=float))
    rs = class_results.rs_drag()              # Mpc
    d_a = class_results.angular_distance(zs)  # Mpc (physical D_A)
    hubble = class_results.Hubble(zs)         # 1/Mpc, i.e. H(z)/c
    d_m = (1.0 + zs) * d_a                     # Mpc (comoving angular-diameter distance)
    d_v = (d_m**2 * zs / hubble)**(1.0 / 3.0)  # Mpc
    return rs / d_v


def get_spectra(camb_params, class_params, lmax, camb_results=None, raw_cl=True, 
        CMB_unit='muK', cmb_types=['lensed', 'unlensed'], use_class=False):
    """Returns the theoretical lensed and/or unlensed CMB TT, EE, BB, and TE 
    power spectra and the lensing convergence power spectrum computed by CAMB.

    Parameters
    ----------
    camb_params : camb.model.CAMBparams
        The `camb.model.CAMBparams` instance to be used in the calculation.
    lmax : int
        The maximum multipole of the spectra to be returned. This cannot be
        higher than the value used in the CAMBparams instance, which is 
        `camb.model.CAMBparams.max_l`.
    camb_results : camb.results.CAMBdata, default=None
        A `camb.results.CAMBdata` instance that has already been initialized.
        If `None`, it will be initialized within this function.
    raw_cl : bool, default=True
        If `True`, returns only C_ell, instead of ell * (ell + 1) * C_ell / 2pi,
        for the CMB power spectra.
    CMB_unit : str, default='muK'
        The units of the CMB power spectra. Must be a valid `CMB_unit` that 
        may be passed to `camb.results.CAMBdata.get_cmb_power_spectra`.
    cmb_types : list of str, default=['lensed', 'unlensed']
        The kinds of spectra to return.

    Returns
    -------
    theo : nested dict of array_like of float
        A nested dictionary with the `cmb_types` as keys. Each holds another
        dictionary with keys `'tt'`, `'te'`, `'ee'`, and `'bb'` holding the
        CMB power spectra, a key 'kk' holding the lensing power spectrum as 
        C_L^kk = [L(L+1)]^2 C_L^phiphi / 4, and a key `'ells'` holding the 
        multipoles starting from zero.
    """
    if use_class == False:
        if camb_results is None:
            camb_results = camb.get_results(camb_params)
            camb_results.calc_power_spectra()
        theo_keys = {'lensed': 'total', 'unlensed': 'unlensed_total'}
        camb_spectra = ['tt', 'ee', 'bb', 'te']
        theo = {}
        ells = np.arange(lmax + 1)
        powers = camb_results.get_cmb_power_spectra(camb_params, CMB_unit=CMB_unit, raw_cl=raw_cl, lmax=lmax)
        clkk = camb_results.get_lens_potential_cls(lmax=lmax)[:,0] * 2 * np.pi / 4
        for cmb_type in cmb_types:
            theo[cmb_type] = {'ells': ells.copy(), 'kk': clkk.copy()}
            for i, s in enumerate(camb_spectra):
                theo[cmb_type][s] = powers[theo_keys[cmb_type]][:,i].copy()
                theo[cmb_type][s][:2] = 0
        return theo
    elif use_class == True:
        if camb_results is None:
            M = Class()
            M.empty()
            M.set(class_params)
            M.compute()
            camb_results = M

        unl = camb_results.raw_cl(lmax)
        lensed = camb_results.lensed_cl(lmax)

        # 3) prepare output
        theo = {}
        ells = np.arange(lmax + 1)
        camb_spectra = ['tt', 'ee', 'bb', 'te']

        pp = unl['pp']
        clkk = ells**2 * (ells + 1)**2 * pp / 4

        # 4) fill the dict for each requested type
        for cmb_type in cmb_types:
            theo[cmb_type] = {'ells': ells.copy(), 'kk': clkk.copy()}
            data = unl if (cmb_type == 'unlensed' or cmb_type == 'unlensed_total') else lensed
            for spec in camb_spectra:
                theo[cmb_type][spec] = data[spec].copy()
                theo[cmb_type][spec] *= (2.7255e6)**2
                theo[cmb_type][spec][:2] = 0

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
    cmb_types = ['lensed', 'unlensed', 'delensed'] 
    cmb_spectra = ['tt', 'ee', 'bb', 'te'] # order output by CAMB
    theo_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk'] # order when saving to file


    def __init__(self, lmax, output_dir, output_root=None, param_file=None, nlkk=None, 
             recon_lmin=None, recon_lmax=None, use_H0=False, use_class=False, 
             **cosmo_params):
        """Initialization of the theory calculation for a specific set of 
        cosmological, accuracy, and experimental parameters.
        
        Parameters
        ----------
        lmax : int
            The  maximum multipole to be used for the theory spectra.
        output_dir: str
            The full path to the directory where the input CAMB parameters, 
            output theory CMB and lensing power spectra, and the BAO theory 
            are saved.
        output_root : str, default=None
            If not `None`, all files saved in the `output_dir` will begin
            with the `output_root`.
        param_file : str, default=None
            The file name, including the absolute path, of a YAML file that
            contains the cosmological parameter names and values that can be 
            passed to the CAMB function `camb.set_params()`.. If not provided, 
            the default/fiducial values are used (including accuracy settings).
        nlkk : array_like of float, default=None
            The lensing reconstruction noise, used to calculate the delensed
            spectra. This should be the noise on the lensing convergence 
            power spectrum, i.e. N_L^kappakappa = [L(L+1)]^2 N_L^phiphi / 4,
            starting from L = 0.
        recon_lmin, recon_lmax : int, default=None
            The minimum and maximum multipoles to use for delensing, 
            corresponding to those used in the lensing reconstruction. 
            If `nlkk` is not None, `recon_lmin` must be passed. If 
            `recon_lmax` is `None`, `lmax` will be used.
        use_H0 : bool, default=False
            Pass the Hubble constant instead of CosmoMC theta to CAMB, if both are 
            present in the parameter file.
        **cosmo_params : dict of float, optional
            An optional dictionary of parameter names and values to override 
            the values loaded from the YAML file. Note that this won't add any 
            new parameters; it will only update existing parameters loaded 
            from the YAML file.
        use_class : bool, default=False
            If `True`, calculate the CMB and BAO theory with CLASS instead
            of CAMB. NOTE that delensed spectra are not calculated with
            CLASS: `Theory.get_theory` will omit the `'delensed'` key from
            its output, and a warning is issued.

        Notes
        -----
        For the CMB lensing power spectrum, we use the convention
        C_L^kappakappa = C_L^phiphi * [L * (L + 1)]^2 / 4,
        rather than the CAMB convention, 
        [C_L^kappakappa]_CAMB = (2 pi / 4) * C_L^kappakappa.

        All power spectrum arrays begin at a multipole ell = 0. The CMB power 
        spectra are in C_ell's (i.e., no factor of ell * (ell + 1) / (2 * pi)
        applied), in units of uK^2.
        """
         # the parameter file's `lmax` (CAMB) or `l_max_scalars` (CLASS) sets
        # the multipole to which the theory is saved, and takes precedence
        # over the `lmax` argument; the Boltzmann code computes to
        # `self.lmax + LMAX_BUFFER`
        self.lmax = get_theory_lmax(lmax, param_file=param_file,
                                    use_class=use_class, **cosmo_params)
        self.ells = np.arange(self.lmax+1) # CAMB starts at lmin = 0
        # filenames to save/load theory
        self.output_dir = output_dir
        self.theo_fnames = config.camb_theo_fnames(output_dir, theo_root=output_root)

        # set up empty dict to hold theory 
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
        # set up the Boltzmann-code parameters
        self.results = None # only calculate the results if necessary
        self.use_class = use_class
        self._setup_boltzmann_params(param_file=param_file, use_H0=use_H0,
                                     **cosmo_params)



    def _setup_boltzmann_params(self, param_file=None, use_H0=False, **cosmo_params):
        """Set up `self.camb_params` (and `self.class_params` when
        `use_class=True`) from the parameter file and overrides.

        This is the extension point for the Boltzmann-code parameter setup:
        subclasses (e.g. hdInitPk's `Theory`) whose parameter files contain
        keys that `camb.set_params` does not accept (e.g. binned-Pk bin
        amplitudes, or kSZ template parameters) can override this single
        method, without re-implementing the rest of `Theory.__init__`.
        `self.lmax` and `self.use_class` are set before this is called.
        """
        if self.use_class:
            # CAMB is not used anywhere on the CLASS path
            self.camb_params = None
            self.class_params = set_class_params(
                self.lmax, param_file=param_file, use_H0=use_H0,
                **cosmo_params)
        else:
            self.class_params = None
            self.camb_params = set_camb_params(self.lmax, param_file=param_file, use_H0=use_H0, **cosmo_params)


    def get_camb_results(self, save=False):
        """Get the `CAMBdata` instance from the `CAMBparams` (stored in
        `Theory.camb_params`), and store it in `Theory.results`.
        
        Parameters
        ----------
        save : bool, default=False
            If `True`, save the values set in the `CAMBparams` instance used 
            to calculate the theory. The file will be saved in the `output_dir`
            passed when initializing the `Theory` class.

        Returns
        -------
        Theory.results : camb.results.CAMBdata
            The `CAMBdata` instance calculated from the input `CAMBparams`.
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
        """Compute the CLASS results from the parameters stored in
        `Theory.class_params`, and store them in `Theory.results`.

        Parameters
        ----------
        save : bool, default=False
            If `True`, save the parameters that were passed to CLASS. The
            file is written to the `output_dir` given when initializing the
            `Theory` class.

        Returns
        -------
        Theory.results : classy.Class
            The `Class` instance, after `.compute()` has been called on it.

        Note
        ----
        Delensed spectra are not calculated with CLASS; see
        `Theory.get_theory`.
        """
        if self.results is None:
            M = Class()
            M.empty()
            M.set(self.class_params)
            M.compute()
            self.results = M
        if save:
            with open(self.theo_fnames['params'], 'w') as f:
                f.write(str(self.class_params))
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
            if self.use_class:
                if self.results is None:
                    self.get_class_results(save=save)
                rs_dv = get_class_rs_dv(self.results, zs)
            else:
                if self.results is None:
                    self.get_camb_results(save=save)
                rs_dv = self.results.get_BAO(zs, self.camb_params)[:, 0]
            if save:
                header = 'z, r_s/d_V(z)'
                np.savetxt(fname, np.column_stack([zs, rs_dv]), header=header)
        return rs_dv


    def _compute_spectra(self, cmb_type, save=False):
        """Compute the theory spectra for a single `cmb_type` (`'lensed'` or
        `'unlensed'`) from the Boltzmann code, ignoring any saved or cached
        theory.

        This is the extension point for the (lensed/unlensed) spectra
        calculation: subclasses (e.g. hdInitPk's `BinnedPkTheory`) can
        override this single method to change how the spectra are computed,
        without re-implementing the caching, loading, and saving logic in
        `Theory.get_theory_spectra`.

        Parameters
        ----------
        cmb_type : str
            `'lensed'` or `'unlensed'`.
        save : bool, default=False
            Passed through to `Theory.get_camb_results` or
            `Theory.get_class_results` if the Boltzmann results have not been
            computed yet.

        Returns
        -------
        theo : dict of array_like of float
            A dict with keys `'ells'`, `'tt'`, `'te'`, `'ee'`, `'bb'`, and
            `'kk'` holding the theory spectra (C_ell's) in units of uK^2,
            starting at ell = 0.
        """
        if self.results is None:
            if self.use_class == False:
                self.get_camb_results(save=save)
            elif self.use_class == True:
                self.get_class_results(save=save)
        return get_spectra(self.camb_params, self.class_params, self.lmax,
                           camb_results=self.results, raw_cl=True, CMB_unit='muK',
                           cmb_types=[cmb_type], use_class=self.use_class)[cmb_type]


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
        for cmb_type in self.cmb_types[:2]: # loop through lensed, unlensed
            fname = self.theo_fnames[cmb_type]
            # if not overwriting, check if we already have all the spectra, or if its saved
            if all([s in self.theo[cmb_type] for s in self.cmb_spectra + ['kk']]) and (not overwrite):
                theo[cmb_type] = self.theo[cmb_type].copy()
            elif os.path.exists(fname) and (not overwrite): # load it
                print(f"loading {cmb_type} theory from {fname}")
                theo[cmb_type] = utils.load_from_file(fname, self.theo_cols)
            else: # get it from the Boltzmann code
                theo[cmb_type] = self._compute_spectra(cmb_type, save=save)
            # save it
            if save:
                print(f"saving {cmb_type} theory to {fname}") 
                utils.save_to_file(fname, theo[cmb_type], keys=self.theo_cols)
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
        


    def _compute_delensed_spectra(self, save=False):
        """Compute the delensed theory spectra from the Boltzmann code,
        ignoring any saved or cached theory. `Theory.check_delensing_vars` is
        assumed to have been called already.

        This is the extension point for the delensed-spectra calculation:
        subclasses (e.g. hdInitPk's `BinnedPkTheory`) can override this
        single method without re-implementing the caching, loading, and
        saving logic in `Theory.get_delensed_spectra`.

        Parameters
        ----------
        save : bool, default=False
            Passed through to `Theory.get_camb_results` if the CAMB results
            have not been computed yet.

        Returns
        -------
        delensed_theo : dict of array_like of float
            A dict with keys `'ells'`, `'tt'`, `'te'`, `'ee'`, `'bb'`, and
            `'kk'` holding the delensed theory spectra (C_ell's) in units of
            uK^2, starting at ell = 0.
        """
        if self.results is None:
            self.get_camb_results(save=save)
        return get_delensed_spectra(
            self.camb_params, self.lmax, self.nlkk, self.Lmin,
            Lmax=self.Lmax, camb_results=self.results, raw_cl=True,
            CMB_unit='muK')


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
        if all([s in self.theo['delensed'] for s in self.cmb_spectra + ['kk']]) and (not overwrite):
            delensed_theo = self.theo['delensed'].copy()
        elif os.path.exists(fname) and (not overwrite):
            print(f"loading delensed theory from {fname}")
            delensed_theo = utils.load_from_file(fname, self.theo_cols)
        else:
            self.check_delensing_vars()
            delensed_theo = self._compute_delensed_spectra(save=save)
        if save:
            print(f"saving delensed theory to {fname}")
            utils.save_to_file(fname, delensed_theo, keys=self.theo_cols)
        self.theo['delensed'] = delensed_theo.copy()
        return delensed_theo
   

    def get_theory(self, cmb_types=None, save=False, overwrite=False):
        """Returns the lensed, unlensed, and/or delensed CMB and lensing 
        potential theory spectra.

        Parameters
        ----------
        cmb_types : str or list of str, default=None
            The type of CMB spectra to return. Must be 'lensed', 'unlensed', 
            and/or 'delensed'. Returns all three by default.
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
        theo : nested dict of array_like of float
            A dictionary with key(s) given by the `cmb_types`. Each holds
            another dict containing one-dimensional arrays for the spectra, with
            keys 'ells' for the multipoles, 'tt', 'ee', 'te', 'bb' for the CMB
            spectra (C_ell's in units of uK^2), and 'kk' for the lensing potential
            spectrum (C_L^kappakappa = C_L^phiphi * [L * (L + 1)]^2 / 4). 
            Everything begins at ell = 0.

        Notes
        -----
        If `use_class=True`, delensed spectra are not calculated. The
        returned dictionary will not contain a `'delensed'` key even when
        one was requested in `cmb_types`, and a warning is issued.

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
                err_msg = f"You passed `cmb_types = {cmb_types}`, but `{cmb_type}` is not a valid option; must be one of {self.cmb_types}."
                raise ValueError(err_msg)
        # get theory for each cmb_type
        theo = {}
        for cmb_type in cmb_types:
            if len(list(self.theo[cmb_type].keys())) < len(self.theo_cols):
                if 'delens' in cmb_type.lower():
                    if self.use_class:
                        msg = "Delensed spectra are not calculated with CLASS: the 'delensed' spectra will be missing from the returned dictionary. Set `use_class=False` to calculate them with CAMB."
                        warnings.warn(msg)
                        continue
                    theo_spectra = self.get_delensed_spectra(save=save, overwrite=overwrite)
                else:
                    theo_spectra = self.get_theory_spectra(save=save, overwrite=overwrite)[cmb_type]
                theo[cmb_type] = theo_spectra.copy()
            else:
                theo[cmb_type] = self.theo[cmb_type].copy()
        return theo
