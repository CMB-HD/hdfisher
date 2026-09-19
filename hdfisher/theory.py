"""Calculate CMB and BAO theory to make a Fisher matrix"""
import os
import warnings
from copy import deepcopy
import numpy as np
import yaml
import camb
from . import mpi, utils, config


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
        param_file = config.fiducial_param_file()
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
    """
    input_params = set_cosmo_params(param_file=param_file, use_H0=use_H0, **cosmo_params).copy()
    input_params['lmax'] = int(lmax+500)
    pars = camb.set_params(**input_params)
    return pars.copy()




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


def get_spectra(camb_params, lmax, camb_results=None, raw_cl=True, 
        CMB_unit='muK', cmb_types=['lensed', 'unlensed']):
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


    def __init__(self, lmax, output_dir, output_root=None, param_file=None, nlkk=None, recon_lmin=None, recon_lmax=None, use_H0=False, **cosmo_params):
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
        self.lmax = int(lmax)
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

        # set up the camb params 
        self.camb_params = set_camb_params(self.lmax, param_file=param_file, use_H0=use_H0, **cosmo_params)
        self.results = None # only calculate the `CAMBdata` if necessary


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
                self.get_camb_results(save=save)
            rs_dv = self.results.get_BAO(zs, self.camb_params)[:,0]
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
        for cmb_type in self.cmb_types[:2]: # loop through lensed, unlensed
            fname = self.theo_fnames[cmb_type]
            # if not overwriting, check if we already have all the spectra, or if its saved
            if all([s in self.theo[cmb_type] for s in self.cmb_spectra + ['kk']]) and (not overwrite):
                theo[cmb_type] = self.theo[cmb_type].copy()
            elif os.path.exists(fname) and (not overwrite): # load it
                print(f"loading {cmb_type} theory from {fname}")
                theo[cmb_type] = utils.load_from_file(fname, self.theo_cols)
            else: # get it from camb
                if self.results is None:
                    self.get_camb_results(save=save)
                theo[cmb_type] = get_spectra(self.camb_params, self.lmax, camb_results=self.results, raw_cl=True, CMB_unit='muK', cmb_types=[cmb_type])[cmb_type] 
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
            delensed_theo = utils.load_from_file(fname, self.theo_cols)
        else: # calculate it
            self.check_delensing_vars()
            if self.results is None:
                self.get_camb_results(save=save)
            delensed_theo = get_delensed_spectra(self.camb_params, self.lmax, self.nlkk, self.Lmin, Lmax=self.Lmax, camb_results=self.results, raw_cl=True, CMB_unit='muK')
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
                    theo_spectra = self.get_delensed_spectra(save=save, overwrite=overwrite)
                else:
                    theo_spectra = self.get_theory_spectra(save=save, overwrite=overwrite)[cmb_type]
                theo[cmb_type] = theo_spectra.copy()
            else:
                theo[cmb_type] = self.theo[cmb_type].copy()
        return theo
