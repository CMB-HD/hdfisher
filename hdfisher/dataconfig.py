import os
import warnings
import numpy as np
from hd_mock_data import hd_data
from . import utils, theory, mpi, config


# if using MPI, only issue a warning from the main MPI process:
if mpi.rank != 0:
    warnings.simplefilter('ignore', category=UserWarning)


# mock DESI BAO data:

def desi_theory_fname():
    """Returns the path to the file containing the theoretical BAO
    measurement r_s/d_V(z) for mock DESI BAO. The first column of the
    file contains the redshift z, and the second contains the quantity
    r_s/d_V evaluated at that redshift.
    """
    fname = os.path.join(config.data_path('bao'),
                         'mock_desi_bao_rs_over_DV_fid_data.txt')
    return fname


def load_desi_theory():
    """Returns one-dimensional arrays containing the redshifts (`z`)
    and the theoretical BAO measurement r_s/d_V(z) at those redshifts
    (`rs_dv`) for mock DESI BAO.
    """
    z, rs_dv = np.loadtxt(desi_theory_fname(), unpack=True)
    return z, rs_dv


def desi_redshifts():
    """Returns a one-dimensional array holding the redshifts at which the
    mock DESI BAO theory and covariance matrix were calculated.
    """
    z, _ = load_desi_theory()
    return z


def desi_covmat_fname():
    """Returns the path to the covariance matrix for the mock DESI BAO
    measurements r_s/d_V(z)."""
    fname = os.path.join(config.data_path('bao'),
                         'mock_desi_bao_rs_over_DV_fid_cov.txt')
    return fname


def load_desi_covmat():
    """Returns a two-dimensional array holding the covariance matrix
    for the mock DESI BAO measurements, r_s/d_V(z).
    """
    covmat = np.loadtxt(desi_covmat_fname())
    return covmat


def precomputed_desi_fisher_fname(use_H0=False):
    """Returns the path to the file holding a Fisher matrix calculated
    from the mock DESI BAO measurements and covariance matrix. The
    parameters in the Fisher matrix are the six LCDM parameters, the
    effective number of relativistic species, and the sum of the neutrino
    masses.

    Parameters
    ----------
    use_H0: bool, default=False
        If `True`, the Hubble constant is used as one of the six LCDM
        parameters. If `False`, the cosmoMC approximation to the angular
        scale of the sound horizon at last scattering (multiplied by 100)
        is used instead.

    Returns
    -------
    fname : str
        The requested file name.
    """
    H0_info = '_useH0' if use_H0 else ''
    fname = os.path.join(config.data_path('fisher_matrices'),
                         f'desi_bao{H0_info}_fisher.txt')
    return fname


def load_precomputed_desi_fisher(use_H0=False):
    """Returns a Fisher matrix calculated from the mock DESI BAO
    measurements and covariance matrix. The parameters in the Fisher
    matrix are the six LCDM parameters, the effective number of
    relativistic species, and the sum of the neutrino masses.

    Parameters
    ----------
    use_H0: bool, default=False
        If `True`, the Hubble constant is used as one of the six LCDM
        parameters. If `False`, the cosmoMC approximation to the angular
        scale of the sound horizon at last scattering (multiplied by 100)
        is used instead.

    Returns
    -------
    fisher_matrix : array_like of float
        The eight-parameter Fisher matrix for the mock DESI BAO data.
    fisher_params : list of str
        A list of parameter names for the parameters in the Fisher
        matrix, in the same order as their corresponding rows/columns.
    """
    fname = precomputed_desi_fisher_fname(use_H0=use_H0)
    fisher_matrix, fisher_params = utils.load_fisher_matrix(fname)
    return fisher_matrix, fisher_params


class Data:
    """Holds the experimental configuration information for CMB-HD, along 
    with for the experiments considered in MacInnis et. al. (2023), and 
    defines methods to access to the associated files provided with 
    `hdfisher`.
    """
    cmb_exps = ['so', 's4', 'hd']
    # CMB and lensing theory spectra:
    cmb_types = ['lensed', 'delensed', 'unlensed']
    cmb_spectra = ['tt', 'te', 'ee', 'bb']
    
    def __init__(self, hd_data_version='latest'):
        """Initialization of the experimental configurations.
        
        Parameters
        ----------
        hd_data_version : str, default='latest'
            The CMB-HD data version to use. This determines which CMB-HD
            covariance matrix, noise spectra, etc. is used. By default, 
            the latest version is used. To reproduce the results in 
            MacInnis et. al. (2023), use `hd_data_version='v1.0'`. 
            See the `hdMockData` repository for a list of versions.
        """
        # initialize the `HDMockData` class to access CMB-HD data:
        self.hd_datalib = hd_data.HDMockData(version=hd_data_version.lower())
        self.hd_data_version = self.hd_datalib.version
        
        # multipole ranges:
        self.lmins = {'hd': self.hd_datalib.lmin, 'so': 30, 's4': 30}
        self.lmaxs = {'hd': self.hd_datalib.lmax, 'so': 5000, 's4': 5000}
        self.lmaxsTT = {'hd': self.hd_datalib.lmax, 'so': 3000, 's4': 3000}
        self.Lmaxs = {'hd': self.hd_datalib.Lmax, 'so': 3000, 's4': 3000}
        self.theo_lmaxs = {'hd': self.hd_datalib.theo_lmax, 'so': 5000, 's4': 5000} 
        self.ell_ranges = {}
        for exp in self.cmb_exps:
            self.ell_ranges[exp] = {'tt': [self.lmins[exp], self.lmaxsTT[exp]],
                                    'te': [self.lmins[exp], self.lmaxs[exp]],
                                    'ee': [self.lmins[exp], self.lmaxs[exp]],
                                    'bb': [self.lmins[exp], self.lmaxs[exp]],
                                    'kk': [self.lmins[exp], self.Lmaxs[exp]]}

        # available covariance matrices:
        self.cov_cmb_types = {'hd': ['lensed', 'delensed'], 's4': ['delensed']}
        if self.hd_data_version in ['v1.0', 'v1.1']:
            self.cov_cmb_types['so'] = ['delensed']
        else:
            self.cov_cmb_types['so'] = ['lensed', 'delensed']
        
        # CMB-HD has some additional files calculated for a lower maximum multipole
        # (NOTE: these were only calculated for v1.0 of the HD data)
        if self.hd_data_version == 'v1.0':
            self.hd_lmaxs = [1000, 3000, 5000, 10000]
        else:
            self.hd_lmaxs = []


    # ----- functions to check arguments passed to other functions: -----

    def _check_cmb_exp(self, exp, valid_exps=None):
        """Checks whether the value passed for `exp` is valid."""
        exp = exp.lower()
        valid_exps = self.cmb_exps if (valid_exps is None) else valid_exps
        if exp not in valid_exps:
            err_msg = (f"Invalid experiment name. You passed `{exp = }`,"
                       f" but `exp` must be one of {valid_exps}.")
            raise ValueError(err_msg)
        else:
            return exp


    def _check_hd_lmax(self, hd_lmax, cmb_type='delensed', feedback=False,
                       include_fg=True, data_info=None):
        """Checks whether the value passed for `hd_lmax` is valid. If it
        is not, the `data_info` string is used to give a more descriptive
        error message.
        """
        data_info = 'data' if (data_info is None) else data_info
        hd_data_info = f'CMB-HD {self.hd_data_version} {data_info}'
        hd_lmax = self.lmaxs['hd'] if (hd_lmax is None) else int(hd_lmax)
        valid_hd_lmaxs =  [*self.hd_lmaxs, self.lmaxs['hd']]
        if hd_lmax not in valid_hd_lmaxs:
            err_msg = (f"Invalid `{hd_lmax = }` for the {hd_data_info}."
                       f" The valid options are: {valid_hd_lmaxs}.")
            if self.hd_data_version != 'v1.0':
                version_msg = ("If you are trying to reproduce the results "
                               "of MacInnis et. al. (2023), you must pass "
                               "`hd_data_version='v1.0'` during initialization.")
                err_msg = f'{err_msg} {version_msg}'
            raise ValueError(err_msg)
        if hd_lmax < self.lmaxs['hd']:
            if feedback:
                errmsg = (f"There is no {hd_data_info} saved with baryonic "
                          f"feedback calculated out to `{hd_lmax = }`. You "
                          "must either pass `feedback=False` or "
                          f"`hd_lmax={self.lmaxs['hd']}`.")
                raise ValueError(errmsg)
            if not include_fg:
                err_msg = (f"There is no {hd_data_info} saved with "
                           f" `{include_fg = }` and `{hd_lmax = }`. "
                           "You must pass `include_fg=True` or "
                           f"`hd_lmax={self.lmaxs['hd']}`.")
                raise ValueError(err_msg)
            if cmb_type.lower() != 'delensed':
                err_msg = (f"There is no {hd_data_info} saved with "
                           f" `{cmb_type = }` and `{hd_lmax = }`. "
                           "You must pass `cmb_type='delensed'` or "
                           f"`hd_lmax={self.lmaxs['hd']}`.")
                raise ValueError(err_msg)
        return hd_lmax


    def _check_hd_fgs(self, include_fg=True, cmb_type='lensed', data_info=None):
        """Checks whether a given CMB-HD mock data product has been
        calculated without including residual extragalactic foregrounds.
        """
        data_info = 'data' if (data_info is None) else data_info
        hd_data_info = f'CMB-HD {self.hd_data_version} {data_info}'
        if not include_fg:
            if self.hd_data_version != 'v1.0':
                err_msg = f"There is no {hd_data_info} with `{include_fg=}`."
                version_msg = ("If you are trying to reproduce the results "
                               "of MacInnis et. al. (2023), you must pass "
                               "`hd_data_version='v1.0'` during initialization.")
                raise ValueError(f'{err_msg} {version_msg}')
            elif cmb_type.lower() != 'lensed':
                raise ValueError(f"There is no {cmb_type} {hd_data_info} with `{include_fg=}`.")


    # ----- functions that resturn file names of data: -----

    def fiducial_param_file(self, feedback=False):
        """Absolute path to the YAML file holding the fiduical parameters
        (including cosmological and accuracy parameters) passed to CAMB
        when calculating the CMB and BAO theory.

        Parameters
        ----------
        feedback : bool, default=False
            If `True`, the parameter file sets the CAMB `halofit_version`
            to `mead2020_feedback`, i.e. uses the HMCode 2020 + baryonic
            feedback non-linear model. Otherwise, the HMCode 2016
            CDM-only model is used by setting `halofit_version` to
            `mead2016`.

        Returns
        -------
        fname : str
            The absolute path to the file.

        See Also
        --------
        config.fiducial_param_file
        """
        fname = config.fiducial_param_file(feedback=feedback,
                                           hd_data_version=self.hd_data_version)
        return fname


    def fiducial_fisher_steps_file(self, feedback=False):
        """Absolute path to the YAML file holding the fiduical parameter
        step sizes used to calculate the Fisher matrices.

        Parameters
        ----------
        feedback : bool, default=False
            If `True`, the file includes a step size for the HMCode 2020
            baryonic feedback parameter, `HMCode_logT_AGN`. Otherwise
            this parameter is excluded.

        Returns
        -------
        fname : str
            The absolute path to the file.

        See Also
        --------
        config.fiducial_fisher_steps_file
        """
        fname = config.fiducial_fisher_steps_file(feedback=feedback)
        return fname


    def example_hd_fisher_fname(self, cmb_type='delensed',
                                use_H0=False, with_desi=False):
        """Path to an example CMB-HD Fisher matrix that was calculated
        with the correct `hd_data_version`.

        All Fisher matrices contain 8 parameters (LCDM + N_eff + sum m_nu)
        and all have a Gaussian prior on the optical depth applied. For
        versions 1.0 and 1.1 of the CMB-HD mock data, the prior is
        sigma(tau) = 0.007; for version 1.2, it is sigma(tau) = 0.005.

        Parameters
        ----------
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the Fisher matrix was calculated
            with delensed CMB TT, TE, EE, and BB power spectra, in
            addition to the CMB lensing convergence power spectrum. If
            `cmb_type='lensed'`, the Fisher matrix was computed with
            lensed CMB spectra instead of delensed.
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the
            angular scale of the sound horizon at last scattering
            (multiplied by 100) is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB
            spectra. If `True`, the Fisher matrix is the sum of a CMB and
            a mock DESI BAO Fisher matrix.

        Returns
        -------
        fname : str
            The path to the file.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed.

        See Also
        --------
        load_example_hd_fisher
        """
        cmb_type = cmb_type.lower()
        if cmb_type not in self.cov_cmb_types['hd']:
            raise ValueError(f"Invalid `{cmb_type = }`. The options are:"
                             f" {self.cov_cmb_types['hd']}")
        lmin = self.lmins['hd']
        lmax = self.lmaxs['hd']
        lmaxTT = self.lmaxsTT['hd']
        Lmax = self.Lmaxs['hd']
        ell_info = f'lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}'
        H0_info = '_useH0' if use_H0 else ''
        desi_info = '_desi_bao' if with_desi else ''
        fname_root = f'hd_fsky0pt6_{ell_info}_{cmb_type}{desi_info}{H0_info}'
        version = self.hd_data_version
        fisher_dir = os.path.join(config.data_path(f'fisher_matrices'), 'hd_examples')
        fname = os.path.join(fisher_dir, f'{fname_root}_fisher_{version}.txt')
        return fname


    def hd_theory_fname(self, spectrum_type, feedback=False, hd_lmax=None, **kwargs):
        """Path to the file containing the CMB-HD theory CMB and lensing
        spectra for a given CMB type (e.g., delensed).

        Parameters
        ----------
        spectrum_type : str
            The name of the kind of spectra. Must be either `'lensed'`,
            `'delensed'`, or `'unlensed'` for files containing the CMB TT,
            TE, EE, and BB spectra along with the lensing (kappa kappa)
            spectrum. For version `'v1.0'` of the CMB-HD mock data, you
            may also pass `'clkk_res'` for files containing only the
            residual lensing power.
        feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            theory calculated with the HMCode2020 + baryonic feedback
            non-linear model, as opposed to the HMCode2016 CDM-only
            model.

        Returns
        -------
        fname : str
            The absolute path and name of the requested file.

        Other Parameters
        ----------------
        hd_lmax : int, default=None
            Used for CMB-HD spectra that were calculated with a lower
            maximum multipole than the baseline case. Only available for
            `'v1.0'` of the CMB-HD mock data.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `cmb_theory_fname`
            method of `hd_mock_data.hd_data.HDMockData`. Ignored for v1.0
            CMB-HD mock data if `spectrum_type='clkk_res'` or `hd_lmax`
            is lower than the default value.

        Raises
        ------
        ValueError
            If the requested file does not exist or is incompatible with
            the `hd_data_version` attribute.

        Notes
        -----
        If `cmb_type = 'clkk_res'`, the file will contain a single column
        holding the residual CMB lensing power spectrum. Otherwise, the
        file will have a column for the multipoles of the spectra, the
        CMB TT, TE, EE, and BB power spectra (in units of uK^2, without
        any multiplicative factors applied), and the CMB lensing power
        spectrum C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where L is the
        lensing multipole and C_L^phiphi is the CMB lensing potential
        power spectrum.
        """
        # make sure the `hd_lmax` is valid:
        if spectrum_type in self.cmb_types:
            hd_lmax = self._check_hd_lmax(hd_lmax, feedback=feedback,
                                          data_info=f'{spectrum_type} theory')
        else:
            hd_lmax = self._check_hd_lmax(hd_lmax, data_info='residual lensing power')
        # for CMB theory spectra, if `hd_lmax` is the default value,
        # get the file for this HD mock data version; otherwise, get the
        # file provided with `hdfisher`:
        if (spectrum_type in self.cmb_types) and (hd_lmax >= self.lmaxs['hd']):
            kwargs = {**kwargs, 'baryonic_feedback': feedback}
            fname = self.hd_datalib.cmb_theory_fname(spectrum_type, **kwargs)
        else:
            feedback_info = '_hmcode2020_feedback' if feedback else ''
            lmin = self.lmins['hd']
            ell_info = f'lmin{lmin}lmax{hd_lmax}Lmax{hd_lmax}'
            if spectrum_type in self.cmb_types:
                spec_info = f'{spectrum_type}_cls'
            else:
                spec_info = spectrum_type
            theo_dir = config.data_path('theory')
            fname = os.path.join(theo_dir, f'hd{feedback_info}_{ell_info}_{spec_info}.txt')
        return fname


    def cmb_theory_fname(self, exp, spectrum_type, feedback=False,
                         hd_lmax=None, **kwargs):
        """Path to the file containing the theory CMB and CMB lensing
        power spectra for a given CMB experiment and CMB type (e.g.,
        delensed).

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        spectrum_type : str
            The name of the kind of spectra. Must be either `'lensed'`,
            `'delensed'`, or `'unlensed'` for files containing the CMB TT,
            TE, EE, and BB spectra along with the lensing (kappa kappa)
            spectrum. For version `'v1.0'` of the CMB-HD mock data, you
            may also pass `'clkk_res'` for files containing only the
            residual lensing power.
        feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            theory calculated with the HMCode2020 + baryonic feedback
            non-linear model, as opposed to the HMCode2016 CDM-only
            model. Ignored if `exp` is not `'HD'`.

        Returns
        -------
        fname : str
            The path to the requested file.

        Other Parameters
        ----------------
        hd_lmax : int, default=None
            Used for CMB-HD power spectra that were calculated with a
            lower maximum multipole than the baseline case. Only
            available for `'v1.0'` of the CMB-HD mock data. Ignored if
            `exp` is not `'HD'`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `cmb_theory_fname`
            method of `hd_mock_data.hd_data.HDMockData`. Ignored if `exp`
            is not `'HD'`; also ignored for v1.0 CMB-HD mock data if
            `spectrum_type='clkk_res'` or `hd_lmax` is lower than the
            default value.

        Raises
        ------
        ValueError
            If the requested file is not available.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored.

        Notes
        -----
        If `cmb_type = 'clkk_res'`, the file will contain a single column
        holding the residual CMB lensing power spectrum. Otherwise, the
        file will have a column for the multipoles of the spectra, the
        CMB TT, TE, EE, and BB power spectra (in units of uK^2, without
        any multiplicative factors applied), and the CMB lensing power
        spectrum C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where L is the
        lensing multipole and C_L^phiphi is the CMB lensing potential
        power spectrum.
        """
        # check the input
        exp = self._check_cmb_exp(exp)
        spectrum_type = spectrum_type.lower()
        valid_spec_types = [*self.cmb_types, 'clkk_res']
        if spectrum_type not in valid_spec_types:
            raise ValueError(f"Invalid spectrum type: `{spectrum_type = }`. "
                             f"The options are: {valid_spec_types}.")
        if exp == 'hd':
            fname = self.hd_theory_fname(spectrum_type, feedback=feedback,
                                          hd_lmax=hd_lmax, **kwargs)
        else:
            if (hd_lmax is not None) or feedback:
                warnings.warn("Ignoring the `hd_lmax` and `feedback`"
                              f" arguments for `exp = '{exp}'`.")
            lmin = self.lmins[exp]
            lmax = self.lmaxs[exp]
            Lmax = self.Lmaxs[exp]
            if spectrum_type in self.cmb_types:
                spec_info = f'{spectrum_type}_cls'
            else:
                spec_info = spectrum_type
            fname_root = f'{exp}_lmin{lmin}lmax{lmax}Lmax{Lmax}_{spec_info}'
            theo_dir = config.data_path('theory')
            fname = os.path.join(theo_dir, f'{fname_root}.txt')
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname


    def cmb_noise_fname(self, exp, include_fg=True):
        """Path to the file containing the power spectra of the noise on
        the CMB TT, TE, EE, and BB spectra, coadded from 90 and 150 GHz.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, the temperature noise power spectrum at each
            frequency is the sum of the instrumental noise and the
            residual extragalactic foreground power spectrum. If `False`,
            it will only contain instrumental noise. Ignored if `exp` is
            not `'HD'`.

        Returns
        -------
        fname : str
            The path to the file.

        Raises
        ------
        ValueError
            If the `exp` is invalid.

        Warns
        -----
        If the value of `include_fg` will be ignored, or if the file does
        not exist.

        Notes
        -----
        The returned file will have a column for the multipoles and
        columns for the CMB TT, TE, EE, and BB noise power spectra (in
        units of uK^2, without any multiplicative factors applied).
        """
        # check the input
        exp = self._check_cmb_exp(exp)
        if exp == 'hd': # for HD, get the correct version:
            fname = self.hd_datalib.cmb_noise_fname(include_fg=include_fg)
        else: # otherwise, there is only one version:
            fname_root = f'{exp}_coaddf090f150_cmb_noise_cls_lmax5000'
            noise_dir = config.data_path('noise')
            fname = os.path.join(noise_dir, f'{fname_root}.txt')
            if not include_fg:
                msg = f"Ignoring the `include_fg` argument for `exp = '{exp}'`."
                warnings.warn(msg)
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname


    def hd_lensing_noise_fname(self, include_fg=True, hd_Lmax=None, **kwargs):
        """Path to the file containing the CMB-HD lensing noise.

        Parameters
        ----------
        include_fg : bool, default=True
            If `True`, return the file name for lensing noise that was
            calculated including the effects of residual extragalactic
            foregrounds in temperature. If `False`, return the file name
            for lensing noise that was calculated by neglecting these
            effects; only available for v1.0 CMB-HD mock data and
            `hd_Lmax=None` or `hd_Lmax=20100`.
        hd_Lmax : int or None, default=None
            Used for CMB-HD lensing noise spectra that were calculated
            with a lower maximum (lensing and CMB) multipole than the
            baseline case. Only available for `'v1.0'` of the CMB-HD mock
            data when `include_fg=True`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the
            `lensing_noise_fname` method of
            `hd_mock_data.hd_data.HDMockData`. Ignored for v1.0 CMB-HD
            mock data if `include_fg=False` or `hd_Lmax` is lower than
            the default value.

        Returns
        -------
        fname : str
            The path to the file.

        Raises
        ------
        ValueError
            If the lensing noise has not been calculated for the given
            values of `hd_Lmax` and/or `include_fg`.

        Notes
        -----
        The returned file contains two columns: L, N_L^kk, where L is the
        CMB lensing multipole and N_L^kk is the noise on the CMB lensing
        power spectrum, C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where
        C_L^phiphi is the CMB lensing potential power spectrum.
        """
        # check if the file is available:
        self._check_hd_fgs(include_fg=include_fg, data_info='lensing noise')
        hd_Lmax = self._check_hd_lmax(hd_Lmax, include_fg=include_fg,
                                      data_info='lensing noise')
        # if `include_fg=True` and `hd_Lmax` is the default value, get
        # the file for this HD mock data version; otherwise, get the file
        # provided with `hdfisher`:
        if include_fg and (hd_Lmax >= self.Lmaxs['hd']):
            # then we can load the latest version
            fname = self.hd_datalib.lensing_noise_fname(**kwargs)
        else:
            extra_info = '' if include_fg else '_nofg'
            lmin = self.lmins['hd']
            fname_root = f'hd{extra_info}_lmin{lmin}lmax{hd_Lmax}Lmax{hd_Lmax}_nlkk'
            fname = os.path.join(config.data_path('noise'), f'{fname_root}.txt')
        return fname


    def cmb_lensing_noise_fname(self, exp, include_fg=True, hd_Lmax=None, **kwargs):
        """Path to the file containing the CMB lensing noise power
        spectrum for the given CMB experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'HD'`,
            `'SO'`, or `'S4'`. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, return the file name for lensing noise that was
            calculated including the effects of residual extragalactic
            foregrounds in temperature. If `False`, return the file name
            for lensing noise that was calculated by neglecting these
            effects; only available for v1.0 CMB-HD mock data and
            `hd_Lmax=None` or `hd_Lmax=20100`. Ignored if `exp` is not
            `'HD'`.
        hd_Lmax : int or None, default=None
            Used for CMB-HD lensing noise spectra that were calculated
            with a lower maximum (lensing and CMB) multipole than the
            baseline case. Only available for `'v1.0'` of the CMB-HD mock
            data when `include_fg=True`. Ignored if `exp` is not `'HD'`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the
            `lensing_noise_fname` method of
            `hd_mock_data.hd_data.HDMockData`. Ignored if `exp` is not
            `'HD'`, or for v1.0 CMB-HD mock data if `include_fg=False`
            or `hd_Lmax` is lower than the default value.

        Returns
        -------
        fname : str
            The path to the file.

        Warns
        -----
        If the values of `include_fg` or `hd_Lmax` were provided, but
        will be ignored.

        Notes
        -----
        The returned file contains two columns: L, N_L^kk, where L is the
        CMB lensing multipole and N_L^kk is the noise on the CMB lensing
        power spectrum, C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where
        C_L^phiphi is the CMB lensing potential power spectrum.
        """
        # check the input
        exp = self._check_cmb_exp(exp)
        if exp == 'hd':
            fname = self.hd_lensing_noise_fname(include_fg=include_fg,
                                                hd_Lmax=hd_Lmax, **kwargs)
        else:
            if (hd_Lmax is not None) or (not include_fg):
                warnings.warn("Ignoring the `hd_Lmax` and `include_fg`"
                              f" arguments for `exp = '{exp}'`.")
            lmin = self.lmins[exp]
            lmax = self.lmaxs[exp]
            Lmax = self.Lmaxs[exp]
            exp_name = exp
            if (exp == 'so') and (self.hd_data_version not in ['v1.0', 'v1.1']):
                # updated to use "goal" noise levels for
                # enhanced SO (arXiv:arXiv:2503.00636):
                exp_name = 'enhanced_so'
            fname_root = f'{exp_name}_lmin{lmin}lmax{lmax}Lmax{Lmax}_nlkk'
            fname = os.path.join(config.data_path('noise'), f'{fname_root}.txt')
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname

    
    def hd_covmat_fname(self, cmb_type='delensed', include_fg=True,
                        hd_lmax=None, **kwargs):
        """Path to the file holding the covariance matrix of the mock
        CMB-HD TT, TE, EE, BB and CMB lensing power spectra for the the
        given CMB type (lensed or delensed).

        Parameters
        ----------
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a covariance matrix
            for delensed CMB TT, TE, EE, and BB power spectra, in
            addition to the CMB lensing spectrum. If `cmb_type='lensed'`,
            the covariance matrix is for lensed CMB spectra instead, but
            otherwise includes the same set of power spectra as the
            delensed case.

        Returns
        -------
        fname : str
            The path to the file.

        Other Parameters
        ----------------
        include_fg : bool, default=True
            If `True`, return the path to the covariance matrix that was
            calculated including the effects of residual extragalactic
            foregrounds in temperature. If `False`, return the path to
            the covariance matrix that was calculated by neglecting these
            effects; only available for v1.0 CMB-HD mock data with
            `cmb_type='lensed'` and `hd_lmax=None` or `hd_lmax=20100`.
        hd_lmax : int, default=None
            Used for CMB-HD covariance matrices that were calculated with
            a lower maximum (CMB and lensing) multipole than the default,
            baseline case. Only available for `'v1.0'` of the CMB-HD mock
            data when `cmb_type='delensed'` and `include_fg=True`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `block_covmat`
            method of `hd_mock_data.hd_data.HDMockData`. Ignored for v1.0
            CMB-HD mock data if `include_fg=False` or `hd_lmax` is lower
            than the default value.

        Raises
        ------
        ValueError
            If the requested covariance matrix does not exist.
        """
        # check if the requested covariance matrix is available
        # for this HD data version and combination of arguments:
        dinfo = 'covariance matrix'
        self._check_hd_fgs(include_fg=include_fg, cmb_type=cmb_type, data_info=dinfo)
        hd_lmax = self._check_hd_lmax(hd_lmax, cmb_type=cmb_type, data_info=dinfo)
        if cmb_type not in ['lensed', 'delensed']:
            raise ValueError(f"Invalid `{cmb_type = }`. The options for "
                             "CMB-HD are either `'lensed'` or `'delensed'`.")
        # if `hd_lmax` is the default value and `include_fg=True`,
        # get the file for this HD mock data version; otherwise, get the
        # file provided with `hdfisher`:
        if include_fg and (hd_lmax >= self.lmaxs['hd']):
            fname = self.hd_datalib.block_covmat_fname(cmb_type, **kwargs)
        else:
            # get the multipole ranges:
            lmin = self.lmins['hd']
            ell_info = f'lmin{lmin}lmax{hd_lmax}lmaxTT{hd_lmax}Lmax{hd_lmax}'
            extra_info = '' if include_fg else '_nofg'
            fname_root = f'hd{extra_info}_fsky0pt6_{ell_info}_binned_{cmb_type}_cov'
            fname = os.path.join(config.data_path('covmats'), f'{fname_root}.txt')
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname


    def cmb_covmat_fname(self, exp, cmb_type='delensed',
                         include_fg=True, hd_lmax=None, **kwargs):
        """Path to the covariance matrix of the mock CMB TT, TE, EE, BB
        and CMB lensing power spectra for the given experimental
        configuration and CMB type (lensed or delensed).

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a covariance matrix
            for delensed CMB TT, TE, EE, and BB power spectra, in
            addition to the CMB lensing spectrum. If `cmb_type='lensed'`,
            the covariance matrix is for lensed CMB spectra instead, but
            otherwise includes the same set of power spectra as the
            delensed case.
            - If `exp='HD'`, the `cmb_type` can be either `'delensed'` or
              `'lensed'`.
            - If `exp='S4'`, the `cmb_type` must be `'delensed'`.
            - If `exp='SO'`, the `cmb_type` can be either `'delensed'` or
              `'lensed'` when using a version >= v1.2 of the CMB-HD mock
              data; these covariance matrices were computed using the
              "goal" enhanced SO noise levels in arXiv:2503.00636.
              For lower CMB-HD mock data versions (v1.0 and v1.1), the
              `cmb_type` must be `'delensed'` if `exp='SO'`; this
              covariance matrix was calculated using the "goal" SO noise
              levels in arXiv:1808.07445.

        Returns
        -------
        fname : str
            The path to the file.

        Other Parameters
        ----------------
        include_fg : bool, default=True
            If `True`, return the path to the covariance matrix that was
            calculated including the effects of residual extragalactic
            foregrounds in temperature. If `False`, return the path to
            the covariance matrix that was calculated by neglecting these
            effects; only available for v1.0 CMB-HD mock data with
            `cmb_type='lensed'` and `hd_lmax=None` or `hd_lmax=20100`.
            Ignored if `exp` is not `'HD'`.
        hd_lmax : int, default=None
            Used for CMB-HD covariance matrices that were calculated with
            a lower maximum (CMB and lensing) multipole than the default,
            baseline case. Only available for `'v1.0'` of the CMB-HD mock
            data when `cmb_type='delensed'` and `include_fg=True`.
            Ignored if `exp` is not `'HD'`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `block_covmat`
            method of `hd_mock_data.hd_data.HDMockData`. Ignored if `exp`
            is not `'HD'`, and for v1.0 CMB-HD mock data if
            `include_fg=False` or `hd_lmax` is lower than the default
            value.

        Raises
        ------
        ValueError
            If the requested covariance matrix does not exist.

        Warns
        -----
        If the values of `include_fg` or `hd_lmax` were provided, but
        will be ignored.
        """
        # check the input
        exp = self._check_cmb_exp(exp)
        cmb_type = cmb_type.lower()
        if exp == 'hd':
            fname = self.hd_covmat_fname(cmb_type=cmb_type, include_fg=include_fg,
                                         hd_lmax=hd_lmax, **kwargs)
        else:
            if (hd_lmax is not None) or (not include_fg):
                warnings.warn("Ignoring the `hd_lmax` and `include_fg`"
                              f" arguments for `exp = '{exp}'`.")
            # check if we have the requested covmat:
            if cmb_type not in self.cov_cmb_types[exp]:
                raise ValueError(f"Invalid `{cmb_type = }` for `{exp = }`. "
                                 f"The options are: {self.cov_cmb_types[exp]}")
            # get the file name:
            lmin = self.lmins[exp]
            lmax = self.lmaxs[exp]
            lmaxTT = self.lmaxsTT[exp]
            Lmax = self.Lmaxs[exp]
            ell_info = f'lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}'
            exp_name = exp
            if (exp == 'so') and (self.hd_data_version not in ['v1.0', 'v1.1']):
                # updated to use "goal" noise levels for
                # enhanced SO (arXiv:arXiv:2503.00636):
                exp_name = 'enhanced_so'
            fname_root = f'{exp_name}_fsky0pt6_{ell_info}_binned_{cmb_type}_cov'
            fname = os.path.join(config.data_path('covmats'), f'{fname_root}.txt')
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname


    def desi_theory_fname(self):
        """Returns the name of the file containing the theoretical BAO 
        measurement r_s/d_V(z) for mock DESI BAO. 

        See Also
        --------
        dataconfig.desi_theory_fname

        Notes
        -----
        This method is is defined here for backwards compatibility.
        """
        return desi_theory_fname()


    def desi_covmat_fname(self):
        """Returns the name of the covariance matrix for the mock DESI BAO
        measurements r_s/d_V(z).

        See Also
        --------
        dataconfig.desi_covmat_fname

        Notes
        -----
        This method is is defined here for backwards compatibility.
        """
        return desi_covmat_fname()


    def precomputed_desi_fisher_fname(self, use_H0=False):
        """Returns the name of a file holding a Fisher matrix calculated
        from the mock DESI BAO measurements and covariance matrix. 
        
        See Also
        --------
        dataconfig.precomputed_desi_fisher_fname

        Notes
        -----
        This method is is defined here for backwards compatibility.
        """
        return precomputed_desi_fisher_fname(use_H0=use_H0)


    def precomputed_cmb_fisher_fname(self, exp, cmb_type='delensed',
                                     with_desi=False, feedback=False,
                                     use_H0=False, hd_lmax=None,
                                     include_fg=True):
        """Path to a Fisher matrix calculated for the given experiment
        and kind of CMB spectra (lensed or delensed).

        These are the Fisher matrices from in MacInnis et. al. (2023),
        calculated using version `'v1.0'` of the CMB-HD mock data. For
        Fisher matrices calculated using later versions of the CMB-HD
        mock data, see the `example_hd_fisher_fname` method.

        The parameters in the Fisher matrix are the six LCDM parameters,
        the effective number of relativistic species, and the sum of the
        neutrino masses. If `feedback=True`, the baryonic feedback
        parameter of the single-parameter HMCode2020 + feedback model is
        also included. A Gaussian prior on the optical depth of
        sigma(tau) = 0.007 has already been applied.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a Fisher matrix
            calculated from delensed CMB TT, TE, EE, and BB power
            spectra, in addition to the CMB lensing convergence power
            spectrum. If `cmb_type='lensed'`, the Fisher matrix was
            computed with lensed CMB spectra instead of delensed.
            - If `exp` is 'SO'` or 'S4'`, the `cmb_type` must be
              `'delensed'`.
            - If `exp` is `'HD'`, the `cmb_type` may be `'lensed'` or
              `'delensed'` if `hd_lmax` is `None` (or `20100`) and
              `feedback=False`; if `include_fg=False`, the `cmb_type`
              must be `'lensed'`; otherwise, the `cmb_type` must be
              `'delensed'`.
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the
            angular scale of the sound horizon at last scattering
            (multiplied by 100) is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB
            spectra. If `True`, the Fisher matrix is the sum of a CMB and
            a mock DESI BAO Fisher matrix.
        feedback : bool, default=False
            If `True`, the Fisher matrix was calculated with the
            HMCode2020 + baryonic feedback non-linear model, and also
            contains the baryonic feedback parameter of this model (i.e.,
            a total of 9 parameters). If `False`, the Fisher matrix was
            calculated with the HMCode2016 CDM-only model. Only available
            for `exp='HD'` if `cmb_type='delensed'` and `hd_lmax=None`.
            Ignored if `exp` is not `'HD'`.
        include_fg : bool, default=True
            If `True`, return the path to a CMB-HD Fisher matrix that was
            calculated including the effects of residual extragalactic
            foregrounds in temperature. If `False`, return the path to a
            Fisher matrix that was calculated by neglecting these
            effects; only available for v1.0 CMB-HD mock data with
            `cmb_type='lensed'` and `hd_lmax=None` or `hd_lmax=20100`.
            Ignored if `exp` is not `'HD'`.
        hd_lmax : int, default=None
            Used for CMB-HD Fisher matrices that were calculated with a
            lower maximum (CMB and lensing) multipole than the default,
            baseline case. Only available for `'v1.0'` of the CMB-HD mock
            data when `cmb_type='delensed'` and `include_fg=True`.
            Ignored if `exp` is not `'HD'`.

        Returns
        -------
        fname : str
            The path to the file.

        Raises
        ------
        ValueError
            If the requested Fisher matrix does not exist.

        Warns
        -----
        If the `hd_lmax`, `include_fg`, or `feedback` arguments were
        changed from their default value, but will be ignored.

        See Also
        --------
        load_precomputed_cmb_fisher
        example_hd_fisher_fname
        """
        # check if the Fisher matrix was saved for
        # this combination of input arguments:
        cmb_type = cmb_type.lower()
        exp = self._check_cmb_exp(exp)
        if self.hd_data_version != 'v1.0':
            err_msg = ("The precomputed Fisher matrices from MacInnis et. al."
                       " (2023) are only available for version `'v1.0'` of"
                       " the CMB-HD mock data; you are using version"
                       f" {self.hd_data_version}.")
            if exp == 'hd':
                err_msg = (f"{err_msg} For later CMB-HD mock data versions, "
                           "see the `load_example_hd_fisher` method, or "
                           "calculate a new Fisher matrix.")
            raise ValueError(err_msg)
        if cmb_type not in self.cov_cmb_types[exp]:
            raise ValueError(f"Invalid `{cmb_type = }` for `{exp = }`. "
                             f"The options are: {self.cov_cmb_types[exp]}")
        # check other arguments for HD:
        if exp == 'hd':
            dinfo = 'Fisher matrix'
            self._check_hd_fgs(include_fg=include_fg, cmb_type=cmb_type, data_info=dinfo)
            hd_lmax = self._check_hd_lmax(hd_lmax, cmb_type=cmb_type, feedback=feedback,
                                          include_fg=include_fg, data_info=dinfo)
            if feedback and (cmb_type == 'lensed'):
                raise ValueError("There are no precomputed CMB-HD Fisher"
                                 " matrices saved for `cmb_type='lensed'`"
                                 " and `feedback=True`.")
        elif (hd_lmax is not None) or (not include_fg) or feedback:
            warnings.warn("Ignoring the `hd_lmax`, `include_fg`, and"
                          f" `feedback` arguments for `exp = '{exp}'`.")
        # get the file name:
        fname_info = [exp]
        if (exp == 'hd') and (not include_fg):
            fname_info.append('nofg')
        fname_info.append('fsky0pt6') # same for all (in v1.0 of HD data)
        lmin = self.lmins[exp]
        if (exp == 'hd') and (hd_lmax is not None):
            lmax = hd_lmax
            lmaxTT = lmax
            Lmax = lmax
        else:
            lmax = self.lmaxs[exp]
            lmaxTT = self.lmaxsTT[exp]
            Lmax = self.Lmaxs[exp]
        fname_info.append(f'lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}')
        fname_info.append(cmb_type)
        if with_desi:
            fname_info.append('desi_bao')
        if (exp == 'hd') and feedback:
            fname_info.append('feedback')
        if use_H0:
            fname_info.append('useH0')
        fname_root = '_'.join(fname_info)
        fisher_dir = config.data_path(f'fisher_matrices')
        fname = os.path.join(fisher_dir, f'{fname_root}_fisher.txt')
        return fname
        


    # ----- functions that load the data: -----


    def load_example_hd_fisher(self, cmb_type='delensed',
                               use_H0=False, with_desi=False):
        """Load an example CMB-HD Fisher matrix that was calculated with
        the correct `hd_data_version`.

        All Fisher matrices contain 8 parameters (LCDM + N_eff + sum m_nu)
        and all have a Gaussian prior on the optical depth applied. For
        versions 1.0 and 1.1 of the CMB-HD mock data, the prior is
        sigma(tau) = 0.007; for version 1.2, it is sigma(tau) = 0.005.

        Parameters
        ----------
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the Fisher matrix was calculated
            with delensed CMB TT, TE, EE, and BB power spectra, in
            addition to the CMB lensing convergence power spectrum. If
            `cmb_type='lensed'`, the Fisher matrix was computed with
            lensed CMB spectra instead of delensed.
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the
            angular scale of the sound horizon at last scattering
            (multiplied by 100) is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB
            spectra. If `True`, the Fisher matrix is the sum of a CMB and
            a mock DESI BAO Fisher matrix.

        Returns
        -------
        fisher_matrix : array_like of float
            The Fisher matrix, with shape `(8,8)`.
        fisher_params : list of str
            A list of names for the parameters in the Fisher matrix, in
            the same order as their corresponding rows/columns.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed.
        """
        fname = self.example_hd_fisher_fname(cmb_type=cmb_type,
                                             use_H0=use_H0,
                                             with_desi=with_desi)
        fisher_matrix, fisher_params = utils.load_fisher_matrix(fname)
        return fisher_matrix, fisher_params


    def load_cmb_theory_spectra(self, exp, cmb_type, output_lmax=None,
                                feedback=False, hd_lmax=None, **kwargs):
        """Theory CMB and CMB lensing power spectra for a given CMB
        experiment and CMB type (e.g. delensed).

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        cmb_type : str
            The name of the kind of power spectra. Must be either
            `'lensed'`, `'delensed'`, or `'unlensed'`.
        output_lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by
            `output_lmax`. Otherwise, use the default maximum multipole
            for the given `exp`.
        feedback : bool, default=False
            If `True`, returns power spectra calculated with the
            HMCode2020 + baryonic feedback non-linear model. Otherwise,
            the the HMCode2016 CDM-only model was used. Ignored if `exp`
            is not `'HD'`.

        Returns
        -------
        theo : dict of array_like of float
            A dictionary with a key `'ells'` holding the multipoles of
            the power spectra; keys `'tt'`, `'te'`, `'ee'`, and `'bb'`
            for the CMB power spectra for the requested `cmb_type`; and a
            key`'kk'` for the CMB lensing spectrum.

        Other Parameters
        ----------------
        hd_lmax : int or None, default=None
            Used for CMB-HD power spectra that were calculated with a
            lower maximum multipole than the baseline case. Only
            available for `'v1.0'` of the CMB-HD mock data. Ignored if
            `exp` is not `'HD'`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `cmb_theory_fname`
            method of `hd_mock_data.hd_data.HDMockData`. Ignored if `exp`
            is not `'HD'`; also ignored for v1.0 CMB-HD mock data if
            `hd_lmax` is lower than the default value.

        Raises
        ------
        ValueError
            If the `exp`, `cmb_type`, or `hd_lmax` value is invalid.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored.

        Notes
        -----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum is C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where L is the
        lensing multipole and C_L^phiphi is the CMB lensing potential
        power spectrum.

        See also
        --------
        load_all_cmb_theory_spectra
        cmb_theory_fname
        """
        exp = self._check_cmb_exp(exp)
        cmb_type = cmb_type.lower()
        if cmb_type not in self.cmb_types:
            raise ValueError(f"Invalid `{cmb_type = }`. The options are: {self.cmb_types}.")
        # get the file name and load in the spectra
        fname = self.cmb_theory_fname(exp, cmb_type, feedback=feedback, hd_lmax=hd_lmax, **kwargs)
        theo = utils.load_from_file(fname, config.theo_cols)
        if output_lmax is not None:
            lmax = int(output_lmax)
            theo_lmax = int(theo['ells'][-1])
            if lmax > theo_lmax:
                warnings.warn("You requested theory power spectra out to "
                              f"`{output_lmax = }`, but the spectra were "
                              f"only computed out to {theo_lmax}.")
        else:
            lmax = max([self.lmaxs[exp], self.Lmaxs[exp], self.lmaxsTT[exp]])
        for key in theo.keys():
            theo[key] = theo[key][:lmax+1]
        return theo


    def load_all_cmb_theory_spectra(self, exp, output_lmax=None,
                                    feedback=False, hd_lmax=None, **kwargs):
        """Theory lensed, delensed, and unlensed CMB and CMB lensing
        power spectra for a given CMB experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        output_lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by
            `output_lmax`. Otherwise, use the default maximum multipole
            for the given `exp`.
        feedback : bool, default=False
            If `True`, returns power spectra calculated with the
            HMCode2020 + baryonic feedback non-linear model. Otherwise,
            the the HMCode2016 CDM-only model was used. Ignored if `exp`
            is not `'HD'`.

        Returns
        -------
        theo : dict of dict of array_like of float
            A nested dictionary with a key for each `cmb_type`
            (`'lensed'`, `'delensed'`, and `'unlensed'`);
            `theo[cmb_type]` is a dictionary with a key `'ells'` for the
            multipoles of the power spectra; keys `'tt'`, `'te'`, `'ee'`,
            and `'bb'` for the CMB power spectra for that `cmb_type`; and
            a key`'kk'` for the CMB lensing power spectrum.

        Other Parameters
        ----------------
        hd_lmax : int or None, default=None
            Used for CMB-HD power spectra that were calculated with a
            lower maximum multipole than the baseline case. Only
            available for `'v1.0'` of the CMB-HD mock data. Ignored if
            `exp` is not `'HD'`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `cmb_theory_fname`
            method of `hd_mock_data.hd_data.HDMockData`. Ignored if `exp`
            is not `'HD'`; also ignored for v1.0 CMB-HD mock data if
            `hd_lmax` is lower than the default value.

        Raises
        ------
        ValueError
            If the `exp` or `hd_lmax` value is invalid.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored.

        Notes
        -----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum is C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where L is the
        lensing multipole and C_L^phiphi is the CMB lensing potential
        power spectrum.

        See also
        --------
        load_cmb_theory_spectra
        cmb_theory_fname
        """
        theo = {}
        theo_kwargs = {'feedback': feedback, 'output_lmax': output_lmax,
                       'hd_lmax': hd_lmax, **kwargs}
        for cmb_type in self.cmb_types:
            theo[cmb_type] = self.load_cmb_theory_spectra(exp, cmb_type, **theo_kwargs)
        return theo


    def load_residual_cmb_lensing_spectrum(self, exp, output_Lmax=None,
                                           feedback=False, hd_Lmax=None):
        """The residual CMB lensing power spectrum, i.e. the difference
        between the total lensing power spectrum and the Wiener-filtered
        lensing power spectrum for a given experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        output_Lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by
            `output_Lmax`. Otherwise, use the default maximum lensing
            multipole for the given `exp`.
        feedback : bool, default=False
            If `True`, the power spectra were calculated with the
            HMCode2020 + baryonic feedback non-linear model. Otherwise,
            they were calculated with the HMCode2016 CDM-only model.
            Ignored if `exp` is not `'HD'`.
        hd_Lmax : int or None, default=None
            Used for CMB-HD power spectra that were calculated with a
            lower maximum multipole than the baseline case. Only
            available for `'v1.0'` of the CMB-HD mock data. Ignored if
            `exp` is not `'HD'`.

        Returns
        -------
        L, clkk_res : array_like of float
            The lensing multipoles and residual lensing power spectrum,
            respectively.

        Raises
        ------
        ValueError
            If the `exp` or the `hd_Lmax` value is invalid.

        Warns
        -----
        If the `hd_Lmax` and `feedback` arguments will be ignored.

        Notes
        -----
        The (residual) lensing power spectrum is
        C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where L is the lensing
        multipole and C_L^phiphi is the CMB lensing potential power
        spectrum.

        See Also
        --------
        theory.get_residual_lensing
        """
        exp = self._check_cmb_exp(exp)
        fname = self.cmb_theory_fname(exp, 'clkk_res', hd_lmax=hd_Lmax, feedback=feedback)
        clkk_res = np.loadtxt(fname)
        L = np.arange(len(clkk_res))
        if output_Lmax is not None:
            Lmax = int(output_Lmax)
            theo_Lmax = len(clkk_res) - 1
            if Lmax > theo_Lmax:
                warnings.warn("You requested the residual lensing power "
                              f"spectrum out to `{output_Lmax = }`, but "
                              f"it was only computed out to {theo_Lmax}.")
        else:
            Lmax = self.Lmaxs[exp]
        clkk_res = clkk_res[:Lmax+1]
        L = L[:Lmax+1]
        return L, clkk_res


    def load_cmb_noise_spectra(self, exp, include_fg=True, output_lmax=None):
        """The power spectra of the noise on the CMB TT, TE, EE, and BB
        power spectra, coadded from 90 and 150 GHz.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, the temperature noise power spectrum at each
            frequency is the sum of the instrumental noise and the
            residual extragalactic foreground power spectrum. If `False`,
            it will only contain instrumental noise. Ignored if `exp` is
            not `'HD'`.
        output_lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by
            `output_lmax`. Otherwise, use the default maximum multipole
            for the given `exp`.

        Returns
        -------
        noise : dict of array_like of float
            A dictionary with a key `'ells'` for the multipoles of the
            noise power spectra, and keys `'tt'`, `'te'`, `'ee'`, and
            `'bb'` for the TT, TE, EE, and BB noise power spectra,
            respectively.

        Raises
        ------
        ValueError
            If the `exp` is invalid.

        Warns
        -----
        If the value of `include_fg` will be ignored.

        Notes
        -----
        The noise spectra are in units of uK^2, without any
        multiplicative factors applied.

        The noise levels used for `'SO'` and `'S4'` are those used in
        MacInnis et. al. (2023). The SO-like noise curves have not been
        updated with the noise levels given in arXiv:2503.00636.
        """
        exp = self._check_cmb_exp(exp)
        fname = self.cmb_noise_fname(exp, include_fg=include_fg)
        noise = utils.load_from_file(fname, config.noise_cols)
        if output_lmax is not None:
            lmax = int(output_lmax)
            noise_lmax = int(noise['ells'][-1])
            if lmax > noise_lmax:
                warnings.warn("You requested noise power spectra out to "
                              f"`{output_lmax = }`, but the spectra were "
                              f"only computed out to {noise_lmax}.")
        else: # automatically trim to be consistent with theory spectra
            lmax = max([self.lmaxs[exp], self.Lmaxs[exp], self.lmaxsTT[exp]])
        for key in noise.keys():
            noise[key] = noise[key][:lmax+1]
        return noise


    def load_cmb_lensing_noise_spectrum(self, exp, output_Lmax=None,
                                        include_fg=True, hd_Lmax=None, **kwargs):
        """The CMB lensing noise power spectrum for the given CMB
        experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'HD'`,
            `'SO'`, or `'S4'`. The name is case-insensitive.
        output_Lmax : int or None, default=None
            If provided, cut the spectrum at a maximum lensing multipole
            given by `output_Lmax`. Otherwise, use the default maximum
            multipole for the given `exp`.

        Returns
        -------
        L, nlkk : array_like of float
            The lensing multipoles and lensing noise spectrum,
            respectively.

        Other Parameters
        ----------------
        include_fg : bool, default=True
            If `True`, return the lensing noise that was calculated
            including the effects of residual extragalactic foregrounds
            in temperature. If `False`, return the lensing noise that was
            calculated by neglecting these effects; only available for
            v1.0 CMB-HD mock data and `hd_Lmax=None` or `hd_Lmax=20100`.
            Ignored if `exp` is not `'HD'`.
        hd_Lmax : int or None, default=None
            Used for CMB-HD lensing noise spectra that were calculated
            with a lower maximum (lensing and CMB) multipole than the
            baseline case. Only available for `'v1.0'` of the CMB-HD mock
            data when `include_fg=True`. Ignored if `exp` is not `'HD'`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the
            `lensing_noise_fname` method of
            `hd_mock_data.hd_data.HDMockData`. Ignored if `exp` is not
            `'HD'`, or for v1.0 CMB-HD mock data if `include_fg=False`
            or `hd_Lmax` is lower than the default value.

        Notes
        -----
        The CMB lensing noise N_L^kk is the noise on the CMB lensing
        spectrum C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where C_L^phiphi
        is the CMB lensing potential power spectrum and L is the lensing
        multipole.
        """
        exp = self._check_cmb_exp(exp)
        fname = self.cmb_lensing_noise_fname(exp, include_fg=include_fg,
                                             hd_Lmax=hd_Lmax, **kwargs)
        L, nlkk = np.loadtxt(fname, unpack=True)
        if output_Lmax is not None:
            Lmax = int(output_Lmax)
            theo_Lmax = int(L[-1])
            if Lmax > theo_Lmax:
                warnings.warn("You requested the lensing noise power spectrum"
                              f" out to `{output_Lmax = }`, but it was only"
                              f" computed out to {theo_Lmax}.")
        else:
            Lmax = self.Lmaxs[exp]
        nlkk = nlkk[:Lmax+1]
        L = L[:Lmax+1]
        return L, nlkk

   
    def load_hd_fg_spectra(self, frequency, output_lmax=None):
        """Returns a dictionary holding power spectra of residual 
        extragalactic foregrounds in temperature at the given frequency 
        for CMB-HD.

        Parameters
        ----------
        frequency : str or int
            The frequency for the foreground power spectra. Pass `90` or 
            `'f090'` for the foreground components at 90 GHz, or pass 
            `150` or  `'f150'` for the foregrounds at 150 GHz.
        output_lmax : int or None, default=None
            If provided, cut the spectra at a maximum multipole given by 
            the `output_lmax` value.

        Returns
        -------
        fgs : dict of array_like of float
            A dictionary of one-dimensional arrays for the multipoles
            (with key `'ells'`) and the residual power spectrum of each
            extragalactic foreground component (with a `str` key for the
            component name). See the `fg_spectra` method of the
            `hd_mock_data.hd_data.Data` class for the returned components.

        Notes
        -----
        The power spectra are in units of uK^2, without any multiplicative
        factors applied.

        See also
        --------
        load_hd_coadd_fg_spectrum
        hd_mock_data.hd_data.Data.fg_spectra
        """
        fgs = self.hd_datalib.fg_spectra(frequency, output_lmax=output_lmax)
        return fgs


    def load_hd_coadd_fg_spectrum(self, output_lmax=None):
        """Returns the power spectrum of the residual extragalactic 
        foregrounds in temperature for CMB-HD, coadded from 90 and 
        150 GHz. 

        Parameters
        ----------
        output_lmax : int or None, default=None
            If provided, cut the spectra at a maximum multipole given by the
            `output_lmax` value.

        Returns
        -------
        ells, coadd_fg_cls : array_like of float
            One-dimensional arrays holding the multipoles of the foreground
            power spectrum (`ells`) and the coadded foreground power spectrum
            (`coadd_fg_cls`).

        Notes
        -----
        The power spectrum is in units of uK^2, without any multiplicative
        factors applied.

        See also
        --------
        load_hd_fg_spectra
        hd_mock_data.hd_data.Data.coadded_fg_spectrum
        """
        ells, coadd_fg_cls = self.hd_datalib.coadded_fg_spectrum(output_lmax=output_lmax)
        return ells, coadd_fg_cls


    def load_cmb_covmat(self, exp, cmb_type='delensed',
                        include_fg=True, hd_lmax=None, **kwargs):
        """Covariance matrix for the mock CMB TT, TE, EE, BB and CMB
        lensing power spectra for the given experimental configuration
        and CMB type (lensed or delensed).

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, returns the covariance matrix for
            delensed CMB TT, TE, EE, and BB power spectra and the CMB
            lensing power spectrum. If `cmb_type='lensed'`, returns the
            covariance matrix for lensed CMB spectra instead, but
            otherwise includes the same set of power spectra as the
            delensed case.
            - If `exp='HD'`, the `cmb_type` can be either `'delensed'` or
              `'lensed'`.
            - If `exp='S4'`, the `cmb_type` must be `'delensed'`.
            - If `exp='SO'`, the `cmb_type` can be either `'delensed'` or
              `'lensed'` when using a version >= v1.2 of the CMB-HD mock
              data; these covariance matrices were calculated using the
              "goal" enhanced SO noise levels in arXiv:2503.00636.
              For lower CMB-HD mock data versions (v1.0 and v1.1), the
              `cmb_type` must be `'delensed'` if `exp='SO'`; this
              covariance matrix was calculated using the "goal" SO noise
              levels in arXiv:1808.07445.

        Returns
        -------
        covmat : array_like of float
            The full covariance matrix for the mock CMB and CMB lensing
            power spectra.

        Other Parameters
        ----------------
        include_fg : bool, default=True
            If `True`, return the covariance matrix that was calculated
            including the effects of residual extragalactic foregrounds
            in temperature. If `False`, return the covariance matrix that
            was calculated by neglecting these effects; only available
            for v1.0 CMB-HD mock data with `cmb_type='lensed'` and
            `hd_lmax=None` or `hd_lmax=20100`. Ignored if `exp` is not
            `'HD'`.
        hd_lmax : int, default=None
            Used for CMB-HD covariance matrices that were calculated with
            a lower maximum (CMB and lensing) multipole than the default,
            baseline case. Only available for `'v1.0'` of the CMB-HD mock
            data when `cmb_type='delensed'` and `include_fg=True`.
            Ignored if `exp` is not `'HD'`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the `block_covmat`
            method of `hd_mock_data.hd_data.HDMockData`. Ignored if `exp`
            is not `'HD'`, and for v1.0 CMB-HD mock data if
            `include_fg=False` or `hd_lmax` is lower than the default
            value.

        Raises
        ------
        ValueError
            If the requested covariance matrix does not exist.

        Notes
        -----
        The covariance matrix is binned and contains 25 blocks; each
        block has shape `(nbin_x, nbin_y)` where `nbin_x` is the number
        of bins in the multipole range of the power spectrum `x` (TT, TE,
        etc.) for the given experimental configuration. The diagonal
        blocks contain the covariance matrices for TT x TT, TE x TE,
        EE x EE, BB x BB, and kk x kk (in that order), where kk refers to
        the CMB lensing convergence power spectrum. The off-diagonal
        blocks contain the cross-covariances, e.g. TT x TE, TT x EE, etc.

        We use units of  uK^2 for the CMB spectra, and do not apply any
        multiplicative factors. We use the CMB lensing convergence power
        spectrum, C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where C_L^phiphi
        is the CMB lensing potential power spectrum and L is the lensing
        multipole.

        See also
        --------
        cmb_covmat_fname
        """
        fname = self.cmb_covmat_fname(exp, cmb_type=cmb_type,
                                      include_fg=include_fg,
                                      hd_lmax=hd_lmax, **kwargs)
        covmat = np.loadtxt(fname)
        return covmat


    def load_desi_theory(self):
        """Returns one-dimensional arrays containing the redshifts (`z`)
        and the theoretical BAO measurement r_s/d_V(z) at those redshifts
        (`rs_dv`) for mock DESI BAO. 

        See also
        --------
        dataconfig.load_desi_theory

        Notes
        -----
        This method is is defined here for backwards compatibility.
        """
        return load_desi_theory()


    def load_desi_covmat(self):
        """Returns a two-dimensional array holding the covariance matrix
        for the mock DESI BAO measurements, r_s/d_V(z).

        See also
        --------
        dataconfig.load_desi_covmat

        Notes
        -----
        This method is is defined here for backwards compatibility.
        """
        return load_desi_covmat()


    def load_bin_edges(self):
        """Returns a one-dimensional array holding the upper bin edge for
        each bin, except the first element, which is the lower bin edge of
        the first bin.
        """
        bin_edges = self.hd_datalib.bin_edges()
        return bin_edges
    
    
    def load_precomputed_desi_fisher(self, use_H0=False):
        """Returns a Fisher matrix calculated from the mock DESI BAO 
        measurements and covariance matrix.

        See Also
        --------
        dataconfig.load_precomputed_desi_fisher

        Notes
        -----
        This method is defined here for backwards compatibility.
        """
        fisher_matrix, fisher_params = load_precomputed_desi_fisher(use_H0=use_H0)
        return fisher_matrix, fisher_params


    def load_precomputed_cmb_fisher(self, exp, cmb_type='delensed',
                                     with_desi=False, feedback=False,
                                     use_H0=False, hd_lmax=None,
                                     include_fg=True):
        """A Fisher matrix calculated for the given experiment and kind
        of CMB spectra (lensed or delensed).

        These are the Fisher matrices from in MacInnis et. al. (2023),
        calculated using version `'v1.0'` of the CMB-HD mock data. For
        Fisher matrices calculated using later versions of the CMB-HD
        mock data, see the `example_hd_fisher_fname` method.

        The parameters in the Fisher matrix are the six LCDM parameters,
        the effective number of relativistic species, and the sum of the
        neutrino masses. If `feedback=True`, the baryonic feedback
        parameter of the single-parameter HMCode2020 + feedback model is
        also included. A Gaussian prior on the optical depth of
        sigma(tau) = 0.007 has already been applied.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the Fisher matrix was calculated
            from delensed CMB TT, TE, EE, and BB power  spectra, in
            addition to the CMB lensing convergence power spectrum. If
            `cmb_type='lensed'`, the Fisher matrix was computed with
            lensed CMB spectra instead of delensed.
            - If `exp` is 'SO'` or 'S4'`, the `cmb_type` must be
              `'delensed'`.
            - If `exp` is `'HD'`, the `cmb_type` may be `'lensed'` or
              `'delensed'` if `hd_lmax` is `None` (or `20100`) and
              `feedback=False`; if `include_fg=False`, the `cmb_type`
              must be `'lensed'`; otherwise, the `cmb_type` must be
              `'delensed'`.
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the
            angular scale of the sound horizon at last scattering
            (multiplied by 100) is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB
            spectra. If `True`, the Fisher matrix is the sum of a CMB and
            a mock DESI BAO Fisher matrix.
        feedback : bool, default=False
            If `True`, the Fisher matrix was calculated with the
            HMCode2020 + baryonic feedback non-linear model, and also
            contains the baryonic feedback parameter of this model (i.e.,
            a total of 9 parameters). If `False`, the Fisher matrix was
            calculated with the HMCode2016 CDM-only model. Only available
            for `exp='HD'` if `cmb_type='delensed'` and `hd_lmax=None`.
            Ignored if `exp` is not `'HD'`.
        include_fg : bool, default=True
            If `True`, return the CMB-HD Fisher matrix that was
            calculated including the effects of residual extragalactic
            foregrounds in temperature. If `False`, return the Fisher
            matrix that was calculated by neglecting these effects; only
            available for v1.0 CMB-HD mock data with `cmb_type='lensed'`
            and `hd_lmax=None` or `hd_lmax=20100`.  Ignored if `exp` is
            not `'HD'`.
        hd_lmax : int, default=None
            Used for CMB-HD Fisher matrices that were calculated with a
            lower maximum (CMB and lensing) multipole than the default,
            baseline case. Only available for `'v1.0'` of the CMB-HD mock
            data when `cmb_type='delensed'` and `include_fg=True`.
            Ignored if `exp` is not `'HD'`.

        Returns
        -------
        fisher_matrix : array_like of float
            The eight-parameter (if `feedback=False`) or nine-parameter
            (if `feedback=True`) Fisher matrix for the given experiment,
            or its combination with mock DESI BAO data if
            `with_desi=True`.
        fisher_params : list of str
            A list of parameter names for the parameters in the Fisher
            matrix, in the same order as their corresponding rows/columns.

        Raises
        ------
        ValueError
            If the requested Fisher matrix does not exist.

        Warns
        -----
        If the `hd_lmax`, `include_fg`, or `feedback` arguments were
        changed from their default value, but will be ignored.

        See also
        --------
        precomputed_cmb_fisher_fname
        load_example_hd_fisher
        """
        fname = self.precomputed_cmb_fisher_fname(exp, cmb_type=cmb_type, use_H0=use_H0,
                                                  with_desi=with_desi, hd_lmax=hd_lmax,
                                                  include_fg=include_fg, feedback=feedback)
        fisher_matrix, fisher_params = utils.load_fisher_matrix(fname)
        return fisher_matrix, fisher_params


    # ----- convenience functions -----

    def desi_redshifts(self):
        """Returns a one-dimensional array holding the redshifts at which
        the mock DESI BAO theory and covariance matrix were calculated.

        See Also
        --------
        dataconfig.desi_redshifts

        Notes
        -----
        This method is defined here for backwards compatibility.
        """
        return desi_redshifts()


    def binning_matrix(self, exp):
        """Returns a binning matrix of shape (num_bins, num_ells) to bin
        spectra in the multipole range for the given experiment, where
        num_bins is the total number of bins in the multipole range, and
        num_ells is the number of multipoles from 2 to the maximum
        multipole value (`lmax`), i.e. it is equal to `lmax - 1`.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.

        Returns
        -------
        bmat : array_like of float
            The two-dimensional binning matrix.

        Raises
        ------
        ValueError
            If an invalid `exp` was passed.

        Notes
        -----
        The spectra to be binned are expected to begin at a multipole of
        ell = 2, even if the minimum multipole is greater than this value.
        To bin spectrum with values from `lmin = 0` to `lmax` held in an 
        array `cl` of length `lmax+1`, you would multiply it by the
        binning matrix as `bmat @ cl[2:]`.
        """
        exp = self._check_cmb_exp(exp)
        bmat = self.hd_datalib.binning_matrix(lmin=self.lmins[exp], lmax=self.lmaxs[exp])
        return bmat


    def lbin(self, exp):
        """Returns the binned multipoles (i.e., the bin centers) in the
        range for the given experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be `'HD`', `'SO'`,
            or `'S4'`. The name is case-insensitive.

        Returns
        -------
        lbin : array_like of float
            The binned multipoles.

        Raises
        ------
        ValueError
            If an invalid `exp` was passed.

        See also
        --------
        binning_matrix
        """
        exp = self._check_cmb_exp(exp)
        bmat = self.binning_matrix(exp)
        ells = np.arange(2, self.lmaxs[exp] + 1)
        lbin = bmat @ ells
        return lbin


    def fiducial_params(self, param_names=None, feedback=False):
        """Returns a dictionary containing cosmological parameter names
        and their fiducial values, along with any other names and values
        (e.g., for CAMB accuracy parameters) that are used when
        calculating the theory.

        Parameters
        ----------
        param_names : None or list of str, default=None
            A list of parameter names. If provided, only these names will
            appear as keys in the returned dictionary. If `None`, all 
            available parameters are included.
        feedback : bool, default=None
            If `True`, the dictionary will contain the name of the 
            HMCode2020 + feedback model that is passed to CAMB, and the
            name and fiducial value of its baryonic feedback parameter.

        Returns
        -------
        params : dict
            A dictionary containing parameter names and their fiducial
            values.

        Raises
        ------
        ValueError
            If `param_names` is not `None`, but contains a name that is
            not specified in the fiducial parameter file.

        See also
        --------
        fiducial_param_file
        """
        param_file = self.fiducial_param_file(feedback=feedback)
        fid_params = theory.get_params(param_file=param_file)
        params = {}
        if param_names is not None:
            for param in param_names:
                if param not in fid_params.keys():
                    err_msg = (f"Invalid parameter name `'{param}'` in "
                               "`param_names`: there are only fiducial "
                               f"values set for {fid_params.keys()}.")
                    raise ValueError(err_msg)
                else:
                    params[param] = fid_params[param]
        else:
            params = fid_params.copy()
        return params

