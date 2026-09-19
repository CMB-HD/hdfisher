import os
import warnings
import numpy as np
from hd_mock_data import hd_data
from . import utils, theory, mpi, config


# create a custom warning category to always issue a warning about
# the HD data version, without changing the filter for general `UserWarning`s:

class DataVersionWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return str(self.message)

def warn(msg):
    warnings.warn(msg, category=DataVersionWarning, stacklevel=2)

# if using MPI, only issue a warning from the main MPI process:
if mpi.rank == 0:
    warnings.simplefilter('always', category=DataVersionWarning)
else:
    warnings.simplefilter('ignore', category=DataVersionWarning)
    warnings.simplefilter('ignore', category=UserWarning)



class Data:
    """Holds the experimental configuration information for the experiments
    considered in MacInnis et. al. (2023), and provides access to their 
    associated files that are provided with `hdfisher`.
    """
    
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
        self.hd_data_version = hd_data_version.lower()
        self.hd_datalib = hd_data.HDMockData(version=self.hd_data_version)

        # constants
        self.cmb_exps = ['so', 's4', 'hd']
        self.fsky = 0.6
        
        # columns for spectra in files
        self.theo_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk']
        self.noise_cols = self.theo_cols[:-1].copy()
        # note that kSZ is only reionization kSZ for v1.0 of HD data; 
        #  later versions include both reion. and late-time kSZ
        self.fg_cols = ['ells', 'ksz', 'tsz', 'cib', 'radio'] 
        
        self.cmb_spectra = self.theo_cols[1:-1].copy()
        self.cov_spectra = self.theo_cols[1:].copy()
        
        # for theory spectra
        self.cmb_types = ['lensed', 'delensed', 'unlensed']
        
        # multipole ranges
        self.lmins = {'hd': 30, 'so': 30, 's4': 30}
        self.lmaxs = {'hd': 20100, 'so': 5000, 's4': 5000}
        self.lmaxsTT = {'hd': 20100, 'so': 3000, 's4': 3000}
        self.Lmaxs = {'hd': 20100, 'so': 3000, 's4': 3000}
        
        self.ell_ranges = {}
        for exp in self.cmb_exps:
            self.ell_ranges[exp] = {'tt': [self.lmins[exp], self.lmaxsTT[exp]],
                                    'te': [self.lmins[exp], self.lmaxs[exp]],
                                    'ee': [self.lmins[exp], self.lmaxs[exp]],
                                    'bb': [self.lmins[exp], self.lmaxs[exp]],
                                    'kk': [self.lmins[exp], self.Lmaxs[exp]]}
        
        # CMB-HD has some additional files calculated for a lower maximum multipole
        # (NOTE: these were only calculated for v1.0 of the HD data)
        self.hd_lmaxs = [1000, 3000, 5000, 10000]
        
        # file names
        self.bin_edges_fname = self.hd_datalib.bin_edges_fname()


    # ----- functions to check arguments passed to other functions: -----

    def check_cmb_exp(self, exp, valid_exps=None):
        """Checks whether the value passed for `exp` is valid."""
        exp = exp.lower()
        if valid_exps is None:
            valid_exps = self.cmb_exps
        if exp not in valid_exps:
            err_msg = f"Invalid experiment name. You passed `exp = '{exp}'`, but `exp` must be one of {valid_exps}."
            raise ValueError(err_msg)
        else:
            return exp


    def check_hd_lmax(self, hd_lmax, hd_version_warning=False, data_info=None):
        """Checks whether the value passed for `hd_lmax` is valid. 
        If `hd_version_warning = True`, warn the user if we need to use an
        older data version to load in the data with a lower lmax. In that 
        case, the `data_info` string is used to give a more descriptive
        warning.
        """
        if hd_lmax is None:
            hd_lmax = self.lmaxs['hd']
        else:
            hd_lmax = int(hd_lmax)
        valid_hd_lmaxs = self.hd_lmaxs + [self.lmaxs['hd']]
        if hd_lmax not in valid_hd_lmaxs:
            err_msg = f"Invalid `hd_lmax`: {hd_lmax}. Valid options are: {valid_hd_lmaxs}."
            raise ValueError(err_msg)
        else:
            # warn the user if they want a lower lmax, but also passed a
            #  `hd_data_version` that's higher than the `'v1.0'` for which
            #  the data was calculated for the lower lmax value:
            hd_version_info = f"`hd_data_version = '{self.hd_data_version}'`"
            if 'late' in self.hd_data_version.lower(): # include version number
                hd_version_info = f"{hd_version_info} (version '{self.hd_datalib.version}')"
            if data_info is None:
                data_info = 'data'
            msg = (f"You are using {hd_version_info}, but the CMB-HD data "
                   f"with `hd_lmax = {hd_lmax}` was only calculated for "
                   f"'v1.0'. The 'v1.0' {data_info} will be returned.")
            if hd_version_warning:
                if (hd_lmax < self.lmaxs['hd']) and (self.hd_datalib.version_number > 1.0):
                    warn(msg)
            return hd_lmax


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
        fname = config.fiducial_param_file(feedback=feedback)
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


    def example_hd_fisher_fname(self, cmb_type='delensed', use_H0=False, with_desi=False):
        """Returns the name of the file holding an example CMB-HD Fisher 
        matrix that was calculated with the correct `hd_data_version`.
        All Fisher matrices contain 8 parameters (LCDM + N_eff + sum m_nu)
        and all have a Gaussian prior of sigma(tau) = 0.007 applied.

        Parameters
        ----------
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a Fisher matrix calculated 
            from delensed CMB TT, TE, EE, and BB power spectra, in addition to 
            the CMB lensing spectrum. If `cmb_type='lensed'`, the Fisher matrix
            was computed with lensed CMB spectra instead, as well as the CMB
            lensing spectrum. 
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the angular 
            scale of the sound horizon at last scattering (multiplied by 100)
            is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB spectra.
            If `True`, the Fisher matrix is the sum of a CMB and a mock DESI BAO
            Fisher matrix.

        Returns
        -------
        fname : str
            The absolute path of the file holding the Fisher matrix.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed.
        """
        cmb_type = cmb_type.lower()
        if cmb_type not in ['lensed', 'delensed']:
            errmsg = (f"Invalid `cmb_type`: `'{cmb_type}'`. The `cmb_type` "
                      "must be `'lensed'` or `'delensed'`.")
            raise ValueError(errmsg)
        lmin = self.lmins['hd']
        lmax = self.lmaxs['hd']
        lmaxTT = self.lmaxsTT['hd']
        Lmax = self.Lmaxs['hd']
        ell_info = f'lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}'
        H0_info = '_useH0' if use_H0 else ''
        desi_info = '_desi_bao' if with_desi else ''
        fname_root = f'hd_fsky0pt6_{ell_info}_{cmb_type}{desi_info}{H0_info}'
        version = self.hd_datalib.version
        fisher_dir = os.path.join(config.data_path(f'fisher_matrices'), 'hd_examples')
        fname = os.path.join(fisher_dir, f'{fname_root}_fisher_{version}.txt')
        return fname


    def hd_theory_fname(self, spectrum_type, hd_lmax=None, feedback=False):
        """Returns the name of the file containing the theory CMB and lensing
        spectra for a given CMB experiment and CMB type (e.g. delensed).
        
        Parameters
        ----------
        spectrum_type : str
            The name of the kind of spectra. Must be either `'lensed'`, 
            `'delensed'`, or `'unlensed'` for files containing the CMB TT, TE, 
            EE, and BB spectra along with the lensing (kappa kappa) spectrum;
            or `'clkk_res'` for files containing only the residual lensing power.
        hd_lmax : int, default=None
            Used for CMB-HD spectra that were calculated with a lower maximum 
            multipole than the baseline case.
        feedback : bool, default=False
            If `True`, the file name returned will be for a file holding 
            theory calculated with the HMCode2020 + baryonic feedback 
            non-linear model, as opposed to the HMCode2016 CDM-only model.
        
        Returns
        -------
        fname : str
            The absolute path and name of the requested file.
    
        Raises
        ------
        ValueError 
            If the requested file does not exist.

        Warns
        -----
        If `hd_lmax < 20100` and you are  using a `hd_data_version` higher
        than the original `'v1.0'` for which the spectra were calculated
        to a lower maximum multipole.

        Note
        ----
        If `cmb_type = 'clkk_res'`, the file will contain a single column
        holding the residual CMB lensing power spectrum. Otherwise, the
        file will have a column for the multipoles of the spectra, the 
        CMB TT, TE, EE, and BB power spectra (in units of uK^2, without
        any multiplicative factors applied), and the lensing power spectrum,
        using the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where
        L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        if spectrum_type in self.cmb_types:
            data_info=f'{spectrum_type} theory'
        else:
            data_info = 'residual lensing power'
        hd_lmax = self.check_hd_lmax(hd_lmax, hd_version_warning=True, data_info=data_info)
        # make sure we have the file:
        if (hd_lmax < self.lmaxs['hd']) and feedback:
            errmsg = ("There is no CMB-HD theory saved with baryonic feedback "
                      f"calculated out to `hd_lmax = {hd_lmax}`. You must "
                      "either set `feedback=False` or use the baseline "
                      f"`hd_lmax = {self.lmaxs['hd']}`.")
            raise ValueError(errmsg)
        # get the file name:
        # for HD, if the `hd_lmax` is `None` or it's 20k, get the file name
        #  for the correct HD data version:
        if (spectrum_type in self.cmb_types) and (hd_lmax >= self.lmaxs['hd']):
            # then we can load the latest version of the data
            fname = self.hd_datalib.cmb_theory_fname(spectrum_type, baryonic_feedback=feedback)
        # otherwise, use the original data:
        else:
            feedback_info = '_hmcode2020_feedback' if feedback else ''
            lmin = self.lmins['hd']
            lmax = hd_lmax
            Lmax = hd_lmax # for HD, lmax and Lmax will be the same
            if spectrum_type in self.cmb_types:
                spec_info = f'{spectrum_type}_cls'
            else:
                spec_info = spectrum_type
            theo_dir = config.data_path('theory')
            fname = os.path.join(theo_dir, f'hd{feedback_info}_lmin{lmin}lmax{lmax}Lmax{Lmax}_{spec_info}.txt')
        return fname


    def cmb_theory_fname(self, exp, spectrum_type, hd_lmax=None, feedback=False):
        """Returns the name of the file containing the theory CMB and lensing
        spectra for a given CMB experiment and CMB type (e.g. delensed).
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        spectrum_type : str
            The name of the kind of spectra. Must be either `'lensed'`, 
            `'delensed'`, or `'unlensed'` for files containing the CMB TT, TE, 
            EE, and BB spectra along with the lensing (kappa kappa) spectrum;
            or `'clkk_res'` for files containing only the residual lensing power.
        hd_lmax : int, default=None
            Used for CMB-HD spectra that were calculated with a lower maximum 
            multipole than the baseline case.
        feedback : bool, default=False
            Used for CMB-HD only. If `True`, the file name returned will be for
            a file holding theory calculated with the HMCode2020 + baryonic
            feedback non-linear model, as opposed to the HMCode2016 CDM-only
            model.
        
        Returns
        -------
        fname : str
            The absolute path and name of the requested file.
    
        Raises
        ------
        ValueError 
            If the requested file does not exist.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored, or if 
        `hd_lmax < 20100` and you are  using a `hd_data_version` higher
        than the original `'v1.0'` for which the spectra were calculated
        to a lower maximum multipole.

        Note
        ----
        If `cmb_type = 'clkk_res'`, the file will contain a single column
        holding the residual CMB lensing power spectrum. Otherwise, the
        file will have a column for the multipoles of the spectra, the 
        CMB TT, TE, EE, and BB power spectra (in units of uK^2, without
        any multiplicative factors applied), and the lensing power spectrum,
        using the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where
        L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        # check the input
        exp = self.check_cmb_exp(exp)
        spectrum_type = spectrum_type.lower()
        valid_spec_types = self.cmb_types + ['clkk_res']
        if spectrum_type not in valid_spec_types:
            err_msg = f"Invalid spectrum type. You passed `spectrum_type = '{spectrum_type}'`, but `spectrum_type` must be one of {valid_spec_types}."
            raise ValueError(err_msg)
        if exp == 'hd':
            fname = self.hd_theory_fname(spectrum_type, hd_lmax=hd_lmax, feedback=feedback)
        else:
            if (hd_lmax is not None) or feedback:
                msg = f"Ignoring the `hd_lmax` and `feedback` arguments for `exp = '{exp}'`."
                warnings.warn(msg)
            lmin = self.lmins[exp]
            lmax = self.lmaxs[exp]
            Lmax = self.Lmaxs[exp]
            if spectrum_type in self.cmb_types:
                spec_info = f'{spectrum_type}_cls'
            else:
                spec_info = spectrum_type
            theo_dir = config.data_path('theory')
            fname = os.path.join(theo_dir, f'{exp}_lmin{lmin}lmax{lmax}Lmax{Lmax}_{spec_info}.txt')
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname


    def cmb_theory_fnames(self, exp, hd_lmax=None, feedback=False):
        """Returns the name of the files containing the lensed, delensed, 
        and unlensed theory CMB and lensing spectra, and the residual lensing 
        power spectrum (used for delensing), for a given CMB experiment.
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        hd_lmax : int, default=None
            Used for CMB-HD spectra that were calculated with a lower maximum 
            multipole than the baseline case.
        feedback : bool, default=False
            Used for CMB-HD only. If `True`, the file name returned will be for
            a file holding theory calculated with the HMCode2020 + baryonic
            feedback non-linear model, as opposted to the HMCode2016 CDM-only
            model.
    
        Returns
        -------
        fnames : dict of str
            A dictionary with the absolute path and names of the requested 
            files. The keys are `'lensed'`, `'delensed'`, or `'unlensed'` 
            for files containing the CMB TT, TE, EE, and BB spectra along
            with the lensing (kappa kappa) spectrum; and `'clkk_res'` for
            the file containing only the residual lensing power.
    
        Raises
        ------
        ValueError 
            If the experiment name is invalid.

        See also
        --------
        dataconfig.Data.cmb_theory_fname
        """
        exp = self.check_cmb_exp(exp)
        spectrum_types = self.cmb_types + ['clkk_res']
        fnames = {}
        for spectrum_type in spectrum_types:
            fnames[spectrum_type] = self.cmb_theory_fname(exp, spectrum_type, hd_lmax=hd_lmax, feedback=feedback)
        return fnames


    def cmb_noise_fname(self, exp, include_fg=True):
        """Returns the name of the file containing the power spectra of the
        noise on the CMB TT, TE, EE, and BB spectra.
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. You may also pass `exp = 'aso'` for an advanced SO-like 
            experiment. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, the temperature noise in the returned file is the sum of
            the instrumental noise and the residual extragalactic foreground
            power spectrum. If `False`, it will only contain instrumental noise.
            Used only when `exp = 'hd'`.

        Returns
        -------
        fname : str
            The name of the file holding the requested noise spectra.

        Raises
        ------
        ValueError 
            If the `exp` is invalid.

        Warns
        -----
        If the value of `include_fg` was changed from its default, but 
        will be ignored.

        Note
        ----
        The returned file will have a column for the multipoles of the spectra, 
        and columns for the CMB TT, TE, EE, and BB noise spectra (in units 
        of uK^2, without any multiplicative factors applied).
        """
        # check the input
        valid_exps = ['aso'] + self.cmb_exps
        exp = self.check_cmb_exp(exp, valid_exps=valid_exps)
        # for HD, get the correct version:
        if exp == 'hd':
            fname = self.hd_datalib.cmb_noise_fname(include_fg=include_fg)
        # otherwise, there is only one version:
        if exp in valid_exps[:-1]:
            noise_dir = config.data_path('noise')
            fname = os.path.join(noise_dir, f'{exp}_coaddf090f150_cmb_noise_cls_lmax5000.txt')
            if not include_fg:
                msg = f"Ignoring the `include_fg` argument for `exp = '{exp}'`."
                warnings.warn(msg)
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname


    def hd_lensing_noise_fname(self, include_fg=True, hd_Lmax=None):
        """Returns the name of the file containing the CMB-HD lensing noise.

        Parameters
        ----------
        include_fg : bool, default=True
            If `True`, return the file name for lensing noise that was 
            calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for lensing
            noise that was calculated by neglecting the effects. 
            `include_fg=False` is only available when `hd_Lmax=None` or 
            `hd_Lmax=20100`.
        hd_Lmax : int or None, default=None
            The maximum lensing multipole used in the calculation of the 
            noise, if lower than the baseline value of 20100. The allowed
            values are contained in the list `Data.hd_lmaxs`. Available 
            only for `include_fg=True`.

        Returns
        -------
        fname : str
            The name of the file holding the requested lensing noise spectrum.

        Raises
        ------
        ValueError 
            If the value of `hd_Lmax` is invalid, or if `hd_Lmax < 20100` 
            and `include_fg=False`.

        Warns
        -----
        If `include_fg=False` or `hd_Lmax < 20100` and you are using a 
        `hd_data_version` higher than the original `'v1.0'` for which the 
        noise was calculated without foregrounds or to a lower multipole.

        Note
        ----
        The returned file contains two columns: L, N_L^kk, where L is the
        CMB lensing multipole and N_L^kk is the noise on the CMB lensing
        power spectrum, C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where 
        C_L^phiphi is the CMB lensing potential power spectrum.
        """
        hd_Lmax = self.check_hd_lmax(hd_Lmax, hd_version_warning=True, data_info='lensing noise')
        # check if we have the file:
        if (not include_fg) and (hd_Lmax < self.Lmaxs['hd']):
            err_msg = ("There is no CMB lensing noise curve for CMB-HD with "
                       f"`include_fg = {include_fg}` and `hd_Lmax = {hd_Lmax}`"
                       "; you must either set `include_fg = True`, or use "
                       "the default `hd_Lmax = 20100`.")
            raise ValueError(err_msg)
        # get the file name:
        # for HD, if including FG and using hd_lmax = 20k, get the correct version:
        if include_fg and (hd_Lmax >= self.Lmaxs['hd']):
            # then we can load the latest version
            fname = self.hd_datalib.lensing_noise_fname()
        # otherwise, use the original version:
        else:
            if (not include_fg) and (self.hd_datalib.version_number > 1.0):
                # warn that we only have lensing noise w/o FG for v1.0:
                hd_version_info = f"`hd_data_version = '{self.hd_data_version}'`"
                if 'late' in self.hd_data_version.lower(): # include version number
                    hd_version_info = f"{hd_version_info} (version '{self.hd_datalib.version}')"
                msg = (f"You are using {hd_version_info}, and passed "
                        "`include_fg = False` for the lensing noise. The "
                        "CMB-HD lensing noise was only calculated without "
                        "foregrounds for version `'v1.0'`, so that version "
                        "will be returned.")
                warn(msg)
            extra_info = '' if include_fg else '_nofg'
            lmin = self.lmins['hd']
            lmax = hd_Lmax # for HD, lmax and Lmax will be the same
            Lmax = hd_Lmax
            fname = os.path.join(config.data_path('noise'), f'hd{extra_info}_lmin{lmin}lmax{lmax}Lmax{Lmax}_nlkk.txt')
        return fname


    def cmb_lensing_noise_fname(self, exp, include_fg=True, hd_Lmax=None):
        """Returns the name of the file containing the CMB lensing noise 
        spectrum for the given CMB experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, return the file name for lensing noise that was 
            calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for lensing
            noise that was calculated by neglecting the effects. 
            Used only when `exp='hd'` and `hd_Lmax=None` or `hd_Lmax=20100`.
        hd_Lmax : int or None, default=None
            The maximum lensing multipole used in the calculation of the 
            noise, if lower than the baseline value of 20100. The allowed
            values are contained in the list `Data.hd_lmaxs`. Used only
            when `exp = 'hd'` and `include_fg=True`.

        Returns
        -------
        fname : str
            The name of the file holding the requested lensing noise spectrum.

        Raises
        ------
        ValueError 
            If either the `exp` or value of `hd_Lmax` are invalid, or if 
            `hd_Lmax < 20100` and `include_fg=False`.

        Warns
        -----
        If the values of `include_fg` or `hd_Lmax` were provided, but will
        be ignored.

        Note
        ----
        The returned file contains two columns: L, N_L^kk, where L is the
        CMB lensing multipole and N_L^kk is the noise on the CMB lensing
        power spectrum, C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where 
        C_L^phiphi is the CMB lensing potential power spectrum.
        """
        # check the input
        exp = self.check_cmb_exp(exp)
        if exp == 'hd':
            fname = self.hd_lensing_noise_fname(include_fg=include_fg, hd_Lmax=hd_Lmax)
        else:
            if (hd_Lmax is not None) or (not include_fg):
                msg = f"Ignoring the `hd_Lmax` and `include_fg` arguments for `exp = '{exp}'`."
                warnings.warn(msg)
            lmin = self.lmins[exp]
            lmax = self.lmaxs[exp]
            Lmax = self.Lmaxs[exp]
            fname = os.path.join(config.data_path('noise'), f'{exp}_lmin{lmin}lmax{lmax}Lmax{Lmax}_nlkk.txt')
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname

    
    def hd_fg_fname(self, frequency='coadd'):
        """Returns the name of the file containing the residual extragalactic 
        foreground power spectra (or a single coadded spectrum) for CMB-HD.

        Parameters
        ----------
        frequency : str or int, default='coadd'
            If `'coadd'`, the file will contain the coadded foreground
            power spectrum for the combination of 90 and 150 GHz. Otherwise,
            pass `90` or `'f090'` for a file containing columns for the
            different foreground components at 90 GHz, or pass `150` or 
            `'f150'` for the corresponding file at 150 GHz.

        Returns
        -------
        fname : str
            The file name (including its absolute path).

        Raises
        ------
        ValueError
            If an invalid `frequency` was passed.
        """
        freq = str(frequency).lower()
        if 'coadd' in freq:
            fname = self.hd_datalib.coadded_fg_spectrum_fname()
        elif ('90' in freq) or ('150' in freq):
            fname = self.hd_datalib.fg_spectra_fname(freq)
        else:
            err_msg = f"Invalid `frequency`. You passed `frequency = '{freq}'`; valid choices are `'coadd'`, `'f090'`, or `'f150'`."
            raise ValueError(err_msg)
        return fname


    def hd_covmat_fname(self, cmb_type='delensed', include_fg=True, 
            hd_lmax=None):
        """Returns the name of the file holding the covariance matrix for the
        mock CMB-HD TT, TE, EE, BB and CMB lensing power spectra for the  
        the given CMB type (lensed or delensed).
        
        Parameters
        ----------
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a covariance matrix for
            delensed CMB TT, TE, EE, and BB power spectra, in addition to the
            CMB lensing spectrum. If `cmb_type='lensed'`, the covariance matrix
            is for lensed CMB spectra instead, but otherwise includes the same
            set of power spectra as the delensed case. Note that passing 
            `cmb_type='lensed'` is only an option when `hd_lmax` is `None`
            or if `hd_lmax = 20100`.
        include_fg : bool, default=True
            If `True`, return the file name for CMB-HD covariance matrix that 
            was calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for the covariance 
            matrix that was calculated by neglecting these effects. 
            `include_fg=False` is only possible when `cmb_type='lensed'`.
            Note that there is only a lensed covariance matrix calculated
            without foregrounds for the original HD data version, `'v1.0'`.
        hd_lmax : int, default=None
            Used to return CMB-HD covariance matrices that were calculated with 
            a lower maximum multipole than the baseline case. Only used when
            `cmb_type='delensed'`. Note that the covariance matrices with
            `hd_lmax` < 20100 were only calculated for the original HD data 
            version, `'v1.0'`.

        Returns
        -------
        fname : str
            The name of the file that contains the requested covariance matrix.
        
        Raises
        ------
        ValueError 
            If either the value of `hd_lmax` is invalid, or if the requested 
            covariance matrix does not exist.

        Warns
        -----
        If you are using a `hd_data_version` higher than the original `'v1.0'`,
        but the requested covariance matrix was only calculated for `'v1.0'`.
        """
        # get the multipole ranges:
        hd_version_warning = True if (cmb_type == 'delensed') else False
        hd_lmax = self.check_hd_lmax(hd_lmax, hd_version_warning=hd_version_warning, data_info=f'{cmb_type} covariance matrix')
        lmax = hd_lmax
        lmaxTT = hd_lmax # for HD, lmax and TT lmax will be the same
        Lmax = hd_lmax # for HD, lmax and Lmax will be the same
        lmin = self.lmins['hd']
        # check if we have the requested covmat:
        if cmb_type not in ['lensed', 'delensed']:
            err_msg = "Invalid `cmb_type`: the options for CMB-HD are either `'lensed'` or `'delensed'`."
            raise ValueError(err_msg)
        if (cmb_type == 'lensed') and (lmax < self.lmaxs['hd']):
            err_msg = f"No covariance matrix for lensed HD data with lmax < {self.lmaxs['hd']}."
            raise ValueError(err_msg)
        if (cmb_type == 'delensed') and (not include_fg):
            err_msg = "No covariance matrix for CMB-HD delensed data without foregrounds; you must use `cmb_type = 'lensed'` if you don't want to include foregrounds."
            raise ValueError(err_msg)
        # warn the user if we have the covmat, but not for their requested HD data version:
        #  (we already did some of this above when setting the `hd_lmax` value)
        if (self.hd_datalib.version_number > 1.0) and (not include_fg):
            hd_version_info = f"`hd_data_version = '{self.hd_data_version}'`"
            if 'late' in self.hd_data_version.lower(): # include version number
                hd_version_info = f"{hd_version_info} (version '{self.hd_datalib.version}')"
            msg = (f"You are using {hd_version_info}, and passed "
                    "`include_fg = False` for the lensed covariance matrix. "
                    "The CMB-HD lensed covariance matrix was only calculated "
                    "without foregrounds for version `'v1.0'`, so that "
                    "version will be returned.")
            warn(msg)
        # finally, get the file name:
        # if including FG and `hd_lmax` = 20k, get the correct version:
        if include_fg and (hd_lmax >= self.lmaxs['hd']):
            fname = self.hd_datalib.block_covmat_fname(cmb_type)
        # otherwise, get the original file:
        else:
            extra_info = '' if include_fg else '_nofg'
            ell_info = f'lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}'
            fname = os.path.join(config.data_path('covmats'), f'hd{extra_info}_fsky0pt6_{ell_info}_binned_{cmb_type}_cov.txt')
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname


    def cmb_covmat_fname(self, exp, cmb_type='delensed', include_fg=True, 
            hd_lmax=None):
        """Returns the name of the file holding the covariance matrix for the
        mock CMB TT, TE, EE, BB and CMB lensing power spectra corresponding to 
        the given experimental configuration and CMB type (lensed or delensed).
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a covariance matrix for
            delensed CMB TT, TE, EE, and BB power spectra, in addition to the
            CMB lensing spectrum. If `cmb_type='lensed'`, the covariance matrix
            is for lensed CMB spectra instead, but otherwise includes the same
            set of power spectra as the delensed case. Note that passing 
            `cmb_type='lensed'` is only an option for `exp='hd'` when `hd_lmax` 
            is `None` or if `hd_lmax = 20100`.
        include_fg : bool, default=True
            If `True`, return the file name for CMB-HD covariance matrix that 
            was calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for the covariance 
            matrix that was calculated by neglecting these effects. 
            Used only when `exp = 'hd'`; `include_fg=False` is only possible
            when `cmb_type='lensed'`. Note that there is only a lensed CMB-HD 
            covariance matrix calculated without foregrounds for the original 
            HD data version, `'v1.0'`.
        hd_lmax : int, default=None
            Used to return CMB-HD covariance matrices that were calculated with 
            a lower maximum multipole than the baseline case. Only used when
            `exp='hd'` and `cmb_type='delensed'`. Note that the covariance
            matrices with `hd_lmax` < 20100 were only calculated for the 
            original HD data version, `'v1.0'`.

        Returns
        -------
        fname : str
            The name of the file that contains the requested covariance matrix.
        
        Raises
        ------
        ValueError 
            If either the `exp` or value of `hd_lmax` are invalid, or if the
            requested covariance matrix does not exist.

        Warns
        -----
        If the values of `include_fg` or `hd_lmax` were provided, but will
        be ignored; or if `exp='hd'` and you are using a `hd_data_version` 
        higher than the original `'v1.0'`, but the requested covariance matrix 
        was only calculated for `'v1.0'`.
        """
        # check the input
        exp = self.check_cmb_exp(exp)
        cmb_type = cmb_type.lower()
        if exp == 'hd':
            fname = self.hd_covmat_fname(cmb_type=cmb_type, include_fg=include_fg, hd_lmax=hd_lmax)
        else:
            if (hd_lmax is not None) or (not include_fg):
                msg = f"Ignoring the `hd_lmax` and `include_fg` arguments for `exp = '{exp}'`."
                warnings.warn(msg)
            # check if we have the requested covmat:
            if cmb_type != 'delensed':
                err_msg = f"Invalid `cmb_type`. You must pass `cmb_type = 'delensed'` for `exp = '{exp}'`."
                raise ValueError(err_msg)
            # get info for the file name:
            lmin = self.lmins[exp]
            lmax = self.lmaxs[exp]
            lmaxTT = self.lmaxsTT[exp]
            Lmax = self.Lmaxs[exp]
            # get the file name
            ell_info = f'lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}'
            fname = os.path.join(config.data_path('covmats'), f'{exp}_fsky0pt6_{ell_info}_binned_{cmb_type}_cov.txt')
        if not os.path.exists(fname):
            msg = f"The requested file {fname} does not exist."
            warnings.warn(msg)
        return fname


    def desi_theory_fname(self):
        """Returns the name of the file containing the theoretical BAO 
        measurement r_s/d_V(z) for mock DESI BAO. The first column of
        the file contains the redshift z, and the second contains the
        quantity r_s/d_V evaluated at that redshift.
        """
        return os.path.join(config.data_path('bao'), 'mock_desi_bao_rs_over_DV_fid_data.txt')


    def desi_covmat_fname(self):
        """Returns the name of the covariance matrix for the mock DESI BAO
        measurements r_s/d_V(z)."""
        return os.path.join(config.data_path('bao'), 'mock_desi_bao_rs_over_DV_fid_cov.txt')


    def precomputed_desi_fisher_fname(self, use_H0=False):
        """Returns the name of a file holding a Fisher matrix calculated
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
        fname = os.path.join(config.data_path('fisher_matrices'), f'desi_bao{H0_info}_fisher.txt')
        return fname


    def precomputed_cmb_fisher_fname(self, exp, cmb_type='delensed', 
            use_H0=False, with_desi=False, hd_lmax=None, include_fg=True, 
            feedback=False):
        """Returns the name of the file containing a Fisher matrix calculated
        for the given experiment and kind of CMB spectra (lensed or delensed).
        The parameters in the Fisher matrix are the six LCDM parameters, the
        effective number of relativistic species, and the sum of the neutrino
        masses. Note that a Gaussian prior of width 0.007 has already been 
        applied. 

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a Fisher matrix calculated 
            from delensed CMB TT, TE, EE, and BB power spectra, in addition to 
            the CMB lensing spectrum. If `cmb_type='lensed'`, the Fisher matrix
            was computed with lensed CMB spectra instead, as well as the CMB
            lensing spectrum. Note that passing `cmb_type='lensed'` is only an
            option for `exp='hd'`, `hd_lmax=None`, and `feedback=False`.
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the angular 
            scale of the sound horizon at last scattering (multiplied by 100)
            is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB spectra.
            If `True`, the Fisher matrix is the sum of a CMB and a mock DESI BAO
            Fisher matrix.
        include_fg : bool, default=True
            If `True`, return the file name for CMB-HD Fisher matrix that 
            was calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for the Fisher 
            matrix that was calculated by neglecting these effects. 
            Used only when `exp = 'hd'`; `include_fg=False` is only possible
            when `cmb_type='lensed'`.
        hd_lmax : int, default=None
            Used to return CMB-HD Fisher matrices that were calculated with 
            a lower maximum multipole than the baseline case. Only used when
            `exp='hd'`.
        feedback : bool, default=False
            If `True`, the Fisher matrix was calculated with the HMCode2020 
            + baryonic feedback non-linear model, and also contains the 
            baryonic feedback parameter of this model (i.e., a total of 9
            parameters). If `False`, the Fisher matrix was calculated with the 
            HMCode2016 CDM-only model. Used only when `exp='hd'`, for 
            `cmb_type='delensed'` and `hd_lmax=None`.

        Returns
        -------
        fname : str
            The requested file name.

        Raises
        ------
        ValueError
            If the `exp`, `cmb_type`, or `hd_lmax` value is invalid, or if 
            the requested Fisher matrix does not exist.

        Warns
        -----
        If the `hd_lmax`, `include_fg`, or `feedback` arguments were changed
        from their default value, but will be ignored. Or, if `exp='hd'` and
        the requested `hd_data_version` is higher than the original `'v1.0'`
        for which these Fisher matrices were computed.
        """
        exp = self.check_cmb_exp(exp)
        cmb_type = cmb_type.lower()
        if (self.hd_datalib.version_number > 1.0) and (exp == 'hd'):
            hd_version_info = f"`hd_data_version = '{self.hd_data_version}'`"
            if 'late' in self.hd_data_version.lower(): # include version number
                hd_version_info = f"{hd_version_info} (version '{self.hd_datalib.version}')"
            msg = (f"You are using {hd_version_info}, but the Fisher matrices "
                    "were only saved for version `'v1.0'`, so that version "
                    "will be returned. You can calculate new Fisher matrices "
                    "with the latest HD data using the `Fisher` class in "
                    "the `fisher` module of `hdfisher`.")
            warn(msg)
        if (exp != 'hd') and (((hd_lmax is not None) or (not include_fg)) or feedback):
            msg = f"Ignoring the `hd_lmax`, `include_fg`, and `feedback` arguments for `exp = '{exp}'`."
            warnings.warn(msg)
        # we only have pre-computed Fisher matrices from lensed power spectra 
        #  for CMB-HD with lmax, Lmax = 20100
        has_lensed = False 
        if (exp == 'hd') and (not feedback):
            if hd_lmax is None:
                has_lensed = True
            elif int(hd_lmax) == self.lmaxs['hd']:
                has_lensed = True
        if (cmb_type == 'lensed') and (not has_lensed):
            err_msg = f"There are no precomputed Fisher matrices from lensed theory for `exp = '{exp}'`, `feedback = {feedback}`, and `hd_lmax = {hd_lmax}`."
            raise ValueError(err_msg)
        # we only have Fisher matrices excluding foregrounds for HD lensed data
        if (exp == 'hd') and ((cmb_type == 'delensed') and (not include_fg)):
            err_msg = "There are no precomputed Fisher matrices for mock delensed CMB-HD data without foregrounds."
            raise ValueError(err_msg)
        # make sure the `cmb_type` is valid in general (e.g., no unlensed)
        if cmb_type not in self.cmb_types[:-1]:
            err_msg = f"Invalid `cmb_type`: '{cmb_type}'. The `cmb_type` must be one of: {self.cmb_types[:-1]}."
            raise ValueError(err_msg)
        # get the file name
        H0_info = '_useH0' if use_H0 else ''
        desi_info = '_desi_bao' if with_desi else ''
        if (exp == 'hd') and (not include_fg):
            fg_info = '_nofg'
        else:
            fg_info = ''
        if (exp == 'hd') and feedback:
            feedback_info = '_feedback'
        else:
            feedback_info = ''
        lmin = self.lmins[exp]
        if (exp == 'hd') and (hd_lmax is not None):
            hd_lmax = self.check_hd_lmax(hd_lmax)
            lmax = hd_lmax
            lmaxTT = lmax
            Lmax = lmax
        else:
            lmax = self.lmaxs[exp]
            lmaxTT = self.lmaxsTT[exp]
            Lmax = self.Lmaxs[exp]
        fname_root = f'{exp}{fg_info}_fsky0pt6_lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}_{cmb_type}{desi_info}{feedback_info}{H0_info}'
        fisher_dir = config.data_path(f'fisher_matrices')
        fname = os.path.join(fisher_dir, f'{fname_root}_fisher.txt')
        return fname
        


    # ----- functions that load the data: -----


    def load_example_hd_fisher(self, cmb_type='delensed', use_H0=False, with_desi=False):
        """Returns an example CMB-HD Fisher matrix that was calculated with
        the correct `hd_data_version`, and a list of the parameters it 
        contains. All Fisher matrices contain 8 parameters (LCDM + N_eff 
        + sum m_nu) and all have a Gaussian prior of sigma(tau) = 0.007 applied.

        Parameters
        ----------
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a Fisher matrix calculated 
            from delensed CMB TT, TE, EE, and BB power spectra, in addition to 
            the CMB lensing spectrum. If `cmb_type='lensed'`, the Fisher matrix
            was computed with lensed CMB spectra instead, as well as the CMB
            lensing spectrum. 
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the angular 
            scale of the sound horizon at last scattering (multiplied by 100)
            is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB spectra.
            If `True`, the Fisher matrix is the sum of a CMB and a mock DESI BAO
            Fisher matrix.

        Returns
        -------
        fisher_matrix : array_like of float
            An array of shape `(8,8)` holding the elements of the Fisher matrix.
        fisher_params : list of str
            A list of parameter names for the parameters in the Fisher matrix,
            in the same order as their corresponding rows/columns.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed.
        """
        fname = self.example_hd_fisher_fname(cmb_type=cmb_type, use_H0=use_H0, with_desi=with_desi)
        fisher_matrix, fisher_params = utils.load_fisher_matrix(fname)
        return fisher_matrix, fisher_params


    def load_cmb_theory_spectra(self, exp, cmb_type, output_lmax=None, 
            hd_lmax=None, feedback=False):
        """Returns a dictionary containing the theory CMB and lensing
        power spectra, and the corresponding multipoles, for a given CMB 
        experiment and CMB type (e.g. delensed).
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        cmb_type : str
            The name of the kind of spectra. Must be either `'lensed'`, 
            `'delensed'`, or `'unlensed'`. 
        output_lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_lmax` value. Otherwise, use the maximum multipole for 
            the given `exp`.
        hd_lmax : int or None, default=None
            If a value of `hd_lmax` is provided, returns spectra that were 
            calculated with a lower maximum multipole than the baseline case.
            Only used if `exp='hd'`. If not `None`, `hd_lmax` must be one of 
            the values in `Data.hd_lmaxs`.
        feedback : bool, default=False
            If `True`, the power spectra were calculated with the HMCode2020 
            + baryonic feedback non-linear model. Otherwise, they were 
            calculated with the HMCode2016 CDM-only model. Only used if 
            `exp='hd'`.
        
        Returns
        -------
        theo : dict of array_like of float
            A dictionary with a key `'ells'` holding the multipoles for the
            power spectra; keys `'tt'`, `'te'`, `'ee'`, and `'bb'` for the
            CMB power spectra for the requested `cmb_type`; and a key`'kk'`
            for the CMB lensing spectrum.
    
        Raises
        ------
        ValueError 
            If the `exp`, `cmb_type`, or `hd_lmax` value is invalid.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored, or if 
        `hd_lmax < 20100` and you are  using a `hd_data_version` higher
        than the original `'v1.0'` for which the spectra were calculated
        to a lower maximum multipole.

        Note
        ----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, 
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.

        See also
        --------
        dataconfig.Data.load_all_cmb_theory_spectra
        dataconfig.Data.cmb_theory_fname
        """
        exp = self.check_cmb_exp(exp)
        cmb_type = cmb_type.lower()
        if cmb_type not in self.cmb_types:
            if cmb_type == 'clkk_res':
                err_msg = "Use the `load_residual_cmb_lensing_spectrum` method of the `Data` class to load the resudial lensing power spectrum."
                raise ValueError(err_msg)
            else:
                err_msg = f"Invalid `cmb_type`. You passed `cmb_type = '{cmb_type}'`; valid options are {self.cmb_types}."
                raise ValueError(err_msg)
        # get the file name and load in the spectra
        fname = self.cmb_theory_fname(exp, cmb_type, hd_lmax=hd_lmax, feedback=feedback)
        theo = utils.load_from_file(fname, self.theo_cols)
        if output_lmax is not None:
            lmax = int(output_lmax)
            theo_lmax = int(theo['ells'][-1])
            if lmax > theo_lmax:
                msg = f"You requested theory power spectra out to `output_lmax = {output_lmax}`, but the spectra were only computed out to {theo_lmax}."
                warnings.warn(msg)
        else:
            lmax = max([self.lmaxs[exp], self.Lmaxs[exp], self.lmaxsTT[exp]])
        for key in theo.keys():
            theo[key] = theo[key][:lmax+1]
        return theo
    
    
    def load_all_cmb_theory_spectra(self, exp, output_lmax=None, hd_lmax=None, 
            feedback=False):
        """Returns a nested dictionary containing the lensed, delensed, and 
        unlensed theory CMB and lensing power spectra, and the corresponding 
        multipoles, for a given CMB experiment.
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        output_lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_lmax` value.
        hd_lmax : int or None, default=None
            If a value of `hd_lmax` is provided, returns spectra that were 
            calculated with a lower maximum multipole than the baseline case.
            Only used if `exp='hd'`. If not `None`, `hd_lmax` must be one of 
            the values in `Data.hd_lmaxs`.
        feedback : bool, default=False
            If `True`, the power spectra were calculated with the HMCode2020 
            + baryonic feedback non-linear model. Otherwise, they were 
            calculated with the HMCode2016 CDM-only model. Only used if 
            `exp='hd'`.
        
        Returns
        -------
        theo : dict of dict of array_like of float
            A nested dictionary whose first set of keys are `'lensed'`, 
            `'delensed'`, and `'unlensed'`. The value for each key is a 
            dictionary with a key `'ells'` whose value is a one-dimensional 
            array holding the multipoles for the power spectra; keys `'tt'`, 
            `'te'`, `'ee'`, and `'bb'` for one-dimensional arrays holding the 
            corresponding CMB power spectra for that `cmb_type`; and a key`'kk'`
            for a one-dimensional array holding the CMB lensing spectrum.
    
        Raises
        ------
        ValueError 
            If the `exp` or `hd_lmax` value is invalid.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored.

        Note
        ----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, 
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.

        See also
        --------
        dataconfig.Data.load_cmb_theory_spectra
        dataconfig.Data.cmb_theory_fname
        """
        theo = {}
        for cmb_type in self.cmb_types:
            theo[cmb_type] = self.load_cmb_theory_spectra(exp, cmb_type, output_lmax=output_lmax, hd_lmax=hd_lmax, feedback=feedback)
        return theo


    def load_residual_cmb_lensing_spectrum(self, exp, output_Lmax=None, 
            hd_Lmax=None, feedback=False): 
        """Returns the residual CMB lensing power spectrum, i.e. the difference
        between the total lensing power spectrum and the Wiener-filtered 
        lensing power spectrum, and the corresponding lensing multipoles.

        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        output_Lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_Lmax` value.
        hd_Lmax : int or None, default=None
            If a value of `hd_Lmax` is provided, returns the spectrum that was 
            calculated with a lower maximum multipole than the baseline case.
            Only used if `exp='hd'`. If not `None`, `hd_Lmax` must be one of 
            the values in `Data.hd_lmaxs`.
        feedback : bool, default=False
            If `True`, the power spectra were calculated with the HMCode2020 
            + baryonic feedback non-linear model. Otherwise, they were 
            calculated with the HMCode2016 CDM-only model. Only used if 
            `exp='hd'`.
        
        Returns
        -------
        L, clkk_res : array_like of float
            One-dimensional arrays containing the lensing multipoles (`L`) 
            and the residual lensing power spectrum (`clkk_res`).
    
        Raises
        ------
        ValueError 
            If the `exp` or the `hd_Lmax` value is invalid.

        Warns
        -----
        If the `hd_Lmax` and `feedback` arguments will be ignored.

        Note
        ----
        For the residual CMB lensing power spectrum, we use the convention
        C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where L is the lensing multipole
        and C_L^phiphi is the CMB lensing potential power spectrum.

        See also
        --------
        dataconfig.Data.cmb_theory_fname
        """
        exp = self.check_cmb_exp(exp)
        if exp == 'hd':
            hd_Lmax = self.check_hd_lmax(hd_Lmax)
        if (exp == 'hd') and (hd_Lmax >= self.Lmaxs['hd']): # calculate it using the correct version
            _, nlkk = self.hd_datalib.lensing_noise_spectrum()
            hd_theo = self.load_cmb_theory_spectra(exp, 'lensed')
            clkk = hd_theo['kk']
            Lmin = self.lmins['hd']
            Lmax = self.Lmaxs['hd']
            clkk_res = theory.get_residual_lensing(clkk, nlkk, Lmin, Lmax, len(clkk)-1)
        else:
            fname = self.cmb_theory_fname(exp, 'clkk_res', hd_lmax=hd_Lmax, feedback=feedback)
            clkk_res = np.loadtxt(fname)
        L = np.arange(len(clkk_res))
        if output_Lmax is not None:
            Lmax = int(output_Lmax)
            theo_Lmax = len(clkk_res) - 1
            if Lmax > theo_Lmax:
                msg = f"You requested the residual lensing power spectrum out to `output_Lmax = {output_Lmax}`, but it was only computed out to {theo_Lmax}."
                warnings.warn(msg)
        else:
            Lmax = self.Lmaxs[exp]
        clkk_res = clkk_res[:Lmax+1]
        L = L[:Lmax+1]
        return L, clkk_res


    def load_cmb_noise_spectra(self, exp, include_fg=True, output_lmax=None):
        """Returns a dictionary containing the power spectra of the noise on 
        the CMB TT, TE, EE, and BB spectra, and the corresponding multipoles.
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. You may also pass `exp = 'aso'` for an advanced SO-like 
            experiment. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, the temperature noise power spectrum is the sum of
            the instrumental noise and the residual extragalactic foreground
            power spectrum. If `False`, it will only contain instrumental noise.
            Used only when `exp = 'hd'`.
        output_lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_lmax` value. Otherwise use the maximum multipole for 
            the given `exp`.

        Returns
        -------
        noise : dict of array_like of float
            A dictionary with a key `'ells'` whose value is a one-dimensional 
            array holding the multipoles for the noise spectra, and keys `'tt'`,
            `'te'`, `'ee'`, and `'bb'` for one-dimensional arrays holding the
            corresponding noise power spectra.

        Raises
        ------
        ValueError 
            If the `exp` is invalid.

        Warns
        -----
        If the value of `include_fg` was changed from its default, but 
        will be ignored.

        Note
        ----
        The noise spectra are in units of uK^2, without any multiplicative
        factors applied.

        See also
        --------
        dataconfig.Data.cmb_noise_fname
        """
        exp = self.check_cmb_exp(exp)
        fname = self.cmb_noise_fname(exp, include_fg=include_fg)
        noise = utils.load_from_file(fname, self.noise_cols)
        if output_lmax is not None:
            lmax = int(output_lmax)
            noise_lmax = int(noise['ells'][-1])
            if lmax > noise_lmax:
                msg = f"You requested noise power spectra out to `output_lmax = {output_lmax}`, but the spectra were only computed out to {noise_lmax}."
                warnings.warn(msg)
        else: # automatically trim to be consistent with theory spectra
            lmax = max([self.lmaxs[exp], self.Lmaxs[exp], self.lmaxsTT[exp]])
        for key in noise.keys():
            noise[key] = noise[key][:lmax+1]
        return noise


    def load_cmb_lensing_noise_spectrum(self, exp, include_fg=True, 
            output_Lmax=None, hd_Lmax=None):
        """Returns the CMB lensing noise spectrum and the corresponding
        lensing multipoles for the given CMB experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, returns the lensing noise that was calculated including 
            the effects of residual extragalactic foregrounds. If `False`, 
            returns the lensing noise that was calculated by neglecting these 
            effects. Used only when `exp = 'hd'`.
        output_Lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_Lmax` value.
        hd_Lmax : int or None, default=None
            The maximum lensing multipole used in the calculation of the 
            noise, if lower than the baseline value of 20100. The allowed
            values are contained in the list `Data.hd_lmaxs`. Used only
            when `exp = 'hd'`.

        Returns
        -------
        L, nlkk : array_like of float
            One-dimensional arrays containing the lensing multipoles (`L`)
            and the lensing noise spectrum (`nlkk`).

        Raises
        ------
        ValueError 
            If either the `exp` or value of `hd_Lmax` are invalid.

        Warns
        -----
        If the values of `include_fg` or `hd_Lmax` were provided, but will
        be ignored.

        Note
        ----
        The CMB lensing noise N_L^kk is the noise on the CMB lensing spectrum
        C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where C_L^phiphi is the CMB
        lensing potential power spectrum and L is the lensing multipole.

        See also
        --------
        dataconfig.Data.cmb_lensing_noise_fname
        """
        exp = self.check_cmb_exp(exp)
        fname = self.cmb_lensing_noise_fname(exp, include_fg=include_fg, hd_Lmax=hd_Lmax)
        L, nlkk = np.loadtxt(fname, unpack=True)
        if output_Lmax is not None:
            Lmax = int(output_Lmax)
            theo_Lmax = int(L[-1])
            if Lmax > theo_Lmax:
                msg = f"You requested the lensing noise power spectrum out to `output_Lmax = {output_Lmax}`, but it was only computed out to {theo_Lmax}."
                warnings.warn(msg)
        else:
            Lmax = self.Lmaxs[exp] 
        nlkk = nlkk[:Lmax+1]
        L = L[:Lmax+1]
        return L, nlkk

   
    def load_hd_fg_spectra(self, frequency, output_lmax=None):
        """Returns a dictionary holding power spectra of residual extragalactic 
        foregrounds in temperature at the given frequency for CMB-HD.

        Parameters
        ----------
        frequency : str or int
            The frequency for the foreground power spectra. Pass `90` or 
            `'f090'` for the foreground components at 90 GHz, or pass `150` 
            or  `'f150'` for the foregrounds at 150 GHz.
        output_lmax : int or None, default=None
            If provided, cut the spectra at a maximum multipole given by the
            `output_lmax` value.

        Returns
        -------
        fgs : dict of array_like of float
            A dictionary of one-dimensional arrays with a key `'ells'` holding
            the multipoles of the power spectra, and keys `'ksz'`, `'tsz'`, 
            `'cib'`, and `'radio'` holding the residual foreground power 
            spectra for reionization kSZ, tSZ, CIB, and radio sources, 
            respectively.

        Raises
        ------
        ValueError
            If an invalid `frequency` was passed.

        Note
        ----
        The power spectra are in units of uK^2, without any multiplicative
        factors applied.

        See also
        --------
        dataconfig.Data.hd_fg_fname
        dataconfig.Data.load_hd_coadd_fg_spectrum
        """
        freq = str(frequency)
        if ('90' not in freq) and ('150' not in freq):
            if 'coadd' in freq:
                err_msg = "Use the `load_hd_coadd_fg_spectrum` method of the `Data` class to load the total coadded foreground power spectrum."
                raise ValueError(err_msg)
            else:
                err_msg = f"Invalid `frequency`: you passed `frequency = '{frequency}'`; valid options (of type `str` or `int`) are `90` and `150`."
                raise ValueError(err_msg)
        fname = self.hd_fg_fname(frequency=frequency)
        fgs = utils.load_from_file(fname, self.fg_cols)
        if output_lmax is not None:
            lmax = int(output_lmax)
            fg_lmax = int(fgs['ells'][-1])
            if lmax > fg_lmax:
                msg = f"You requested foreground power spectra out to `output_lmax = {output_lmax}`, but the spectra were only computed out to {fg_lmax}."
                warnings.warn(msg)
        else: # automatically trim to be consistent with theory spectra
            lmax =  max([self.lmaxs['hd'], self.Lmaxs['hd'], self.lmaxsTT['hd']]) 
        for key in fgs.keys():
            fgs[key] = fgs[key][:lmax+1]
        return fgs


    def load_hd_coadd_fg_spectrum(self, output_lmax=None):
        """Returns the power spectrum of the residual extragalactic foregrounds
        in temperature for CMB-HD, coadded from 90 and 150 GHz. 

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

        Note
        ----
        The power spectrum is in units of uK^2, without any multiplicative
        factors applied.

        See also
        --------
        dataconfig.Data.hd_fg_fname
        dataconfig.Data.load_hd_fg_spectra
        """
        fname = self.hd_fg_fname(frequency='coadd')
        ells, coadd_fg_cls = np.loadtxt(fname, unpack=True)
        if output_lmax is not None:
            lmax = int(output_lmax)
            fg_lmax = int(ells[-1])
            if lmax > fg_lmax:
                msg = f"You requested the coadded foreground power spectrum out to `output_lmax = {output_lmax}`, but it was only computed out to {fg_lmax}."
                warnings.warn(msg)
        else:
            lmax =  max([self.lmaxs['hd'], self.Lmaxs['hd'], self.lmaxsTT['hd']]) 
        ells = ells[:lmax+1]
        coadd_fg_cls = coadd_fg_cls[:lmax+1]
        return ells, coadd_fg_cls


    def load_cmb_covmat(self, exp, cmb_type='delensed', include_fg=True, 
            hd_lmax=None):
        """Returns the covariance matrix for the mock CMB TT, TE, EE, BB and 
        CMB lensing power spectra corresponding to the given experimental 
        configuration and CMB type (lensed or delensed).
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, returns a covariance matrix for delensed 
            CMB TT, TE, EE, and BB power spectra, in addition to the CMB 
            lensing spectrum. If `cmb_type='lensed'`, the covariance matrix is 
            for lensed CMB spectra instead, but otherwise includes the same
            set of power spectra as the delensed case. Note that passing 
            `cmb_type='lensed'` is only an option for `exp='hd'` and  
            `hd_lmax=None`.
        include_fg : bool, default=True
            If `True`, returns the CMB-HD covariance matrix that was calculated
            including the effects of residual extragalactic foregrounds. If 
            `False`, returns the covariance matrix that was calculated by 
            neglecting these effects. Used only when `exp = 'hd'`; 
            `include_fg=False` is only possible when `cmb_type='lensed'`.
        hd_lmax : int, default=None
            Used to return CMB-HD covariance matrices that were calculated with 
            a lower maximum multipole than the baseline case. Only used when
            `exp='hd'` and `cmb_type='delensed'`.

        Returns
        -------
        covmat : array_like of float
            A two-dimensional holding the full covariance matrix for the
            mock CMB power spectra.
        
        Raises
        ------
        ValueError 
            If either the `exp` or value of `hd_lmax` are invalid, or if the
            requested covariance matrix does not exist.

        Warns
        -----
        If the values of `include_fg` or `hd_lmax` were provided, but will
        be ignored.

        Note
        ----
        The covariance matrix is binned and contains 25 blocks; each block
        has shape `(nbin, nbin)`, where `nbin` is the number of bins in the
        multipole range for the given experiment. The diagonal blocks contain
        the covariance matrices for TT x TT, TE x TE, EE x EE, BB x BB, and 
        kk x kk, where kk refers to the CMB lensing spectrum. The off-diagonal
        blocks contain the cross-covariances, e.g. TT x TE, TT x EE, etc. 
        We use units of  uK^2 for the CMB spectra, and do not apply any 
        multiplicative factors. For the CMB lensing spectrum, we use the 
        convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where C_L^phiphi is 
        the CMB lensing potential power spectrum and L is the lensing multipole.

        See also
        --------
        dataconfig.Data.cmb_covmat_fname
        """
        fname = self.cmb_covmat_fname(exp, cmb_type=cmb_type, include_fg=include_fg, hd_lmax=hd_lmax)
        covmat = np.loadtxt(fname)
        return covmat


    def load_desi_theory(self):
        """Returns one-dimensional arrays containing the redshifts (`z`) and
        the theoretical BAO measurement r_s/d_V(z)  at those redshifts (`rs_dv`)
        for mock DESI BAO. 

        See also
        --------
        dataconfig.Data.desi_theory_fname
        """
        fname = self.desi_theory_fname()
        z, rs_dv = np.loadtxt(fname, unpack=True)
        return z, rs_dv


    def load_desi_covmat(self):
        """Returns a two-dimensional array holding the covariance matrix for 
        the mock DESI BAO measurements, r_s/d_V(z).

        See also
        --------
        dataconfig.Data.desi_covmat_fname
        dataconfig.Data.load_desi_theory
        """
        fname = self.desi_covmat_fname()
        covmat = np.loadtxt(fname)
        return covmat


    def load_bin_edges(self):
        """Returns a one-dimensional array holding the upper bin edge for 
        each bin, except the first element, which is the lower bin edge of 
        the first bin.
        """
        bin_edges = np.loadtxt(self.bin_edges_fname)
        return bin_edges
    
    
    def load_precomputed_desi_fisher(self, use_H0=False):
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
            A list of parameter names for the parameters in the Fisher matrix,
            in the same order as their corresponding rows/columns.
        """
        fname = self.precomputed_desi_fisher_fname(use_H0=use_H0)
        fisher_matrix, fisher_params = utils.load_fisher_matrix(fname)
        return fisher_matrix, fisher_params


    def load_precomputed_cmb_fisher(self, exp, cmb_type='delensed', use_H0=False, 
            with_desi=False, hd_lmax=None, include_fg=True, feedback=False):
        """Returns a Fisher matrix calculated for the given experiment and 
        kind of CMB spectra (lensed or delensed). The parameters in the Fisher 
        matrix are the six LCDM parameters, the effective number of relativistic 
        species, and the sum of the neutrino masses; if `feedback=True`, the 
        Fisher matrices also contain the baryonic feedback parameter of the
        single-parameter HMCode2020 + feedback model. Note that a Gaussian prior 
        of width 0.007 has already been applied. 

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the Fisher matrix was calculated from 
            delensed CMB TT, TE, EE, and BB power spectra, in addition to the 
            CMB lensing spectrum. If `cmb_type='lensed'`, the Fisher matrix
            was computed with lensed CMB spectra instead, as well as the CMB
            lensing spectrum. Note that passing `cmb_type='lensed'` is only an
            option for `exp='hd'`, `hd_lmax=None`, and `feedback=False`.
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the angular 
            scale of the sound horizon at last scattering (multiplied by 100)
            is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB spectra.
            If `True`, the Fisher matrix is the sum of a CMB and a mock DESI BAO
            Fisher matrix.
        include_fg : bool, default=True
            If `True`, return the file name for CMB-HD Fisher matrix that 
            was calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for the Fisher 
            matrix that was calculated by neglecting these effects. 
            Used only when `exp = 'hd'`; `include_fg=False` is only possible
            when `cmb_type='lensed'`.
        hd_lmax : int, default=None
            Used to return CMB-HD Fisher matrices that were calculated with 
            a lower maximum multipole than the baseline case. Only used when
            `exp='hd'`.
        feedback : bool, default=False
            If `True`, the Fisher matrix was calculated with the HMCode2020 
            + baryonic feedback non-linear model, and also contains the 
            baryonic feedback parameter of this model (i.e., a total of 9
            parameters). If `False`, the Fisher matrix was calculated with the 
            HMCode2016 CDM-only model. Used only when `exp='hd'`, for 
            `cmb_type='delensed'` and `hd_lmax=None`.

        Returns
        -------
        fisher_matrix : array_like of float
            The eight-parameter (if `feedback=False`) or nine-parameter (if
            `feedback=True`) Fisher matrix for the given experiment (or its
            combination with mock DESI BAO data if `with_desi=True`).
        fisher_params : list of str
            A list of parameter names for the parameters in the Fisher matrix,
            in the same order as their corresponding rows/columns.

        Raises
        ------
        ValueError
            If the `exp`, `cmb_type`, or `hd_lmax` value is invalid, or if 
            the requested Fisher matrix does not exist.

        Warns
        -----
        If the `hd_lmax`, `include_fg`, or `feedback` arguments were changed
        from their default value, but will be ignored.

        See also
        --------
        dataconfig.Data.precomputed_cmb_fisher_fname
        """
        fname = self.precomputed_cmb_fisher_fname(exp, cmb_type=cmb_type, use_H0=use_H0, with_desi=with_desi, hd_lmax=hd_lmax, include_fg=include_fg, feedback=feedback)
        fisher_matrix, fisher_params = utils.load_fisher_matrix(fname)
        return fisher_matrix, fisher_params





    # ----- convenience functions -----

    def load_cmb_lensing_spectrum(self, exp, output_lmax=None, hd_lmax=None, 
            feedback=False):
        """Returns the theoretical CMB lensing power spectrum, computed to
        the maximum multipole for the given experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`,
            or `'HD`'. The name is case-insensitive.
        output_lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_lmax` value.
        hd_lmax : int or None, default=None
            If a value of `hd_lmax` is provided, returns a spectrum that was
            calculated with a lower maximum multipole than the baseline case.
            Only used if `exp='hd'`. If not `None`, `hd_lmax` must be one of
            the values in `Data.hd_lmaxs`.
        feedback : bool, default=False
            If `True`, the power spectrum was calculated with the HMCode2020
            + baryonic feedback non-linear model. Otherwise, it was calculated
            with the HMCode2016 CDM-only model. Only used if `exp='hd'`.

        Returns
        -------
        L, clkk : array_like of float
            One-dimensional arrays containing the lensing multipoles (`L`)
            and the theoretical CMB lensing power spectrum (`clkk`).

        Raises
        ------
        ValueError
            If the `exp`, `cmb_type`, or `hd_lmax` value is invalid.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored.

        Note
        ----
        We use the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where
        L is the lensing multipole and  C_L^phiphi is the CMB lensing 
        potential power spectrum.

        See also
        --------
        dataconfig.Data.load_cmb_theory_spectra
        dataconfig.Data.cmb_theory_fname
        """
        theo = self.load_cmb_theory_spectra(exp, 'lensed', output_lmax=output_lmax, hd_lmax=hd_lmax, feedback=feedback)
        L = theo['ells'].copy()
        clkk = theo['clkk'].copy()
        return L, clkk


    def desi_redshifts(self):
        """Returns a one-dimensional array holding the redshifts at which
        the mock DESI BAO theory and covariance matrix were calculated.
        """
        z, _ = self.load_desi_theory()
        return z


    def binning_matrix(self, exp):
        """Returns a binning matrix of shape (num_bins, num_ells) to bin
        spectra in the multipole range for the given experiment, where num_bins
        is the total number of bins in the multipole range, and num_ells is the
        number of multipoles from 2 to the maximum multipole value (`lmax`), i.e.
        it is equal to `lmax - 1`.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`,
            or `'HD`'. The name is case-insensitive.

        Returns
        -------
        bmat : array_like of float
            The two-dimensional binning matrix.

        Raises
        ------
        ValueError
            If an invalid `exp` was passed.

        Note
        ----
        The spectra to be binned are expected to begin at a multipole of
        ell = 2, even if the minimum multipole is greater than this value.
        To bin spectrum with values from `lmin = 0` to `lmax` held in an 
        array `cl` of length `lmax+1`, you would multiply it by the binning 
        matrix as `bmat @ cl[2:]`.
        """
        exp = self.check_cmb_exp(exp)
        bin_edges = self.load_bin_edges()
        bmat = utils.binning_matrix(bin_edges, lmin=self.lmins[exp], lmax=self.lmaxs[exp], start_at_ell=2)
        return bmat


    def lbin(self, exp):
        """Returns the binned multipoles (i.e., the bin centers) in the range
        for the given experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`,
            or `'HD`'. The name is case-insensitive.

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
        dataconfig.Data.binning_matrix
        """
        exp = self.check_cmb_exp(exp)
        bmat = self.binning_matrix(exp)
        ells = np.arange(2, self.lmaxs[exp] + 1)
        lbin = bmat @ ells
        return lbin


    def fiducial_params(self, param_names=None, feedback=False):
        """Returns a dictionary containing cosmological parameter names and
        their fiducial values, along with any other names and values (e.g., for
        CAMB accuracy parameters) that are used when calculating the theory.

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
            A dictionary containing parameter names and their fiducial values.

        Raises
        ------
        ValueError
            If `param_names` is not `None`, but contains a name that is not
            specified in the fiducial parameter file.

        See also
        --------
        dataconfig.Data.fiducial_param_file
        """
        fid_params = theory.get_params(param_file=self.fiducial_param_file(feedback=feedback))
        params = {}
        if param_names is not None:
            for param in param_names:
                if param not in fid_params.keys():
                    err_msg = f"Invalid parameter name `'{param}'` in `param_names`: there are only fiducial values set for {fid_params.keys()}."
                    raise ValueError(err_msg)
                else:
                    params[param] = fid_params[param]
        else:
            params = fid_params.copy()
        return params
