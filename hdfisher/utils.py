"""Helpful functions"""
import os
import logging
import logging.handlers
import warnings
import numpy as np
import yaml
from . import mpi


def tmsg(t):
    """given `t` (float) in seconds, return a string giving the time in 
    minutes and seconds.
    """
    m = int(t // 60)
    s = int(t - (m * 60))
    return f'{m:>3d} min {s:>2d} sec'


def cl2dl(cl, ells):
    """Returns the spectrum `cl` multiplied by a factor of 
    `ells * (ells + 1) / (2 * pi)`.
    
    Parameters
    ----------
    ells, cl : array_like of float
        One-dimensional arrays holding a power spectrum (`cl`) and its
        corresponding multipole values (`ells`).

    Returns
    -------
    dl : array_like of float
        An array holding the spectrum `cl` which is multiplied by
        `ells * (ells + 1) / (2 * pi)`.

    See Also
    --------
    cl2dl_dict
    """
    lfact = ells * (ells  + 1) / (2 * np.pi)
    dl = cl * lfact
    return dl


def cl2dl_dict(cls, ells=None, cmb_keys=['tt', 'te', 'ee', 'bb']):
    """Given a dictionary of CMB power spectra, returns a dictionary with 
    the TT, TE, EE, and BB C_ell's multiplied by a factor of 
    ell * (ell + 1) / (2 * pi).
    

    Parameters
    ----------
    cls : dict of array_like of float
        The dictionary containing the CMB power spectra, with their keys
        given by the `cmb_keys` argument. If the dictionary contains the
        multipoles for the spectra, the key is expected to be `'ells'`.
    ells : array_like of float or int, default=None
        An array holding the multipoles of the spectra. Must be passed
        if the `cls` dictionary does not have a key `'ells'` holding
        this array. 
    cmb_keys : list of str, default=['tt', 'te', 'ee', 'bb']
        The keys of the `cls` dict holding the spectra that should be 
        multiplied by `ells * (ells + 1) / (2 * pi)`.

    Returns
    -------
    dls : dict of array_like of float
        The input `cls` dictionary, where the spectra in the `cmb_keys`
        have been multiplied by the factor of `ells * (ells + 1) / (2 * pi)`.
        Also contains a key `'ells'` for the multipoles.

    Raises
    ------
    ValueError
        If the is no key `'ells'` in `cls`, and `ells` is `None`.

    See Also
    --------
    cl2dl
    """
    if ells is None:
        if 'ells' in cls.keys():
            ells = cls['ells'].copy()
        else:
            err_msg = "`ells` is None and `cls['ells']` does not exist: You must pass an array holding the multipoles as either `ells`, or in `cls['ells']`."
            raise ValueError(err_msg)
    dls = {'ells': ells.copy()}
    for key in cls.keys():
        if key in cmb_keys:
            dls[key] = cl2dl(cls[key].copy(), ells)
        else:
            dls[key] = cls[key].copy()
    return dls


def clxcl_to_dlxcl(covmat, cov_ells):
    """Given a covariance matrix `covmat` whose entries are the covariance 
    between power spectra C_ell and C_ell', returns a covariance matrix
    for the covariance between D_ell and D_ell', where 
    D_ell = ell * (ell + 1) / (2 * pi).
    
    Parameters
    ----------
    covmat : array_like of float
        The two-dimensional covariance matrix.
    cov_ells : array_like of float
        A one-dimensional array holding the multipoles (or bin centers) 
        corresponding to each row/column of the `covmat`.

    Returns
    -------
    dl_covmat : array_like of float
        The converted covariance matrix.
    """
    dl_covmat = covmat.copy()
    lfact = cov_ells * (cov_ells + 1) / (2 * np.pi)
    for i, ell_fact in enumerate(lfact):
        dl_covmat[i] = dl_covmat[i] * ell_fact * lfact
    return dl_covmat



# ---------- input / output ---------- 

def load_from_file(fname, columns, skip_cols=[]):
    """Returns dict of columns loaded from a `.txt` file.

    Parameters
    ----------
    fname : str
        The filename to load from.
    columns : list of str
        The names, in order, of each column, which will also serve as the
        dict keys.
    skip_cols : list of str, default=[]
        The names of any columns that should not be included in the output
        dict.

    Returns
    -------
    data : dict of array_like
        A dictionary whose keys are the names in `columns` and values are
        one-dimensional arrays holding the data from the corresponding
        column in the file.
    """
    data = {}
    data_array = np.loadtxt(fname)
    for i, col in enumerate(columns):
        if col not in skip_cols:
            data[col] = data_array[:,i].copy()
    return data


def save_to_file(fname, data, keys=None, col_names=None, extra_header_info=None):
    """Save 1D arrays in a dict as columns in a `.txt` file.

    Parameters
    ----------
    fname : str
        The filename to save to.
    data : dict of array_like
        The dictionary of arrays to save.
    keys : list, default=None
        A list of keys to use. Otherwise, all keys are used
        (and order of columns is not pre-determined).
    col_names : list, default=None
        A list of names to use for each column, in the same order as `keys`
        (which must also be passed). Otherwise the keys are used.
    extra_header_info : str, default=None
        If not `None`, add the `extra_header_info` to the beginning 
        of the header, followed by the column names.
    """
    if keys is None:
        col_names = None
        keys = list(data.keys())
    if col_names is None:
        col_names = keys.copy()
    header = ', '.join(col_names)
    if extra_header_info is not None:
        header = f'{extra_header_info} {header}'
    data_cols = []
    for key in keys:
        data_cols.append(data[key])
    np.savetxt(fname, np.column_stack(data_cols), header=header)


def load_yaml(fname):
    """Return a dictionary loaded from a YAML file.

    Parameters
    ----------
    fname : str
        The YAML file name.

    Returns
    -------
    data : dict
        The dictionary loaded from the YAML file.

    See Also
    --------
    save_yaml
    """
    with open(fname, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(fname, save_dict, overwrite=False):
    """Save a dictionary to a YAML file.

    Parameters
    ----------
    fname : str
        The file name, including the file extension.
    save_dict : dict
        The dictionary to save.
    overwrite : bool, default=False
        Whether to overwrite the file, if it already exists.

    See Also
    --------
    load_yaml
    """
    if overwrite or (not os.path.exists(fname)):
        with open(fname, 'w') as f:
            yaml.dump(save_dict, f,  default_flow_style=False)


def set_dir(dirname):
    """Given the path to a directory, check if it exists. If not, create a 
    new directory.

    Parameters
    ----------
    dirname : str
        The full path of the directory.

    Returns
    -------
    dirname : str
        The full path of the directory (which will not definitely exist).

    Raises
    ------
    FileExistsError
        If the path exists, but it points to a file, not a directory.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif not os.path.isdir(dirname):
        err_msg = f'{dirname} already exists but is not a directory.'
        raise FileExistsError(err_msg)
    return dirname


def save_fisher_matrix(fname, fisher_matrix, params):
    """Save the Fisher matrix to the given file, with a header giving the
    parameter names in the correct order.

    Parameters
    ----------
    fname : str
        The file name (including absolute path) to save the Fisher matrix to.
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    params : list of str
        A list of parameter names, in the same order as in the Fisher matrix.

    See Also
    --------
    load_fisher_matrix
    """
    header = ' '.join(params)
    np.savetxt(fname, fisher_matrix, header=header)


def load_fisher_matrix(fname):
    """Returns the Fisher matrix loaded from the given file, with a list of
    parameter names in the correct order.

    Parameters
    ----------
    fname : str
        The file name (including absolute path) to save the Fisher matrix to.

    Returns
    -------
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    params : list of str
        A list of parameter names, in the same order as in the Fisher matrix.
    
    See Also
    --------
    save_fisher_matrix
    """
    fisher_matrix = np.loadtxt(fname)
    # read the header to get the params
    with open(fname, 'r') as f:
        header = f.readline()
    header = header.strip('# \n')
    params = header.split(' ')
    return fisher_matrix, params


# ---------- parameters ---------- 

def add_param_aliases(params_dict, replace=True, use_class=False):
    """Add the `hdfisher` parameter "aliases" to a dictionary of CAMB
    or CLASS parameters. By default, `hdfisher` varies these "aliases"
    instead of the parameters being replaced.
    
    Parameters
    ----------
    params_dict : dict
        A dictionary of parameter names and values. May include any
        parameter names that can be passed to CAMB (if `use_class=False`)
        or CLASS (if `use_class=True`).
    replace : bool, default=True
        If `replace=True`, for each parameter with an "alias", the
        corresponding CAMB/CLASS parameter name will be removed from the
        dictionary. If `replace=False`, these parameters will not be
        removed.
    use_class : bool, default=False
        Whether CLASS is being used instead of CAMB.
        
    Returns
    -------
    params_dict : dict
        The dictionary of parameter names and values, with keys added for
        each parameter "alias".
        
    See Also
    --------
    add_camb_param_aliases, add_class_param_aliases 
    remove_param_aliases
    """
    # TODO
    if use_class:
        raise NotImplementedError
    else:
        params_dict = add_camb_param_aliases(params_dict, replace=replace)
    return params_dict


def remove_param_aliases(params_dict, use_class=False):
    """Replace the `hdfisher` parameter "aliases" with the corresponding
    CAMB or CLASS parameters. 
    
    Parameters
    ----------
    params_dict : dict
        A dictionary of parameter names and values. May include any
        parameter names that can be passed to CAMB (if `use_class=False`)
        or CLASS (if `use_class=True`) and the `hdfisher` parameter 
        "aliases".
    use_class : bool, default=False
        Whether CLASS is being used instead of CAMB.
        
    Returns
    -------
    params_dict : dict
        The dictionary of CAMB/CLASS parameter names and values.
        
    See Also
    --------
    remove_camb_param_aliases, remove_class_param_aliases 
    add_param_aliases
    """
    # TODO
    if use_class:
        raise NotImplementedError
    else:
        params_dict = remove_camb_param_aliases(params_dict)
    return params_dict


def add_camb_param_aliases(params_dict, replace=True):
    """Add the `hdfisher` parameter "aliases" to a dictionary of CAMB
    parameters; these aliases are:
    - `'theta'` instead of `'cosmomc_theta'` (cosmoMC theta times 100)
    - `'logA'` instead of `'As'` (ln(10^10 As))
    Note that, by default, `hdfisher` varies these "aliases" instead of
    the parameters being replaced.

    If `replace=True`, `'cosmomc_theta'` and `'As'` are removed from the
    dictionary.
    """
    if 'cosmomc_theta' in params_dict:
        if replace:
            cosmomc_theta = params_dict.pop('cosmomc_theta')
        else:
            cosmomc_theta = params_dict['cosmomc_theta']
        params_dict['theta'] = 100 * cosmomc_theta
    if 'As' in params_dict:
        if replace:
            As = params_dict.pop('As')
        else:
            As = params_dict['As']
        params_dict['logA'] = float(np.log(1e10 * As))
    return params_dict


def remove_camb_param_aliases(params_dict):
    """Remove the `hdfisher` parameter "aliases" from a dictionary of
    CAMB parameters, replacing them with the parameters that can be
    passed to CAMB:
    - Replaces `'theta'` with `'cosmomc_theta`'
    - Replaces `'logA'` with `'As'`
    """
    if 'theta' in params_dict:
        params_dict['cosmomc_theta'] = params_dict.pop('theta') / 100
    if 'logA' in params_dict:
        params_dict['As'] = float(np.exp(params_dict.pop('logA')) * 1e-10)
    return params_dict


def add_class_param_aliases(params_dict, replace=True):
    # TODO: will add `sum_m_ncdm` given `m_ncdm`
    raise NotImplementedError


def remove_class_param_aliases(params_dict):
    # TODO: will replace `sum_m_ncdm` with `m_ncdm`
    raise NotImplementedError


# ---------- binning ---------- 

def bin_info(bin_edges, lmin=None, lmax=None):
    """Given an array of `bin_edges`, return arrays of the `lower` and 
    `upper` bin edges, and the bin `center`, between `lmin` and `lmax`.
    
    Parameters
    ----------
    bin_edges : array_like of int
        A one dimensional array holding the upper bin edge for each bin, 
        except the first element, which is the lower bin edge of the first bin.
    lmin, lmax : int or None, default=None
        If provided, return arrays of the `lower` and `upper` bin edges,
        and the bin `center`, between `lmin` and `lmax`. If `lmin` is `None`, 
        we use the first value in the `bin_edges` array; if `lmax` is `None`, 
        we use the last value in the `bin_edges` array.

    Returns
    -------
    lower, upper, center : array_like of float
        One-dimensional arrays of the lower edge, upper edge, and center,
        respectively, of each bin, in the range [`lmin`, `lmax`] if
        at least one was provided.
    """
    lmin = int(lmin) if (lmin is not None) else int(bin_edges[0])
    lmax = int(lmax) if (lmax is not None) else int(bin_edges[-1])
    # get upper and lower edges
    upper = bin_edges[1:].copy()
    lower = bin_edges[:-1].copy()
    # add one to all lower edges, except the first,
    # so each bin includes its lower and its upper edge
    lower[1:] += 1
    # trim between lmin and lmax
    loc = np.where((lower >= lmin) & (upper <= lmax))
    upper = upper[loc]
    lower = lower[loc]
    # get the bin centers
    center = (lower + upper) / 2.
    return lower, upper, center


def nbins_per_spectrum(ell_ranges, bin_edges):
    """Given the minimum and maximum multipoles used for each spectrum, along
    with the bin edges, returns a dictionary holding the number of bins used
    for each spectrum.

    Parameters
    ----------
    ell_ranges : dict of list or tuple of int
        A dictionary with keys given by the elements of `spectra`, holding
        a tuple or list `[lmin, lmax]` giving the minimum and maximum 
        multipoles used for each spectra. 
    bin_edges : array_like of int
        A one dimensional array holding the upper bin edge for each bin, 
        except the first element, which is the lower bin edge of the first bin.

    Returns
    -------
    nbins : dict of int
        A dictionary with the same keys as `ell_ranges`, holding the number
        of bins used for each spectrum.
    """
    nbins = {}
    for s, (lmin, lmax) in ell_ranges.items():
        _, _, lbin = bin_info(bin_edges, lmin=lmin, lmax=lmax)
        nbins[s] = len(lbin)
    return nbins


def binning_matrix(bin_edges, lmin=None, lmax=None, start_at_ell=0):
    """Create a (num_bins, num_ells) binning matrix, which will bin the values
    between `lmin` and `lmax` in a vector/matrix containing values for each 
    multipole between the `start_at_ell` and `lmax` values. For example, for
    an array `c_ell` holding a power spectrum with a value at each multipole 
    `ell` in the range [2, 5000], to bin only the values in the range [30, 3000], 
    you would pass `lmin = 30`, `lmax = 3000`, and `start_at_ell=2`. 
    
    Parameters
    ----------
    bin_edges : array_like of int
        A one dimensional array holding the upper bin edge for each bin, 
        except the first element, which is the lower bin edge of the first bin.
    lmin, lmax : int or None, default=None
        The minimum and maximum multipole values of the quantity to be binned,
        i.e. only values between `lmin` and `lmax` will be binned. If `lmin` is 
        `None`, we use the first value in the `bin_edges` array; if `lmax` 
        is `None`, we use the last value in the `bin_edges` array.
    start_at_ell : int, default=0
        The minimum multipole value in the quantity to be binned, which
    """
    lmin = int(lmin) if (lmin is not None) else int(bin_edges[0])
    lmax = int(lmax) if (lmax is not None) else int(bin_edges[-1])
    ell_min = int(start_at_ell)
    ells = np.arange(ell_min, lmax+1)
    nells = len(ells)
    # get upper and lower edges
    lower, upper, _ = bin_info(bin_edges, lmin=lmin, lmax=lmax)
    nbin = len(upper)
    # make binning matrix
    binmat = np.zeros((nbin, nells))
    for i, (bmin, bmax) in enumerate(zip(lower, upper)):
        loc = np.where((ells >= bmin) & (ells <= bmax))
        n = bmax - bmin + 1 # number of ells in this bin
        binmat[i][loc] = 1 / n
    return binmat


def bin1d(ells, cls, bin_edges, lmin=None, lmax=None):
    """Bin a power spectrum `cls`, which has a value at every multipole in 
    `ells`, between `lmin` and  `lmax` using the given `bin_edges`.
    
    Parameters
    ----------
    ells, cls : array_like of float
        One-dimensional arrays holding a power spectrum (`cls`) and its
        corresponding multipole values (`ells`).
    bin_edges : array_like of int
        A one dimensional array holding the upper bin edge for each bin, 
        except the first element, which is the lower bin edge of the first bin.
    lmin, lmax : int or None, default=None
        The minimum and maximum multipole values of the quantity to be binned,
        i.e. only values between `lmin` and `lmax` will be binned. If `lmin` is 
        `None`, we use the first value in the `bin_edges` array; if `lmax` is 
        `None`, we use the last value in the `bin_edges` array.

    Returns
    -------
    lbin, binned_cls : array_like of float
        The bin centers (`lbin`; i.e., the binned `ells`) and binned power 
        spectrum (`binned_cls`; i.e., the binned `cls`). The number of elements
        in each array corresponds to the total number of bins, or the number of
        bins between `lmin` and `lmax` if either was provided.

    See Also
    --------
    binning_matrix
    bin1d
    """
    # get lmin, lmax if not provided
    if lmin is None:
        spec_lmin = int(ells[0])
        bin_lmin = int(bin_edges[0])
        lmin = max([spec_lmin, bin_lmin])
    if lmax is None:
        spec_lmax = int(ells[-1])
        bin_lmax = int(bin_edges[-1])
        lmax = min([spec_lmax, bin_lmax])
    # get lower/upper edges and the binned ells
    lower, upper, lbin = bin_info(bin_edges, lmin=lmin, lmax=lmax)
    nbin = len(lbin)
    # take the mean of the cls in each bin
    binned_cls = np.zeros(nbin)
    for i, (bmin, bmax) in enumerate(zip(lower, upper)):
        loc = np.where((ells >= bmin) & (ells <= bmax))
        binned_cls[i] = np.mean(cls[loc])
    return lbin, binned_cls


def bin_theo_dict(ells, theo, bin_edges, lmin=None, lmax=None, ell_ranges=None):
    """Bin each one-dimensional power spectrum array in the `theo` dict, each of 
    which has a value at every multipole in `ells`, between `lmin` and  `lmax` 
    using the given `bin_edges`.
    
    Parameters
    ----------
    ells : array_like of float
        A one-dimensional arrays holding the multipole values for each spectrum 
        in the `theo` dict.
    theo : dict of array_like of float
        A dictionary holding one-dimensional arrays of power spectra, each of 
        which has a value for each multipole in `ells`.
    bin_edges : array_like of int
        A one dimensional array holding the upper bin edge for each bin, 
        except the first element, which is the lower bin edge of the first bin.
    lmin, lmax : int or None, default=None
        The minimum and maximum multipole values of the quantity to be binned,
        i.e. only values between `lmin` and `lmax` will be binned. If `lmin` is 
        `None`, we use the first value in the `bin_edges` array; if `lmax` is 
        `None`, we use the last value in the `bin_edges` array.
    ell_ranges : dict of list of int, or None, default=None
        A dictionary with the same keys as the spectra in `theo`, holding a 
        list in the form `[lmin, lmax]` giving the minimum (`lmin`) and 
        maximum (`lmax`) multipoles to be used when binning each spectrum. 
        If `None`, it's assumed that the multipole ranges are the same for 
        all spectra.

    Returns
    -------
    binned_ells : array_like of float
        A one-dimensional array of bin centers. 
    binned_cls : dict of array_like of float
        A dictionary containing binned power spectra for each spectrum in 
        the `theo` dictionary, with the same keys.
    """
    if lmin is None:
        spec_lmin = int(ells[0])
        bin_lmin = int(bin_edges[0])
        lmin = max([spec_lmin, bin_lmin])
    if lmax is None:
        spec_lmax = int(ells[-1])
        bin_lmax = int(bin_edges[-1])
        lmax = min([spec_lmax, bin_lmax])
    if ell_ranges is None:
        ell_ranges = {s: [lmin, lmax] for s in theo.keys()}
    else:
        # get min, max ell out of all ranges
        ell_min = min([ell_ranges[s][0] for s in ell_ranges.keys()])
        ell_max = max([ell_ranges[s][1] for s in ell_ranges.keys()])
        for s in theo.keys():
            if s not in ell_ranges:
                ell_ranges[s] = [ell_min, ell_max]
    binned_theo = {}
    for s in theo.keys():
        binned_ells, binned_theo[s] = bin1d(ells, theo[s], bin_edges, lmin=ell_ranges[s][0], lmax=ell_ranges[s][1])
    return binned_ells, binned_theo


# ---------- covmat ----------


def cov_to_blocks(cov, spectra=['tt', 'te', 'ee', 'bb', 'kk'], ell_ranges=None, bin_edges=None):
    """Given a covariance matrix `cov` containing blocks for the covariance
    between the different `spectra` (e.g. a TT x TT block for the temperature power
    spectrum auto-covariance, a TT x kappakappa block for the temperature and 
    lensing potential power spectra cross-covariance, etc.), returns a nested dict
    of the different blocks.
    
    Parameters
    ----------
    cov : array_like
        The full, two-dimensional covariane matrix for the given `spectra`.
    spectra : list of str, default=['tt', 'te', 'ee', 'bb', 'kk']
        A list of the spectra in the covariance matrix, in the correct order.
    ell_ranges : dict of list of int, or None, default=None
        A dictionary with keys given by the elements of `spectra`, holding
        a tuple or list `[lmin, lmax]` giving the minimum and maximum 
        multipoles used for each spectra. If `None`, it's assumed that
        each block of the covariance matrix has the same multipole range.
        Otherwise the `bin_edges` must be provided.
    bin_edges : array_like of int, default=None
        A 1D array holding the multipole values for the lower edge of the 
        first bin, and upper edges of each bin. Must be passed with
        `ell_ranges`, if the blocks in the covariance matrix have different 
        multipole ranges.

    Returns
    -------
    blocks : dict of dict of array_like
        A nested dict, with keys given by the elements in `spectra`, holding the
        two-dimensional covariance matrix for that block. The first key gives
        the row of the block, and the second gives the column. See the note below.

    Raises
    ------
    ValueError
        If the `ell_ranges` are provided but no `bin_edges` were given.

    Note
    ----
    For a number `nspec` of different spectra, there are `nspec * nspec` blocks 
    in the covariance matrix. For the default `spectra=['tt', 'te', 'ee', 'bb', 'kk']`,
    `nspec = 5`, so there are 25 blocks. The first row of blocks will be TT x TT,
    TT x TE, TT x EE, TT x kappakappa; the second will be TE x TT, TE x TE, etc.
    These are stored in the dict as `blocks['tt']['tt']` for TT x TT,
    `blocks['tt']['te']` for TT x TE, etc.

    See Also
    --------
    cov_from_blocks :
        The inverse function which takes in `blocks` and returns the `cov`.
    """
    nspec = len(spectra)
    # get number of bins for each spectrum
    if ell_ranges is None: # each block has same number of bins
        nbin = cov.shape[0] // nspec # per block
        nbins = {s: nbin for s in spectra}
    else: # each block may have different number of bins
        if bin_edges is None:
            err_msg = 'You must also provide the `bin_edges` with the `ell_ranges`.'
            raise ValueError(err_msg)
        else:
            nbins = nbins_per_spectrum(ell_ranges, bin_edges)
    blocks = {}
    for i1, s1 in enumerate(spectra):
        blocks[s1] = {}
        for i2, s2 in enumerate(spectra):
            # indices where this block starts and ends
            imin1 = sum([nbins[s] for s in spectra[:i1]])
            imin2 = sum([nbins[s] for s in spectra[:i2]])
            imax1 = imin1 + nbins[s1]
            imax2 = imin2 + nbins[s2]
            blocks[s1][s2] = cov[imin1:imax1, imin2:imax2].copy()
    return blocks


def cov_from_blocks(blocks, spectra=['tt', 'te', 'ee', 'bb', 'kk'], ell_ranges=None, bin_edges=None):
    """Given a nested dict `blocks` containing covariance matrices for the
    power spectra types in `spectra` (e.g., `blocks['tt']['ee']` is the 
    TT x EE cross-covariance matrix for the temperature and polarization 
    power spectra), return a single covariance matrix containing the blocks.
    
    Parameters
    ----------
    blocks : nested dict of array_like
        A nested dict, with keys given by the elements in `spectra`, holding the
        two-dimensional covariance matrix for that block. The first key gives
        the row of the block, and the second gives the column. See the note below.
    spectra : list of str, default=['tt', 'te', 'ee', 'bb', 'kk']
        A list of the spectra in the covariance matrix, in the correct order.
    ell_ranges : dict of list or tuple of int, default=None
        A dictionary with keys given by the elements of `spectra`, holding
        a tuple or list `[lmin, lmax]` giving the minimum and maximum 
        multipoles used for each spectra. If `None`, it's assumed that
        each block of the covariance matrix has the same multipole range.
        Otherwise the `bin_edges` must be provided.
    bin_edges : array_like of int, default=None
        A 1D array holding the multipole values for the lower edge of the 
        first bin, and upper edges of each bin. Must be passed with
        `ell_ranges`, if the blocks in the covariance matrix have different 
        multipole ranges.

    Returns
    -------
    cov : array_like
        The full, two-dimensional covariance matrix for the given `spectra`.

    Raises
    ------
    ValueError
        If the `ell_ranges` are provided but no `bin_edges` were given.

    Note
    ----
    For a number `nspec` of different spectra, there are `nspec * nspec` blocks 
    in the covariance matrix. For the default `spectra=['tt', 'te', 'ee', 'bb', 'kk']`,
    `nspec = 5`, so there are 25 blocks. The first row of blocks will be TT x TT,
    TT x TE, TT x EE, TT x kappakappa; the second will be TE x TT, TE x TE, etc.
    These are stored in the dict as `blocks['tt']['tt']` for TT x TT,
    `blocks['tt']['te']` for TT x TE, etc.

    See Also
    --------
    cov_from_blocks :
        The inverse function which takes in the `cov` and returns `blocks`.
    """
    nspec = len(spectra)
    # get number of bins for each spectrum
    if ell_ranges is None: # each block has same number of bins
        nbin = blocks[spectra[0]][spectra[0]].shape[0] # per block
        nbins = {s: nbin for s in spectra}
        nbin_tot = nbin * nspec
    else: # each block may have different number of bins
        if bin_edges is None:
            err_msg = 'You must also provide the `bin_edges` with the `ell_ranges`.'
            raise ValueError(err_msg)
        else:
            nbins = nbins_per_spectrum(ell_ranges, bin_edges)
            nbin_tot = sum([nbins[s] for s in spectra])
    cov = np.zeros((nbin_tot, nbin_tot))
    for i1, s1 in enumerate(spectra):
        for i2, s2 in enumerate(spectra):
            # indices where this block starts and ends
            imin1 = sum([nbins[s] for s in spectra[:i1]])
            imin2 = sum([nbins[s] for s in spectra[:i2]])
            imax1 = imin1 + nbins[s1]
            imax2 = imin2 + nbins[s2]
            cov[imin1:imax1, imin2:imax2] = blocks[s1][s2].copy()
    return cov


def trim_cov(cov, cov_ell_ranges, cov_spectra=['tt', 'te', 'ee', 'bb', 'kk'], ell_ranges=None, bin_edges=None, spectra=None):
    """Given a full covariance matrix `cov` with different blocks for the 
    covariance between the different `cov_spectra`, with the (current) 
    multipole ranges given in `cov_ell_ranges`, returns a covariance
    matrix with blocks for the given list of `spectra` (if provided)
    in the new multipole ranges (if provided) for each spectrum.

    Parameters
    ----------
    cov : array_like
        The full, two-dimensional covariane matrix for the given `cov_spectra`.
    cov_ell_ranges : dict of list or tuple of int
        A dictionary with keys for each spectrum in `cov_spectra`, holding
        a tuple or list `[lmin, lmax]` giving the minimum and maximum 
        multipoles that are currently used for each spectrum. 
    cov_spectra : list of str, default=['tt', 'te', 'ee', 'bb', 'kk']
        A list of the spectra in the full covariance matrix, in the correct order.
    ell_ranges : dict of list or tuple of int, default=None
        A dictionary with keys for each spectrum in `spectra`, holding
        a tuple or list `[lmin, lmax]` giving the minimum and maximum 
        multipoles to use for each spectrum. If `None`, the full multipole
        ranges in `cov_ell_ranges` are used, for each spectrum in `spectra`.
        Otherwise, the `bin_edges` must also be provided.
    bin_edges : array_like of int, default=None
        A 1D array holding the multipole values for the lower edge of the 
        first bin, and upper edges of each bin. Must be provided if 
        `ell_ranges` is not `None`.
    spectra : list of str, default=None
        A list of the spectra to include in the new covariance matrix, in 
        the correct order. If `None`, all spectra in `cov_spectra` are used.

    Returns
    -------
    trimmed_cov : array_like
        The new covariance matrix.

    Raises
    ------
    ValueError
        If `ell_ranges` is not `None` but `bin_edges` is None; or if there 
        are any elements in `spectra` that are not in `cov_spectra`.

    See also
    --------
    cov_to_blocks
    cov_from_blocks
    trim_cov_block_ell_range
    """
    # check the input:
    if spectra is None:
        spectra = cov_spectra.copy()
    else:
        if any([s not in cov_spectra for s in spectra]):
            err_msg = f"You requested a spectrum in `spectra = {spectra}` that is not in `cov_spectra = {cov_spectra}."
            raise ValueError(err_msg)
    if (ell_ranges is not None) and (bin_edges is None):
        err_msg = 'You must also provide the `bin_edges` with the `ell_ranges`.'
        raise ValueError(err_msg)
    # get the blocks, trim the ell range for each (if necessary), and put the
    # requested blocks back together:
    blocks = cov_to_blocks(cov, spectra=cov_spectra, ell_ranges=cov_ell_ranges, bin_edges=bin_edges)
    if ell_ranges is not None:
        for s1 in spectra:
            cov_lmin1, cov_lmax1 = cov_ell_ranges[s1]
            lmin1, lmax1 = ell_ranges[s1]
            imin1, imax1 = get_trimmed_bin_indices(bin_edges, cov_lmin1, cov_lmax1, lmin1, lmax1)
            for s2 in spectra:
                cov_lmin2, cov_lmax2 = cov_ell_ranges[s2]
                lmin2, lmax2 = ell_ranges[s2]
                imin2, imax2 = get_trimmed_bin_indices(bin_edges, cov_lmin2, cov_lmax2, lmin2, lmax2)
                blocks[s1][s2] = blocks[s1][s2][imin1:imax1+1, imin2:imax2+1]
    trimmed_cov = cov_from_blocks(blocks, spectra=spectra, ell_ranges=ell_ranges, bin_edges=bin_edges)
    return trimmed_cov


def get_trimmed_bin_indices(bin_edges, lmin_full, lmax_full, lmin_trim, lmax_trim):
    """
    Given the bin edges, get the first and last index of the bins between 
    `lmin_trim` and `lmax_trim`, relative to the first bin in the range 
    `lmin_full` to `lmax_full`.

    Parameters
    ----------
    bin_edges : array_like of int
        An array of bin edges.
    lmin_full, lmax_full : int
        The minimum and maximum multipoles in the full range.
    lmin_trim, lmax_trim : int
        The minimum and maximum multipoles in the trimmed range.

    Returns
    -------
    imin, imax : int
        The minimum and maximum indices of the trimmed bins, relative to
        the full bins.
    
    Example
    -------
    If your bin edges are `[0, 100, 200, ..., 600]`, and `lmin_full = 100`, 
    `lmax_full = 600` , then there are four bins in the full multipole range
    with centers at `[150.5, 250.5, 350.5, 450.5, 550.5]`. If the trimmed
    multipole range is `lmin_trim, lmax_trim = 200, 500`, there are 3 bins
    with centers at `[250.5, 350.5, 450.5]`, so the function will return
    the indices `(1, 3)`.

    See also
    --------
    trim_cov : where this function is used
    """
    _, _, lbin_full = bin_info(bin_edges, lmin=lmin_full, lmax=lmax_full)
    _, _, lbin_trim = bin_info(bin_edges, lmin=lmin_trim, lmax=lmax_trim)
    # get indices above `lmin_trim` to get first index
    lower_loc = np.where(lbin_full >= lbin_trim[0])
    imin = lower_loc[0][0]
    # get indices below `lmax_trim` to get last index
    upper_loc = np.where(lbin_full <= lbin_trim[-1])
    imax = upper_loc[0][-1]
    return imin, imax


# ---------- logging ---------- 

def get_logger(log_file=None, level='debug', name=None, fmt=None, datefmt=None):
    """Get a `logging.Logger` instance to log messages in a file.

    Parameters
    ----------
    log_file : str, default=None
        The full path to the file to be used for logging output.
        If None, logging output is printed to the terminal screen.
    level : str, default='debug'
        The minimum level of severity of the events passed to the logger.
        Valid options are 'debug', 'info', 'warning', 'error', and 'critical',
        in order of increasing severity.
    name : str, default=None
        An optional unique name to identify the logger. If None, the root logger
        is used.
    fmt : str, default=None
        A format string used to format the messages written to the log file.
        If None, a default is used. See the `logging` module documentation
        for details.
    datefmt : str, default=None
        A format string used to format the date and/or time of the message, if
        this information is included in the format string passed as `fmt`.
        If None, a default is used. See the `logging` module documentation
        for details.

     Returns
    -------
    logger : `logging.Logger`
        An instance of the `logging.Logger` object.

    Raises
    ------
    ValueError
        If the `level` is not a valid logging level.

    See Also
    --------
    logging : The logging module in the Standard Python Library.

    Notes
    -----
    The `level` argument is case-insensitive.  Only messages at or above the
    specified `level` are written to the log file. If an invalid `level` is
    passed, the default is used.

    The default format of each log entry includes the date and time, the 
    of the file and function, along with the line number, from which the 
    entry originated; and the level of severity of the message, followed
    by the message itself. The format is very long, but descriptive.

    If you pass your own format string, `style` (either '%' or '{') will be
    inferred by looking for one of the symbols, so you should not use both.

    Example
    -------
    In the following example, the file 'example.log' will contain a single line,
    "INFO: generating a random number".

    >>> fmt = '{levelname:s}: {message:s}' # choose a simple format
    >>> log = get_logger(log_file='example.log', level='info', fmt=fmt) # get the logger
    >>> log.info("generating a random number") # add a message, like an FYI
    >>> import random
    >>> x = random.uniform(-1,1)
    >>> if x < 0.0: log.debug("x is negative") # won't be logged b/c level='info'

    """
    # ---- set variables to get the logger ----

    # get (and create, if necessary) the log directory and file name
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if mpi.rank == 0:
            set_dir(log_dir)
        mpi.comm.barrier()

    # set the logging level
    levels = {'debug': logging.DEBUG, 'info': logging.INFO,
            'warning': logging.WARNING, 'error': logging.ERROR,
            'critical': logging.CRITICAL}
    valid_levels = list(levels.keys())
    if level.lower() not in valid_levels:
        msg = f"You passed an invalid logging level:`level='{level}'`. Valid levels are: {valid_levels}"
        raise ValueError(msg)
    level = levels[level.lower()]

    # set the formatting
    default_fmt = "[{asctime:s}  {filename:<15s}  L {lineno:<4d} in {funcName:15s}] {levelname:s}: {message:s}"
    default_style = '{'
    if fmt is None:
        fmt   = default_fmt
        style = default_style
    else:
        if '{' in fmt:
            style = '{'
        elif '%' in fmt:
            style = '%'
        else: # use the default and warn the user
            msg = 'You passed the following format string: fmt="{}"'.format(fmt)
            msg += ", but it does not contain any formatting placeholders."
            msg += " Using the default formatting."
            warnings.warn(msg)
            fmt = default_fmt
            style = default_style

    #default_datefmt = "%Y-%m-%d %I:%M:%S %p"
    default_datefmt = "%Y-%m-%d %H:%M:%S"
    if datefmt is None:
        datefmt = default_datefmt

    # ---- get the logger ----
    logger = logging.getLogger(name=name)
    logger.setLevel(level)

    # get the handler
    if log_file is not None:
        # use `RotatingFileHandler` (instead of `FileHandler`) because it will
        #  automatically start over once log gets too large
        max_bytes = 500000000
        backup_count = 3
        handler = logging.handlers.RotatingFileHandler(log_file, mode='a',
                maxBytes=max_bytes, backupCount=backup_count)
    else:
        handler = logging.StreamHandler()
    handler.setLevel(level)

    # get the formatter
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt, style=style)

    # put it all together
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # add a message that log has been initialized
    logger.info("***** logger initialized *****")

    return logger


