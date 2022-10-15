import os
from astropy.table import QTable

default_p_mode_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'broomhall_2009_p_modes.ecsv'
)


def broomhall_p_mode_freqs(path=None):
    """
    Get p-mode frequencies from 23 years of BiSON observations by
    Broomhall et al. (2009), Table 2 [1]_.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2009MNRAS.396L.100B/abstract
    """
    if path is None:
        path = default_p_mode_path

    return QTable.read(path, format='ascii.ecsv')['nu']
