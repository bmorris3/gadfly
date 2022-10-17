import os

import numpy as np

from astropy.io import fits
from astropy.table import QTable, Table
from astropy.time import Time
from astropy.utils.data import download_file

from gadfly.psd import ppm

__all__ = ['download_soho_virgo_time_series']

full_soho_virgo_url = "https://stsci.box.com/shared/static/ejsyj0bse6ciqq0ry2gzjdjgbied7cj1.gz"
one_year_soho_virgo_url = "https://stsci.box.com/shared/static/lk6aqc1w8gjwt1ff1foad4i9lukuj76u.gz"
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


def download_soho_virgo_time_series(full_time_series=False, cache=True, name='SOHO VIRGO/PMO6'):
    """
    Download the total solar irradiance time series measurements
    from the SOHO VIRGO/PMO6 instrument [1]_.

    Interpolate over the missing measurements to provide a
    uniform time series.

    Parameters
    ----------
    full_time_series : bool
        Download the full timeseries (60 MB), otherwise retrieve only
        observations from the year 2001 (6 MB)
    cache : bool
        Cache the downloaded file locally
    name : str
        Name of the resulting light curve

    Returns
    -------
    light_curve : ~lightkurve.lightcurve.LightCurve

    References
    ----------
    .. [1] `<VIRGO Data Products Archive
           https://www.pmodwrc.ch/en/research-development/solar-physics/virgo-data-products-archived_webpage/>`_
    """
    from lightkurve import LightCurve
    meta = dict(name=name)

    if full_time_series:
        path = download_file(full_soho_virgo_url, cache=cache, pkgname='gadfly')
        hdu = fits.open(path)
        raw_fluxes, header = hdu[0].data, hdu[0].header

        soho_mission_day = Time("1995-12-1 00:00")

        # reconstruct the time axis as prescribed in the FITS header:
        times = (
            soho_mission_day.jd +
            header['TIME'] +
            np.arange(header['NAXIS1']) / 1440
        )

        fluxes = raw_fluxes.copy()

        # Interpolate over the missing measurements which are marked with flux=-99
        interp_fluxes = np.interp(
            times[raw_fluxes == -99], times[raw_fluxes != -99], fluxes[raw_fluxes != -99]
        )

        fluxes[raw_fluxes == -99] = interp_fluxes

        # convert the fluxes into units of ppm with zero mean:
        fluxes = 1e6 * (fluxes / np.median(fluxes) - 1) * ppm
        # error = mad_std(fluxes)

        return LightCurve(
            time=Time(times, format='jd'), flux=fluxes, meta=meta
        )

    else:
        tab = Table.read(download_file(one_year_soho_virgo_url, cache=cache, pkgname='gadfly'))
        times = tab['time'].data

        return LightCurve(
            time=Time(times, format='jd'), flux=tab['flux'] * ppm, meta=meta
        )
