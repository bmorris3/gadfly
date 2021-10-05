Validation
==========

Let's check that the prescription for the granulation and p-mode oscillations
scales correctly for solar twins using short-cadence observations from Kepler.

.. plot::
    :include-source:

    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import binned_statistic
    import matplotlib.pyplot as plt

    import astropy.units as u
    from astropy.table import Table
    from astropy.time import Time

    from lightkurve import search_lightcurve, LightCurve, LightCurveCollection

    from gadfly import generate_solar_fluxes, generate_stellar_fluxes
    from gadfly.psd import power_spectrum
    from gadfly.interp import interpolate_missing_data

    # Store a table of stellar parameters from Mather et al. (2012), Gaia (2018)
    stars = Table.read(
        """Star & $T_\mathrm{eff}$ [K] & $M$ [$M_\odot$] & $L$ [$L_\odot$] &  $R$ [$R_\odot$] & $K_P$
        KIC 3656476 & 5581 & 1.09 & 1.627 & 1.36 & 9.516
        KIC 5184732 & 5766 & 1.25 & 1.952 & 1.40 & 8.165
        KIC 6116048 & 5980 & 1.12 & 1.851 & 1.27 & 8.418
        KIC 6106415 & 6034 & 1.12 & 1.897 & 1.26 & 7.179""",
        format='ascii.csv', delimiter='&'
    )

    fig, axes = plt.subplots(4, 2, figsize=(10, 20), dpi=200)

    for i, row in enumerate(stars):
        ax = axes[i, :]

        # Fetch the Kepler short-cadence light curve
        lcs = search_lightcurve(
            row['Star'], cadence='short', mission='Kepler'
        ).download_all()

        # Interpolate over missing data points in each quarter, normalize by a
        # fifth order polynomial to remove systematic trends
        lcs_interped = []
        for lc in lcs:
            lc = lc.normalize().remove_nans()

            t, f = interpolate_missing_data(lc.time.jd, lc.flux)
            e = np.median(lc.flux_err) * np.ones_like(f)

            fit = np.polyval(np.polyfit(t - t.mean(), f, 5),
                             t - t.mean())

            lcs_interped.append(LightCurve(time=t, flux=f/fit, flux_err=e.value))

        # Stitch together all quarters, interpolate again
        lc = LightCurveCollection(lcs_interped).stitch()
        t, f = interpolate_missing_data(lc.time.jd, lc.flux.value)

        # Generate a stellar oscillation and p-mode kernel for the correct
        # stellar parameters
        tm, fm, kernel = generate_stellar_fluxes(
            duration=1*u.day,
            M=row['$M$ [$M_\odot$]'] * u.M_sun,
            T_eff=row['$T_\mathrm{eff}$ [K]'] * u.K,
            R=row['$R$ [$R_\odot$]'] * u.R_sun,
            L=row['$L$ [$L_\odot$]'] * u.L_sun
        )

        # Compute the power spectrum of the observations
        freq, power = power_spectrum(f)

        # Bin the power spectrum of the observations
        left_bs = binned_statistic(1e6 * freq, power, statistic=np.nanmedian,
                                   bins=np.logspace(-2, 4, 100))
        bincenters = 0.5 * (left_bs.bin_edges[1:] + left_bs.bin_edges[:-1])

        # plot the full power spectrum
        ax[0].loglog(1e6 * freq, power, ',', color='silver', rasterized=True)
        ax[0].loglog(1e6 * freq, kernel.get_psd(2*np.pi*freq)*1e6/2/np.pi, 'r', rasterized=True)
        ax[0].loglog(bincenters, left_bs.statistic, 'k', rasterized=True)
        ax[0].set(
            ylim=[1e-4, 1e4]
        )

        # plot the p-mode oscillations
        ax[1].semilogy(1e6 * freq, power, ',', color='silver', rasterized=True)
        ax[1].semilogy(1e6 * freq, kernel.get_psd(2*np.pi*freq)*1e6/2/np.pi, 'r', rasterized=True)
        ax[1].semilogy(1e6 * freq[1:], gaussian_filter1d(power[1:], 100), color='k', rasterized=True)
        ax[1].set(
            xlim=[1e3, 3e3], ylim=[3e-3, 0.3]
        )

        ax[1].set_xlabel('Frequency [$\mu$Hz]')
        ax[1].set_ylabel('Power Density [ppm$^2$/$\mu$Hz]')
        ax[0].set_xlabel('Frequency [$\mu$Hz]')
        ax[0].set_ylabel('Power Density [ppm$^2$/$\mu$Hz]')

        ax[0].set_title(row['Star'])
        ax[1].set_title(row['Star'])

        for s in ['right', 'top']:
            for axis in ax:
                axis.spines[s].set_visible(False)

    fig.tight_layout()
    plt.show()