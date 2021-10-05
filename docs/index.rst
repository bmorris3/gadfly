gadfly Documentation
====================

This is the documentation for gadfly, a package for generating photometry of
stars with granulation and p-mode oscillations.

Basic Usage
^^^^^^^^^^^

Let's synthesize a time series of 60-second cadence observations of the Sun
spanning 100 days of observations:

.. code-block:: python

    from gadfly import generate_solar_fluxes
    import astropy.units as u

    time, flux, kernel = generate_solar_fluxes(
        duration=100 * u.day, cadence=60 * u.s
    )

and let's plot the power spectrum of those simulated observations:

.. code-block:: python

    from gadfly.psd import power_spectrum
    import matplotlib.pyplot as plt
    import numpy as np

    freq, power = power_spectrum(flux)

    plt.loglog(
        1e6 * freq,
        kernel.get_psd(2 * np.pi * freq) * (1e6 / 2 / np.pi),
        'r', label='Kernel'
    )
    plt.loglog(1e6 * freq, power, ',k', label='Sim. obs.')
    plt.ylim([1e-7, 1e3])
    plt.legend()
    plt.gca().set(
        xlabel='Freq [$\mu$Hz]', ylabel='Power [ppm$^2$ $\mu$Hz$^{-1}$]'
    )
    plt.show()

.. plot::

    from gadfly import generate_solar_fluxes
    import astropy.units as u
    import numpy as np

    time, flux, kernel = generate_solar_fluxes(
        duration=100 * u.day, cadence=60 * u.s
    )

    from gadfly.psd import power_spectrum
    import matplotlib.pyplot as plt

    freq, power = power_spectrum(flux)

    plt.loglog(
        1e6 * freq,
        kernel.get_psd(2 * np.pi * freq) * (1e6 / 2 / np.pi),
        'r', label='Kernel'
    )
    plt.loglog(1e6 * freq, power, ',k', label='Sim. obs.')
    plt.ylim([1e-7, 1e3])
    plt.legend()
    plt.gca().set(
        xlabel='Freq [$\\mu$Hz]', ylabel='Power [ppm$^2$ $\\mu$Hz$^{-1}$]'
    )
    plt.show()


Tutorials
^^^^^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gadfly/solar.rst
   gadfly/stellar.rst
   gadfly/validation.rst


.. automodapi:: gadfly


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
