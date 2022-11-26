
import numpy as np

import astropy.units as u
from astropy.units import cds  # noqa
from astropy.time import Time

from celerite2 import GaussianProcess as CeleriteGaussianProcess

__all__ = ['GaussianProcess']


class GaussianProcess(CeleriteGaussianProcess):
    """
    The ``gadfly`` interface to the celerite2 Gaussian Process (GP) solver.

    This subclass of the :py:class:`~celerite2.GaussianProcess` class in
    ``celerite2`` knows how to handle :py:mod:`~astropy.units`, for ease of use
    in ``gadfly``.
    """

    def __init__(self, kernel, t=None, mean=0.0, light_curve=None, **kwargs):
        """
        Parameters
        ----------
        kernel : Subclass of :py:class:`~celerite2.terms.Term`
            ``celerite2``-compatible kernel. Could be a :py:class:`~gadfly.SolarOscillatorKernel`
            or :py:class:`~gadfly.StellarOscillatorKernel`, for example.
        t : ~astropy.units.Quantity or ~astropy.time.Time
        mean : float or callable
            The mean function of the process. This can either
            be a callable (it will be evaluated with a single argument, a
            vector of ``x`` values) or a scalar. (default: ``0.0``)
        light_curve : ~lightkurve.LightCurve
            The light curve on which predictions will be computed
        kwargs : dict
            Other arguments will be passed directly to
            :py:meth:`~gadfly.GaussianProcess.compute` if the argument ``t`` is
            specified.
        """
        self._original_flux_median = None

        if t is not None:
            # convert the times into units compatible with the gadfly parameterization:
            t = self._time_to_freq(t)

        if light_curve is not None:
            t = self._time_to_freq(light_curve.time)
            self._original_flux_median = np.median(light_curve.flux)
            kwargs['yerr'] = self._flux_to_ppm(light_curve.flux_err, is_error=True)

        super().__init__(kernel, t=t, mean=mean, **kwargs)

    @staticmethod
    def _time_to_freq(time, freq_unit=u.uHz):
        """
        Convert times into units compatible with the parameterization
        of kernel hyperparameters in gadfly.

        Parameters
        ----------
        time : ~astropy.units.Quantity, ~astropy.time.Time, or ~np.ndarray
            Time quantity to be converted, or a numpy array (returned as-is).
        freq_unit : ~astropy.units.Quantity
            Frequency unit. ``time`` will be converted into units of the inverse
            of ``freq_unit``.
        Returns
        -------
        time_value : ~numpy.ndarray
            Times in units of inverse of ``freq_unit``
        """
        if isinstance(time, Time):
            time = time.jd * u.day

        if not hasattr(time, 'unit'):
            # assume time is already in the correct units:
            return time

        return time.to(1 / freq_unit).value

    def _flux_to_ppm(self, flux, flux_unit=u.cds.ppm, is_error=False):
        """
        Convert fluxes into units compatible with the parameterization
        of kernel hyperparameters in gadfly.

        Parameters
        ----------
        flux : ~astropy.units.Quantity or ~np.ndarray
            Must be either: (1) a convertible flux quantity or
            (2) a numpy array already in units of ppm (in this case,
            the method will simply return the input).
        flux_unit : ~astropy.units.Quantity
            Flux unit. ``flux`` will be converted into units of
            ``flux_unit``.
        is_error : bool
            Set to true if converting a unitful flux uncertainty to
            its equivalent in [ppm].

        Returns
        -------
        time_value : ~numpy.ndarray
            Times in units of inverse of ``freq_unit``
        """
        if isinstance(flux, np.ndarray) and not hasattr(flux, 'unit'):
            # assume already in [ppm] if given an ndarray
            return flux

        elif hasattr(flux, 'unit') and flux.unit.is_equivalent(u.electron / u.s):
            # if given lightkurve.LightCurve.flux, which is units of e-/s,
            # normalize the light curve into the correct units.
            # Cache the median so we can reconstruct the original units
            # on a call to `_ppm_to_flux` later:
            if is_error:
                flux_ppm = 1e6 * (flux / self._original_flux_median).value
            else:
                flux_ppm = 1e6 * (flux / self._original_flux_median - 1).value
            return flux_ppm

        return flux.to(flux_unit).value

    def _ppm_to_flux(self, value_in_ppm, power=1):
        """
        Convert ~np.ndarray fluxes into unitful quantities.

        Parameters
        ----------
        value_in_ppm :
            Must be either: (1) a convertible flux quantity or
            (2) a numpy array already in units of ppm (in this case,
            the method will simply return the input).
        unit : ~astropy.units.Quantity
            Flux unit. ``flux`` will be returned in units of
            ``flux_unit``.
        Returns
        -------
        flux_ppm : ~astropy.units.Quantity
            Fluxes in units of [ppm]
        """
        if self._original_flux_median is not None and power == 1:
            # this is a flux value:
            value_in_original_units = (
                (1e-6 * value_in_ppm + 1) *
                self._original_flux_median
            )
            return value_in_original_units

        elif self._original_flux_median is not None and power == 2:
            # this is a variance:
            value_in_original_units = (
                (1e-6 * value_in_ppm) *
                self._original_flux_median *
                self._original_flux_median.unit
            )
            return value_in_original_units

        else:
            # otherwise return the flux with units of ppm:
            return u.Quantity(value_in_ppm, unit=u.cds.ppm)

    def compute(
        self, t, yerr=None, diag=None, check_sorted=True, quiet=False
    ):
        """
        Compute the Cholesky factorization of the GP covariance matrix.

        Parameters
        ----------
        t : ~astropy.units.Quantity or ~astropy.time.Time
            The independent coordinates of the observations.
            This must be sorted in increasing order.
        yerr : ~astropy.units.Quantity
            If provided, the diagonal standard
            deviation of the observation model.
        diag : ~astropy.units.Quantity
            If provided, the diagonal variance of
            the observation model.
        check_sorted : bool
            If ``True``, a check is performed to
            make sure that ``t`` is correctly sorted. A ``ValueError`` will
            be thrown when this check fails.
        quiet : bool
            If ``True``, when the matrix cannot be
            factorized (because of numerics or otherwise) the solver's
            ``LinAlgError`` will be silenced and the determiniant will be
            set to zero. Otherwise, the exception will be propagated.
        """
        if isinstance(t, Time) or hasattr(t, 'unit'):
            t = self._time_to_freq(t)

        if yerr is not None and hasattr(yerr, 'unit'):
            yerr = self._flux_to_ppm(yerr, is_error=True)

        if diag is not None and hasattr(yerr, 'unit'):
            diag = self._flux_to_ppm(diag)
        super().compute(
            t, yerr=yerr, diag=diag, check_sorted=check_sorted, quiet=quiet
        )

    def condition(
            self, y, t=None, include_mean=True, kernel=None, return_quantity=False
    ):
        """
        Condition the Gaussian process given observations ``y``.

        Parameters
        ----------
        y : ~astropy.units.Quantity
            Observations
        t : ~astropy.units.Quantity or ~astropy.time.Time
            Times
        include_mean : bool
            Include the mean model in the prediction
        kernel : subclass of :py:class:`~celerite2.terms.Term`
            Evaluate conditional distribution given this kernel
        return_quantity : bool
            Return the fluxes as a :py:class:`~astropy.units.Quantity` in the same units as
            the input light curve.
        """
        if t is not None and (isinstance(t, Time) or hasattr(t, 'unit')):
            t = self._time_to_freq(t)

        if hasattr(y, 'unit'):
            y = self._flux_to_ppm(y)

        result = self.conditional_distribution(
            self, y, t=t, include_mean=include_mean, kernel=kernel
        )

        if return_quantity:
            return self._ppm_to_flux(result.mean)

        return result

    def predict(
        self,
        y,
        t=None,
        return_cov=False,
        return_var=False,
        include_mean=True,
        kernel=None,
        return_quantity=False
    ):
        """
        Compute the conditional distribution

        The factorized matrix from the previous call to
        :py:meth:`~gadfly.GaussianProcess.compute` is used, so that
        method must be called first.

        Parameters
        ----------
        y (shape[N]) : ~astropy.units.Quantity
            The observations at coordinates ``t`` as defined by
            :py:meth:`gadfly.GaussianProcess.compute`.
        t (shape[M]) : ~astropy.units.Quantity or ~astropy.time.Time
            The independent coordinates where the
            prediction should be evaluated. If not provided, this will be
            evaluated at the observations ``t`` from
            :py:meth:`~gadfly.GaussianProcess.compute`.
        return_var : bool
            Return the variance of the conditional
            distribution.
        return_cov : bool
            Return the full covariance matrix of
            the conditional distribution.
        include_mean : bool
            Include the mean function in the
            prediction.
        kernel : subclass of :py:class:`~celerite2.terms.Term`
            If provided, compute the conditional
            distribution using a different kernel. This is generally used
            to separate the contributions from different model components.
            Note that the computational cost and scaling will be worse
            when using this parameter.
        return_quantity : bool
            Return the fluxes as a :py:class:`~astropy.units.Quantity` in the same units as
            the input light curve.
        """

        if hasattr(y, 'unit'):
            y = self._flux_to_ppm(y)
        if isinstance(t, Time) or hasattr(t, 'unit'):
            t = self._time_to_freq(t)

        cond = self.condition(y, t=t, include_mean=include_mean, kernel=kernel)

        if return_var and return_quantity:
            return self._ppm_to_flux(cond.mean), self._ppm_to_flux(cond.variance, power=2)
        elif return_cov and return_quantity:
            return self._ppm_to_flux(cond.mean), self._ppm_to_flux(cond.covariance, power=2)
        elif return_quantity:
            return self._ppm_to_flux(cond.mean)
        elif return_var:
            return cond.mean, cond.variance
        elif return_cov:
            return cond.mean, cond.covariance

        return cond.mean

    def dot_tril(self, y, *, inplace=False):
        """
        Dot the Cholesky factor of the GP system into a vector or matrix

        Compute ``x = L.y`` where ``K = L.L^T`` and ``K`` is the covariance
        matrix of the GP.

        .. note:: The mean function is not applied in this method.

        Parameters
        ----------
        y : ~astropy.units.Quantity
            The vector or matrix ``y`` described above.
        inplace : bool
            If ``True``, ``y`` will be overwritten
            with the result ``x``.
        """
        if hasattr(y, 'unit'):
            y = self._flux_to_ppm(y)
        return super().dot_tril(y, inplace=inplace)

    def log_likelihood(self, y, *, inplace=False):
        """
        Compute the marginalized likelihood of the GP model

        The factorized matrix from the previous call to
        :py:meth:`~gadfly.GaussianProcess.compute`
        is used so that method must be called first.

        Parameters
        ----------
        y (shape[N]) : ~astropy.units.Quantity
            The observations at coordinates ``t`` as defined by
            :py:meth:`~gadfly.GaussianProcess.compute`.
        inplace : bool
            If ``True``, ``y`` will be overwritten
            in the process of the calculation. This will reduce the memory
            footprint, but should be used with care since this will
            overwrite the data.
        """
        if hasattr(y, 'unit'):
            y = self._flux_to_ppm(y)
        return super().log_likelihood(y, inplace=inplace)

    def apply_inverse(self, y, *, inplace=False):
        """
        Apply the inverse of the covariance matrix to a vector or matrix

        Solve ``K.x = y`` for ``x`` where ``K`` is the covariance matrix of
        the GP.

        .. note:: The mean function is not applied in this method.

        Parameters
        ----------
        y : ~astropy.units.Quantity
            The vector or matrix ``y`` described above.
        inplace : bool
            If ``True``, ``y`` will be overwritten with the result ``x``.
        """
        if hasattr(y, 'unit'):
            y = self._flux_to_ppm(y)
        return super().apply_inverse(y, inplace=inplace)

    def sample(self, *, size=None, include_mean=True, return_quantity=False):
        """
        Generate random samples from the prior implied by the GP system

        The factorized matrix from the previous call to
        :py:meth:`~gadfly.GaussianProcess.compute` is used so that
        method must be called first.

        Parameters
        ----------
        size : int
            The number of samples to generate. If not
            provided, only one sample will be produced.
        include_mean : bool
            Include the mean function in the prediction.
        return_quantity : bool
            Return the fluxes as a :py:class:`~astropy.units.Quantity` in the same units as
            the input light curve.
        """
        result = super().sample(size=size, include_mean=include_mean)
        result -= result.mean()
        if return_quantity:
            return self._ppm_to_flux(result)
        return result
