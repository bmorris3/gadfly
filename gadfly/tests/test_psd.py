import pytest
import numpy as np

from ..psd import (
    linear_space_to_jax_parameterization,
    jax_parameterization_to_linear_space,
)


def _test_hyperparams(n_tests=5, n_terms=5):
    np.random.seed(42)
    test_hyperparams = []
    for i in range(n_tests):
        test_S0s = np.sort(3 * 10 ** np.random.uniform(-1, 4, n_terms))[::-1]
        test_omegas = np.sort(10 ** np.random.uniform(1, 4, n_terms))
        test_hyperparams.append([test_S0s, test_omegas])
    return test_hyperparams


@pytest.mark.parametrize("test_S0s, test_omegas", _test_hyperparams())
def test_reparameterization(test_S0s, test_omegas):
    # test round-tripping the parameter conversions. This will pass
    # silently if the round-trip works as expected.

    recon_S0s, recon_omegas = jax_parameterization_to_linear_space(
        linear_space_to_jax_parameterization(
            test_S0s, test_omegas
        )
    )

    np.testing.assert_allclose(test_S0s, recon_S0s)
    np.testing.assert_allclose(test_omegas, recon_omegas)
