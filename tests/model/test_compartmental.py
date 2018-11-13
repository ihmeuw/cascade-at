import numpy as np

from scipy.stats import gamma

from cascade.stats import (
    DemographicInterval,
    build_derivative_prevalence,
    build_derivative_full,
    solve_differential_equation,
    siler_default,
    total_mortality_solution,
    prevalence_solution,
    dismod_solution,
    omega_from_mu,
    mu_from_omega,
    average_over_interval,
    integrand_normalization,
    integrands_from_function,
)


def iota_a(t):
    return 0.05


def rho_a(t):
    return 0


def chi_a(t):
    return 0.05


def omega_a(t):
    return 0.2


def test_build_derivative_prevalence():
    """Check that the implementations above are vectorized for numpy, because
    that's what the ODE solver wants."""
    f_a = build_derivative_prevalence(iota_a, rho_a, chi_a)
    assert len(f_a(0.7, np.array([0.01, 0.02, 0.03]))) == 3


def test_build_derivative_full():
    # Check that the implementations above are vectorized for numpy, because
    # that's what the ODE solver wants.
    f_a = build_derivative_full(iota_a, rho_a, chi_a, omega_a)
    shape = f_a(0.7, np.array([[0.01, 0.02, 0.03], [0.01, 0.02, 0.03]])).shape
    assert shape == (2, 3)


def iota_b(t):
    """This incidence is slightly more complicated."""
    if t > 60:
        return 0.002
    elif t > 20:
        return 0.005
    else:
        return 0


def rho_b(t):
    return 0.0005


def chi_b(t):
    return 0.05


def test_solve_differential_equation():
    """Tests integration of the prevalence-only equation"""
    f_b = build_derivative_prevalence(iota_b, rho_b, chi_b)
    # This little initial value is a stopgap to keep the difeq positive.
    solutions = solve_differential_equation(f_b, np.array([1e-6]))

    def prevalence(t):
        return solutions(t)[0]

    assert prevalence(50) > 0.01


def test_total_mortality():
    """Tests integration of the Siler distribution for total mortality"""
    siler = siler_default()
    n_sample = total_mortality_solution(siler)
    t_sample = np.linspace(0, 100, 50)
    total_population = n_sample(t_sample)
    assert len(total_population) == len(t_sample)
    assert all(total_population >= 0)
    assert total_population[0] > total_population[10]


def from_prevalence_to_dismod(iota, rho, chi, mu):
    """
    Tests that the dismod difeq solution matches the prevalence-only.
    Want to know if you coded a differential equation correctly?
    Solve it two ways and see whether they agree.
    """
    S, C, P = prevalence_solution(iota, rho, chi, mu)
    N = lambda t: S(t) + C(t)
    omega = omega_from_mu(mu, chi, P)
    Sf, Cf = dismod_solution(iota, rho, chi, omega)

    condition_error = np.max(np.abs(Cf - C))
    assert condition_error < 1e-3

    total_error = np.max(np.abs(N - (Sf + Cf)))
    assert total_error < 1e-3


def from_dismod_to_prevalence(iota, rho, chi, omega):
    """Tests that the prevalence-only solution matches the dismod solution"""
    Sf, Cf = dismod_solution(iota, rho, chi, omega)
    P = lambda t: Cf(t) / (Sf(t) + Cf(t))
    N = lambda t: Sf(t) + Cf(t)
    mu = mu_from_omega(omega, chi, P)
    S, C, P = prevalence_solution(iota, rho, chi, mu)

    condition_error = np.max(np.abs(Cf - C))
    assert condition_error < 1e-3

    total_error = np.max(np.abs(N - (S + C)))
    assert total_error < 1e-3


def test_average_over_interval_constant():
    # ten-year intervals to age 100.
    nt = DemographicInterval(np.full((10,), 10.0, dtype=np.float))

    def raw_rate(t):
        return 2.0

    def weight_function(t):
        return 3.0

    averaged_rate = average_over_interval(raw_rate, weight_function, nt)
    assert np.allclose(averaged_rate, 60.0, atol=1e-5)
    averaged_weight = average_over_interval(weight_function, raw_rate, nt)
    assert np.allclose(averaged_weight, 60.0, atol=1e-5)


def test_normalized_over_interval():
    # ten-year intervals to age 100.
    nt = DemographicInterval(np.full((10,), 10.0, dtype=np.float))

    def raw_rate(t):
        return 2.0

    def weight_function(t):
        return 3.0

    normalization = integrand_normalization(weight_function, nt)
    averaged_rate = average_over_interval(raw_rate, weight_function, nt)
    assert np.allclose(averaged_rate / normalization, 2.0, atol=1e-5)

    reversed = integrand_normalization(raw_rate, nt)
    averaged_weight = average_over_interval(weight_function, raw_rate, nt)
    assert np.allclose(averaged_weight / reversed, 3.0, atol=1e-5)


def test_integrand_downslope_exact():
    """
    Tests weighting of integrands given a weighting function that is linear
    and a piecewise-constant rate to weight. The result should have the
    same value as the input weight.
    """
    # ten-year intervals to age 100.
    nt = DemographicInterval(np.full((10,), 10.0, dtype=np.float))

    def raw_rate(t):
        if t < 50:
            return 2.0
        else:
            return 3.0

    def weight_function(t):
        return 1.0 - t / 140.0

    normalization = integrand_normalization(weight_function, nt)
    averaged_rate = average_over_interval(raw_rate, weight_function, nt)
    result = averaged_rate / normalization
    assert np.allclose(result[:5], 2.0, atol=1e-5)
    assert np.allclose(result[5:], 3.0, atol=1e-5)


def test_integrand_continuous_exact():
    """
    Tests weighting of integrands given a weighting function that is
    integrable. Choose :math:`t_0=120` and the function to be

    .. math::

        r(t) = t/t_0

    Then make the weight

    .. math::

        w(t) = e^{-t/t_0}

    Then the resulting integrand should be

    .. math::

        \frac{(a+t_0) e^{-a/t_0} - (b+t_0)e^{-b/t_0}}
        {t_0(e^{-a/t_0} - e^{-b/t_0})}

    which is exact.
    """
    # ten-year intervals to age 100.
    nt = DemographicInterval(np.full((10,), 10.0, dtype=np.float))
    t0 = 120

    def raw_rate(t):
        return t / t0

    def weight_function(t):
        return np.exp(-(t / t0) ** 2)

    integrands, normalization = integrands_from_function([raw_rate], weight_function, nt)
    result = integrands[0]

    expected = np.zeros_like(result)
    for interval_idx in range(len(nt)):
        start = nt.start[interval_idx]
        finish = nt.finish[interval_idx]
        numerator = (start + t0) * np.exp(-(start / t0)) - (finish + t0) * np.exp(-(finish / t0))
        denominator = t0 * (np.exp(-start / t0) - np.exp(-finish / t0))
        expected[interval_idx] = numerator / denominator

    tolerance_due_to_interpolated_functions = 0.015
    assert np.allclose(result, expected, atol=tolerance_due_to_interpolated_functions)


def test_make_test_integrands():
    """
    Make up some distributions and turn them into integrands for Dismod.
    """

    def diabetes_incidence(x):
        return 0.8 * gamma(a=9, scale=7).pdf(x)

    def diabetes_emr(x):
        return 0.015 * (np.exp((x / 100) ** 2) - 1)

    def diabetes_remission(x):
        return np.zeros_like(x)

    total_mortality = siler_default()

    S, C, P = prevalence_solution(diabetes_incidence, diabetes_remission, diabetes_emr, total_mortality)

    def lx(t):
        return S(t) + C(t)

    rates, norm = integrands_from_function(
        [diabetes_incidence, C, diabetes_emr], lx, DemographicInterval(np.full((10,), 10.0, dtype=np.float))
    )
    assert len(rates) == 3
