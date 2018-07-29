import numpy as np

from cascade.model.compartmental import (
    build_derivative_prevalence, build_derivative_full,
    build_derivative_total, solve_differential_equation, siler_default,
    total_mortality_solution, prevalence_solution, dismod_solution,
    omega_from_mu, mu_from_omega
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
    # Check that the implementations above are vectorized for numpy, because
    # that's what the ODE solver wants.
    f_a = build_derivative_prevalence(iota_a, rho_a, chi_a)
    print(f_a(0.7, np.array([0.01, 0.02, 0.03])))


def test_build_derivative_full():
    # Check that the implementations above are vectorized for numpy, because
    # that's what the ODE solver wants.
    f_a = build_derivative_full(iota_a, rho_a, chi_a, omega_a)
    print(
        f_a(0.7, np.array([[0.01, 0.02, 0.03], [0.01, 0.02, 0.03]])).shape)


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
    f_b = build_derivative_prevalence(iota_b, rho_b, chi_b)
    # This little initial value is a stopgap to keep the difeq positive.
    solutions = solve_differential_equation(f_b, np.array([1e-6]))
    def prevalence(t):
        return solutions(t)[0]

    assert prevalence(50) > 0.01


def test_total_mortality():
    siler = siler_default()
    n_sample = total_mortality_solution(siler)
    t_sample = np.linspace(0, 100, 50)
    total_population = n_sample(t_sample)
    assert len(total_population) == len(t_sample)
    assert all(total_population >= 0)


def from_prevalence_to_dismod(iota, rho, chi, mu):
    S, C, P = prevalence_solution(iota, rho, chi, mu)
    N = lambda t: S(t) + C(t)
    omega = omega_from_mu(mu, chi, P)
    Sf, Cf = dismod_solution(iota, rho, chi, omega)

    error = np.max(np.abs(Cf - C))
    assert error < 1e-3


def from_dismod_to_prevalence(iota, rho, chi, omega):
    Sf, Cf = dismod_solution(iota, rho, chi, omega)
    P = lambda t: Cf(t) / (Sf(t) + Cf(t))
    N = lambda t: Sf(t) + Cf(t)
    mu = mu_from_omega(omega, chi, P)
    S, C, P = prevalence_solution(iota, rho, chi, mu)

    error = np.max(np.abs(Cf - C))
    assert error < 1e-3
