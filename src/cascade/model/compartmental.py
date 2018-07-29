"""
DismodAT Differential Equation
------------------------------

These functions manipulate data for the compartmental model
that the differential equation describes. Using
:math:`S` as without this condition and alive, :math:`C` as with
condition, and :math:`R` as removed, or dead,

.. math::

    \frac{dS}{d\tau} = -\iota S + \rho C -\omega S

    \frac{dC}{d\tau} = \iota S - \rho C - (\omega + \chi) C

    \frac{dR}{d\tau} = \omega (S+C) + \chi C.

The time is cohort time, which we denote as :math:`\tau`.

These functions work with continuous functions. They are designed for
creating test data and analyzing interpolated data.

.. _prevalence-only:

Prevalence-Only Equation
------------------------

The prevalence-only form of the ODE.
The full differential equation can be transformed into a space

.. math::

    P = \frac{C}{S+C}

    N = S + C.

to yield two differential equations, one for the prevalence and
one for the total mortality

.. math::

    P' = \iota (1-P) - \rho P - \chi (1-P) P

    N' = -\omega N - \chi C.

The :math:`N` variable doesn't appear in the first equation, so it is
independent and can be solved alone. Then the second equation is
equivalent to total mortality

.. math::

    N' = -\mu N

which indicates that :math:`\mu = \omega + \chi P`.

"""
import numpy as np
from scipy.integrate import solve_ivp


def build_derivative_prevalence(iota, rho, chi):
    """
    Given three functions for the basic rates, this creates a function
    that lets the ODE solver integrate the prevalence-only differential
    equation.

    Args:
        iota (function): Incidence rate
        rho (function): Remission rate
        chi (function): Excess mortality rate

    Returns:
        function: The arguments are time and a sequence of :math:`N` prevalence
            states, given as a :math:`(1, N)` array.
    """

    def ode_right_hand_side(t, y):
        return iota(t) * (1 - y) - rho(t) * y - chi(t) * (1 - y) * y

    return ode_right_hand_side


def build_derivative_total(mu):
    """
    Turns a mortality rate into an argument for the ODE solver.

    Args:
        mu (function): Total mortality rate

    Returns:
        function: The arguments are time and a sequence of :math:`N` prevalence
            states, given as a :math:`(1, N)` array.
    """

    def ode_right_hand_side(t, y):
        return -mu(t) * y

    return ode_right_hand_side


def build_derivative_full(iota, rho, chi, omega):
    """
    The Dismod-AT ODE

    Args:
        iota (function): Incidence rate
        rho (function): Remission rate
        chi (function): Excess mortality rate
        omega (function): Other-cause mortality

    Returns:
        function: The arguments are time and a sequence of :math:`N` prevalence
            states, given as a :math:`(2, N)` array.
    """

    def ode_right_hand_side(t, y):
        sprime = -(iota(t) + omega(t)) * y[0, :] + rho(t) * y[1, :]
        cprime = iota(t) * y[0, :] - (rho(t) + omega(t) + chi(t)) * y[1, :]
        return np.vstack([sprime, cprime])

    return ode_right_hand_side


def omega_from_mu(mu, chi, P):
    """
    Given functions for :math:`(\mu, \chi, P)`, return a function for
    :math:`\omega`.

    Args:
        mu (function): Total mortality rate.
        chi (function): Excess mortality rate.
        P (function): Prevalence.

    Returns:
        function: Other-cause mortality.
    """
    def omega(t):
        return mu(t) - chi(t) * P(t)
    return omega


def mu_from_omega(omega, chi, P):
    """
    Given :math:`(\omega, \chi, P)`, return a function for total
    mortality, :math:`\mu`.

    Args:
        omega (function): Other-cause mortality
        chi (function): Excess mortaltiy.
        P (function): Prevalence.

    Returns:
        function: Total mortality rate.
    """
    def total_mortality(t):
        return omega(t) + chi(t) * P(t)
    return total_mortality


def solve_differential_equation(f_derivatives, initial, oldest=120):
    """
    Solve differential equations between ages 0 and 100.
    Uses ``numpy.integrate.solve_ivp`` underneath.

    Args:
        f_derivatives (function): A function that returns first derivatives
             of the differential equation.
        initial (np.array): A numpy array of initial values. Must be
            the same dimension as the returned by f_derivatives.
        oldest (float): Upper limit of integration. For instance, 100.

    Returns:
        Array of interpolation functions, of same length as input function's
        return values.
    """
    bunch = solve_ivp(
        f_derivatives,
        t_span=(0, oldest),
        y0=initial,
        vectorized=True,
        dense_output=True,
        )
    return bunch.sol


SILER_CONSTANTS = [0, 0.2, 0.0002, 0.003, 1, 0.1, 0.015, 0.01]


def siler_default():
    """
    Construct a total mortality rate using the Siler distribution
    and default constants.
    """
    return siler_time_dependent_hazard(SILER_CONSTANTS)


def siler_time_dependent_hazard(constants):
    """
    This Siler distribution is a good approximation to what a real total
    mortality rate looks like. Both the equations and the parameters come
    from a paper [1] where they were fit to a Scandinavian country.
    We will use this as the one true mortality rate for this session.

    [1] V. Canudas-Romo and R. Schoen, “Age-specific contributions to
    changes in the period and cohort life expectancy,” Demogr. Res.,
    vol. 13, pp. 63–82, 2005.

    Args:
        contants (np.array): List of constants. The first is time because
            this function can model change in a total mortality distribution
            over time.

    Returns:
        A function that returns mortality rate as a function of age.
    """
    t, a1, a2, a3, b1, b2, c1, c2 = constants

    def siler(x):
        return (a1 * np.exp(-b1 * x - c1 * t) +
                a2 * np.exp(b2 * x - c2 * t) +
                a3 * np.exp(-c2 * t))
    return siler


def total_mortality_solution(mu):
    """Given a total mortality rate, as a function, return :math:`N=l(x)`."""
    n_array = solve_differential_equation(build_derivative_total(mu),
                          initial=np.array([1.0], dtype=float))

    def total_pop(t):
        val = n_array(t)[0]
        val[val < 0] = 0
        return val

    return total_pop


def prevalence_solution(iota, rho, chi, mu):
    """This uses the single, prevalence-based equation."""
    N = total_mortality_solution(mu)
    f_b = build_derivative_prevalence(iota, rho, chi)
    bunch = solve_differential_equation(f_b, initial=np.array([1e-6]))
    P = lambda t: bunch(t)[0]
    C = lambda t: P(t) * N(t)
    S = lambda t: (1 - P(t)) * N(t)

    return S, C, P


def dismod_solution(iota, rho, chi, omega):
    """This solves the Dismod-AT equations."""
    f_b = build_derivative_full(iota, rho, chi, omega)
    bunch = solve_differential_equation(
        f_b, initial=np.array([1.0-1e-6, 1e-6], dtype=np.float))
    S = lambda t: bunch(t)[0]
    C = lambda t: bunch(t)[1]
    return S, C
