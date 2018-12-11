"""
Statistical operations on the model.
"""


def estimate_priors_from_posterior_draws(draws, model_context, execution_context):
    r"""
    Given a dataframe of sampled outcomes from a previous run of DismodAT,
    find parameters for the given distributions. This works from the model
    context. The draws are a representation of :math:`p(\theta, u|y,\eta)`
    where :math:`\eta` are the parameters for the prior distributions on
    :math:`\theta`, the model variables, :math:`y` are the data,
    and :math:`u` are the random effects. The goal of this function is to
    calculate the likelihood of the parameters for the prior distribution
    of the next step down in the cascade,

    .. math::

        p(\eta|y) = \int p(\eta|\theta) p(\theta|y) d\theta

    and use that to generate one value of the most likely priors,
    :math:`\eta^*` that will initialize the next step down.

    The priors in this problem are on

     * the Markov Random Field (MRF) for each rate.
     * the MRF for the covariates, :math:`(\alpha,\beta,\gamma)`
     * the MRF for child effects

    Each MRF includes hyper-priors :math:`\lambda` on the standard deviations,
    and most of the priors are the value priors, age differences, and
    time differences.

    Args:
        draws (pd.DataFrame): Has ``fit_var_id``, ``fit_var_value``,
            ``residual_value``, ``residual_dage``, ``residual_dtime``,
            ``sample_index``. Where the residuals can be NaN. The
            zeroth sample is the initial fit, aka the MAP estimate.
            The other samples are samples around that fit.

        model_context: The Model.

        execution_context: Where to find the Dismod File.

    Returns:
        pd.DataFrame: With parameters for each distribution.
    """

    return draws
