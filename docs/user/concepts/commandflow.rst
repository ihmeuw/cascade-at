
.. _dismod-command-flow:

Flow of Commands in Dismod-AT
-----------------------------

There are a few different ways to use Dismod-AT to examine data.
They correspond to different sequences of
`Dismod-AT commands <https://bradbell.github.io/dismod_at/doc/command.htm>`_.

.. _stream-out-prevalence:

**Stream Out Prevalence** The simplest use of Dismod-AT is to ask it to run the
ordinary differential equation on known
rates and produce prevalence, death, and integrands derived from these.

  1. *Precondition* Provide known values for all rates over the whole
     domain. List the integrands desired for the output.

  2. Run *predict* on those rates.

  3. *Postcondition* Dismod-AT places any requested integrands in
     its predict table. These can be rates, prevalence, death, or
     any of the integrands.

.. _fit-and-predict:

**Simple Fit to a Dataset** This describes a fit with the simplest way to determine
uncertainty.

  1. *Precondition* The input data is observations, with standard deviations,
     of any of the known integrands.

  2. Run *fit* on those observations to produce rates and covariate multipliers.

  3. Run *predict* on the rates to produce integrands.

  4. *Postcondition* Integrands are in the predict table.

.. _fit-asymptotic:

**Fit with Asymptotic Uncertainty** This fit produces some values of uncertainty.

  1. *Precondition* The input data is observations, with standard deviations,
     of any of the known integrands.

  2. Run *fit* on those observations to produce rates and covariate multipliers.

  3. Run *sample asymptotic.*

  4. *Postcondition* Integrands are in the predict table.

.. _fit-simulate:

**Fit with Simulated Uncertainty** This uses multiple predictions in order
to obtain a better estimate of uncertainty.

  1. *Precondition* The input data is observations, with standard deviations,
     of any of the known integrands.

  2. Run *fit* on those observations to produce rates and covariate multipliers.

  3. Run *simulate* to generate simulations of measurements data and priors.

  4. Run *sample simulate.*

  5. *Postcondition* Integrands are in the predict table.

