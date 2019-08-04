.. _epiviz-global-model:

Global Model
============

Dismod-AT solved the Illness-Death model in age and time for a hierarchical
model with multiple locations. The Cascade ties these together into a single
estimation for the world. Precedent for this comes from before the
Dismod ii work [Barendregt2003]_, back to [ClaytonKaldor1987]_.


.. _global-model-draft:

Draft Description
-----------------

A draft description of the global model is in a PDF called the
:download:`Global Illness Death Model <./Global_Illness_Death_Model.pdf>`.


Setting Priors
--------------

The global model uses estimates of aggregated data in order to inform
estimates at lower levels. We can think of this as Empirical Bayesian
([Carlin2000]_, [Kass1989]_) but Bobby Reiner suggests understanding it
as bootstrapping.

Two articles discuss how to estimate confidence intervals from maximum a-posteriori
methods, [Pereyra2016]_, [Pereyra2017]_.

Implementation in the code is discussed in a few different areas of documentation:

 *  :ref:`posteriors-to-priors` - Mathematics of setting priors from fits.

 *  :ref:`epiviz-priors-input` - Description of inputs to each estimation that affect priors.

 *  :ref:`epiviz-posteriors` - Outputs from each estimation.


.. [ClaytonKaldor1987] David Clayton and John Kaldor. `*Empirical Bayes Estimates
   of Age-Standardized Relative Risks for Use in Disease Mapping.* <https://www.jstor.org/stable/pdf/2532003.pdf>`_
   Biometrics, 43(3):671–681, 1987.

.. [Barendregt2003] Jan J Barendregt, Gerrit J Van Oortmarssen, Theo Vos, and
   Christopher JL Murray. `*A generic model for the assessment of disease
   epidemiology: the computational basis of dismod ii.* <https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-1-4>`_
   Population health metrics, 1(1):4, 2003.

.. [Pereyra2017] Marcelo Pereyra. `*Maximum-a-posteriori estimation with bayesian
   confidence regions.* <https://epubs.siam.org/doi/pdf/10.1137/16M1071249>`_
   SIAM Journal on Imaging Sciences, 10(1):285–302, 2017.

.. [Pereyra2016] Marcelo Pereyra, Philip Schniter, Emilie Chouzenoux,
   Jean-Christophe Pesquet, Jean-Yves Tourneret, Alfred O Hero, and Steve McLaughlin.
   `*A survey of stochastic simulation and optimization methods in signal processing.*
   <https://ieeexplore.ieee.org/iel7/4200690/5418892/07314898.pdf>`_
   IEEE Journal of Selected Topics in Signal Processing,10(2):224–241, 2016.

.. [Carlin2000] Bradley P Carlin and Thomas A Louis. *Empirical bayes: Past,
   present and future.* Journal of the American Statistical Association, 95(452):1286–1289, 2000.
   The `most-recommended <https://statmodeling.stat.columbia.edu/2008/11/08/carlin_and_loui/>`_
   on empirical Bayes.

.. [Kass1989] Robert E Kass and Duane Steffey. `*Approximate bayesian inference
   in conditionally independent hierarchical models (parametric empirical
   bayes models).* <https://www.jstor.org/stable/pdf/2289653.pdf>`_
   Journal of the American StatisticalAssociation, 84(407):717–726, 1989.
