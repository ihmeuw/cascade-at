===============
Dismod-AT Shape
===============

Introduction
============

The main source of disease data for Dismod-AT is a set of measurements
called a *bundle.* Secondary sources include all-cause mortality,
cause-specific mortality, and other population data from IHME databases.
Modelers use a tool called the EpiUploader to place new records into the
bundle and associate each of those records with study covariates.

This document describes requirements on the *shape* of the bundle. The
shape is the number and type of columns in a bundle, including
preconditions on those columns. These aren't requirements of Dismod-AT
itself but requirements of Cascade, a Python program that runs Dismod-AT
within IHME. Central Comp really wants to know what shape Dismod-AT
needs. That's what this document is for.

There are Hub pages about the Epi Uploader that describe the incoming data.

* https://hub.ihme.washington.edu/pages/viewpage.action?pageId=18685489
* https://hub.ihme.washington.edu/pages/viewpage.action?spaceKey=SR&title=age_start+age_end

Shape of the Bundle
===================

Column by Column
----------------

This is how uploaded data is presented to the Cascade. Which columns
should be used to transform input data?

+------------------------------+--------+---------+----------------------------------------------------------------+
| Column                       | Used   | Type    | Comments                                                       |
+==============================+========+=========+================================================================+
| bundle\_id                   | Yes    | Int     | Matches existing bundle                                        |
+------------------------------+--------+---------+----------------------------------------------------------------+
| seq                          | Yes    | Int     | Unique for record within bundle.                               |
+------------------------------+--------+---------+----------------------------------------------------------------+
| request\_id                  | No     |         |                                                                |
+------------------------------+--------+---------+----------------------------------------------------------------+
| input\_type\_id              | Yes    | Int     | Excludes types 5, 6                                            |
+------------------------------+--------+---------+----------------------------------------------------------------+
| nid                          | No     |         |                                                                |
+------------------------------+--------+---------+----------------------------------------------------------------+
| underlying\_nid              | No     |         |                                                                |
+------------------------------+--------+---------+----------------------------------------------------------------+
| location\_id                 | Yes    | Int     | Must be in given location set ID (35)                          |
+------------------------------+--------+---------+----------------------------------------------------------------+
| sex\_id                      | Yes    | Int     | 1, 2, 3, 4.                                                    |
+------------------------------+--------+---------+----------------------------------------------------------------+
| year\_start                  | Yes    | Float   | This code is OK if it isn't an integer.                        |
+------------------------------+--------+---------+----------------------------------------------------------------+
| year\_end                    | Yes    | Float   | Again, fine if not an integer, >= year\_start                  |
+------------------------------+--------+---------+----------------------------------------------------------------+
| age\_start                   | Yes    | Float   | >= 0.0                                                         |
+------------------------------+--------+---------+----------------------------------------------------------------+
| age\_end                     | Yes    | Float   | >= age\_start                                                  |
+------------------------------+--------+---------+----------------------------------------------------------------+
| age\_demographer             | Yes    | Int     | This is missing from the bundle now. 0 or 1.                   |
+------------------------------+--------+---------+----------------------------------------------------------------+
| measure\_id                  | Yes    | Int     | one of the measures in the list below                          |
+------------------------------+--------+---------+----------------------------------------------------------------+
| source\_type\_id             | No     | Int     | not trusted                                                    |
+------------------------------+--------+---------+----------------------------------------------------------------+
| sampling\_type\_id           | No     | Int     | cluster, multistage, nonprobability, simple random             |
+------------------------------+--------+---------+----------------------------------------------------------------+
| representative\_id           | No     | Int     | Urban, rural, subnational                                      |
+------------------------------+--------+---------+----------------------------------------------------------------+
| urbanicity\_type\_id         | Yes    | Int     | Whether urban, rural, suburban, peri-urban                     |
+------------------------------+--------+---------+----------------------------------------------------------------+
| recall\_type\_id             | No     | Int     | Point, Lifetime, Period years, period months, weeks, days.     |
+------------------------------+--------+---------+----------------------------------------------------------------+
| recall\_type\_value          | No     | Int     | Optional unless recall type is set.                            |
+------------------------------+--------+---------+----------------------------------------------------------------+
| unit\_type\_id               | No     | Int     | Person or Person-year                                          |
+------------------------------+--------+---------+----------------------------------------------------------------+
| unit\_value\_as\_published   | No     |         |                                                                |
+------------------------------+--------+---------+----------------------------------------------------------------+
| uncertainty\_type\_id        | No     | Int     | standard error, eff sample size, conf interval, sample size    |
+------------------------------+--------+---------+----------------------------------------------------------------+
| uncertainty\_type\_value     | No     | Int     | percentage of uncertainty as an int 0-100                      |
+------------------------------+--------+---------+----------------------------------------------------------------+
| mean                         | Yes    | Float   |                                                                |
+------------------------------+--------+---------+----------------------------------------------------------------+
| lower                        | Yes    | Float   | optional, so use when found                                    |
+------------------------------+--------+---------+----------------------------------------------------------------+
| upper                        | Yes    | Float   | optional, so use when found                                    |
+------------------------------+--------+---------+----------------------------------------------------------------+
| standard\_error              | Yes    | Float   |                                                                |
+------------------------------+--------+---------+----------------------------------------------------------------+
| effective\_sample\_size      | No     | Float   | optional, so don't rely on it                                  |
+------------------------------+--------+---------+----------------------------------------------------------------+
| sample\_size                 | No     | Float   | optional, so don't rely on it                                  |
+------------------------------+--------+---------+----------------------------------------------------------------+
| cases                        | No     | Float   | optional, so don't rely on it                                  |
+------------------------------+--------+---------+----------------------------------------------------------------+
| design\_effect               | No     | Float   | optional, so don't rely on it                                  |
+------------------------------+--------+---------+----------------------------------------------------------------+
| outlier\_type\_id            | Yes    | Int     | not outlier, outlier, group review, emr exclude, do not cite   |
+------------------------------+--------+---------+----------------------------------------------------------------+

1. A currently-missing column, ``age_demographer`` determines whether
   ``age_start`` and ``age_end`` extend to the following year if they
   are integral values.

2. Years are integers?! That's a problem because it makes it impossible
   to identify whether a duration is one month or one year. It's also
   impossible to create point data, specified at one time.

3. We should keep the urbanicity type because it's been mentioned as a
   possible way to split computation instead of by sex.

4. Several columns could be useful in the future, for instance, upper
   and lower. Sample size, cases, and design effect have been taken into
   account to determine error before this point, so they won't be used.

5. We haven't been using the outlier type to mark outliers, but that
   should be respected.

6. List of accepted measure\_ids: Below

Allowed Measure IDs
-------------------

This code below from the Cascade. Only those with an entry in the
right-hand column are converted into Dismod-AT input data. The Cascade
could convert several others for use, and those are listed after.

::

    INTEGRAND_ENCODED = """
    measure_id                             measure_name Dismod-AT name
     1                                           Deaths
     2           DALYs (Disability-Adjusted Life Years)
     3               YLDs (Years Lived with Disability)
     4                        YLLs (Years of Life Lost)
     5                                       Prevalence prevalence
     6                                        Incidence 
     7                                        Remission remission
     8                                         Duration
     9                            Excess mortality rate mtexcess
    10               Prevalence * excess mortality rate
    11                                    Relative risk relrisk
    12                     Standardized mortality ratio mtstandard
    13                    With-condition mortality rate mtwith
    14                         All-cause mortality rate mtall
    15                    Cause-specific mortality rate mtspecific
    16                       Other cause mortality rate mtother
    17                               Case fatality rate
    18                                       Proportion Sincidence
    19                                       Continuous Sincidence
    20                                    Survival Rate
    21                                Disability Weight
    22                               Chronic Prevalence
    23                                 Acute Prevalence
    24                                  Acute Incidence
    25                         Maternal mortality ratio
    26                                  Life expectancy
    27                             Probability of death
    28                   HALE (Healthy life expectancy)
    29                           Summary exposure value
    30                Life expectancy no-shock hiv free
    31                Life expectancy no-shock with hiv
    32           Probability of death no-shock hiv free
    33           Probability of death no-shock with hiv
    34                                   Mortality risk
    35                            Short term prevalence
    36                             Long term prevalence
    37           Life expectancy decomposition by cause
    38                                 Birth prevalence prevalence
    39                  Susceptible population fraction susceptible
    40               With Condition population fraction withC
    41                            Susceptible incidence Sincidence
    42                                  Total incidence Tincidence
    43  HAQ Index (Healthcare Access and Quality Index)
    44                                       Population
    45                                        Fertility
    """

Note that ``measure_id=6`` for incidence should be rejected as an input.

Measure IDs I could Imagine Using
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the Cascade gets better at guessing the right thing to do, some of
the other measures might help, but they are not used at all now.

1. 8 Duration: This could be converted into remission.

2. 10 Prevalence x EMR: This determines other-cause mortality, as total
   mortality - prevalence x emr.

3. 17 case fatality. This is used to determine appropriate
   approximations for remission.

4. 20, 21, 22: chronic and acute prevalence, acute incidence. If these
   were reliable, again they relate to incidence and remission
   calculations.

5. 35, 36: Short-term prevalence and long-term prevalence.

6. Population and fertility don't make sense for Dismod-AT because it is
   explicitly in a cohort-space, so you start with a population of 1 for
   all cohorts.

Allowed Values
--------------

For all of these, can values ever be NaN? What does that mean?

-  prevalence: 0 <= p <= 1.
-  remission: 0 <= r (no upper bound)
-  mtexcess: 0 <= e (no upper bound)
-  relrisk: 0 < r (no upper bound)
-  mtstandard: 0 <= s (no upper bound)
-  mtother: 0 <= o (no upper bound)
-  Sincidence: 0 <= i (no upper bound)
-  Birth prevalence: 0 <= p <= 1
-  susceptible: 0 <= s <= 1
-  with condition: 0 <= c <= 1
-  Sincidence: 0 <= s (no upper bound)
-  Tincidence: 0 <= t (no upper bound)

Other Table Input and Output
============================

This lists other inputs to Dismod-AT

-  Lots of tables in the dismod-at-dev/prod dbs, of course.
-  ``epi-db``

   -  bundle\_dismod
   -  bundle\_dismod\_study\_covariates

-  ``db_queries``

   -  age-specific death rate, ``get_demographics``
   -  age IDs and ranges, ``get_age_metadata``
   -  country covariate values, ``country_covariates``
   -  cause-specific mortality from ``get_outputs``
   -  ``get_life_table`` for ax.

-  hierarchies Python module for ``db_trees`` which is location
   hierarchy.
-  ``save_results``, calling ``save_results_at``.
