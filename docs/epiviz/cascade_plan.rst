.. _cascade-plan:

Cascade Plan
============

The Cascade Plan makes two important decisions.
What is the shape of the hierarchical model,
and what settings are used for each sub-model in the hierarchy?

.. _terminology-cascade-plan:

Terminology of the Cascade Plan
-------------------------------

An Estimation
    A single estimation is a two-level hierarchical model
    where the micro level is measurement data and the macro
    level is the set of locations. These locations are always
    siblings which share a parent location in the IHME location
    hierarchy, so we often refer to a single estimation as being
    "for parent location United States with states as child
    locations." Each measurement is taken within whatever
    most-detailed location is registered, so we aggregate all
    data up to the level of the child locations, in order to
    assign it. (More properly, Dismod-AT does this internally.)
    For instance, county data would be counted as belonging to
    states, in the example above, and any data recorded as
    being in the United States (location id 102) would be ignored
    completely for the single estimation with United States
    as parent location.

Location Hierarchy
    IHME assigns a location ID to every country, to districts
    within countries, to groups of countries in the same
    region, to sets of regions known as super-regions, and
    to the world as a whole (the world's location ID is 1).
    The location hierarchy is a directed acyclic graph where
    the world is the root node. IHME uses multiple hierarchies
    for different purposes, and, for some of them, the same
    location can have more than one parent location. This
    hierarchy is simplified for Dismod-AT so that there is
    a unique parent for each location.

Estimation Hierarchy
    This is a hierarchy of two-level mixed-effects models.
    It may mirror the location hierarchy exactly or have
    a different shape, as determined by the Cascade.

Recipe
    A recipe is a set of modeling steps to do for each
    estimation within the estimation hierarchy. For
    instance, a "fit fixed then fit both" recipe
    will fit the location with no random effects and use
    that result as the initial guess for a fit with
    mixed effects.

    Each recipe in the estimation hierarchy has its
    unique recipe identifier, consisting of
    the location ID, the sex (male, female, both),
    and the recipe name, "bundle_setup" or "estimate_location".

.. _hierarchy-shape-plan:

Shape of the Hierarchy
----------------------

The Cascade Plan, in the ``cascade.executor.cascade_plan``
module, decides the shape of the hierarchy.

Given: A location hierarchy, command-line arguments, and
the user's input settings from EpiViz-AT, the Cascade
Plan decides the graph that defines the estimation hierarchy.

 1. Decide the level at which to split sex. Read the value from
    the EpiViz-AT settings. If it is set to "most detailed", then
    use the lowest level of the locations graph.

 2. Add a "bundle_setup" node as the first job to do. This will
    download data to a directory.

 3. For every node whose level is less than the split sex value,
    add one "estimate_location" node with sex="both".

 4. For every node whose level is at or below the split sex,
    add one "estimate_location" node with for each sex. If the
    parent estimation, the one at the parent location, is of the
    same sex, then depend on that node. If it is of both sexes,
    then state that it depends on the both sex node.

 5. Transform the graph of recipes into a graph of Grid Engine
    jobs to do. This is in ``cascade.executor.job_definitions.recipe_to_jobs``.
    Create a ``GlobalPrepareData`` node for the "bundle_setup" recipe.
    Create a sequence of Grid Engine jobs for estimation: fit fixed,
    fit both, draws, summarize.


.. _number-of-estimations:

Number of Estimations
---------------------

Each location-specific estimation of Dismod-AT generates more data, so let's
count the estimations for a global run.

At one parent location, the Cascade estimates rates for all child locations.

 *  **Fit** Runs a Dismod-AT fit to find the most likely estimate.
 *  **Simulation** Runs :math:`s=30` more fits to determine error in that most likely estimate.
 *  **Predict** Runs predict on each of those fits in order to get error for integrands.

Given a location hierarchy with six levels, the most-detailed, or leaf, level
doesn't need a separate run. If the levels are global, super-region, region,
national, and sub-national, then Dismod-AT runs on global, super-region, region,
and national, which is four levels. At each of those levels, there will be
a single fit to find the maximum and then 30 simulations, all 30 of which
can run simultaneously.


.. _parameters-plan:

Parameters for Each Estimation
------------------------------

The second job of the Cascade Plan is to establish parameters
for every estimation in the hierarchy. The idea is to
answer one question:

    Every choice of the form, "if the level is X, then
    do Y, else do Z," should be decided when transforming
    EpiViz settings into estimation-local settings.

This applies to

 * Setting the ``bound_random`` option.

 * Defining the sex to use for this location and what
   sexes the parents or children use.

 * Supplying the ``grandparent_location_id`` and
   the location IDs of ``children``.

 * The ``sample_cnt``, which is the number of draws to take
   in order to estimate uncertainty. This may be larger
   for the most-detailed estimations.
