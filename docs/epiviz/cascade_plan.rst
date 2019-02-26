.. _cascade-plan:

Cascade Plan
============

Given all locations in the world, EpiViz-AT makes a plan
for which locations are included in a single run of the
Cascade and how to use the posterior likelihood from
a one estimation to construct priors for a child estimation.

.. _single-estimation-and-drill:

Single Estimation and Drill
---------------------------

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

A **drill** limits estimations to parents of a single
lowest-level estimation. If the lowest-level estimation
is the United States, then the lowest-level estimation will
predict rates for the states, and the drill could include
the Americas, high-income countries, and the globe, or it
could go only to high-income, or only solve for the
United States.


.. _what-affects-plan:

What Affects the Plan
---------------------

 *  The set of all locations requires a location
    set version ID and GBD round id.
 *  Drill or Global option chooses whether to run the whole
    world, as defined by the location set, or to run a drill.
 *  Drill start and drill end determine the top level and lowest
    level of a drill.
 *  Drill sex can be male, female, or unset. If it is unset,
    the drill will include both male and femal.
 *  The "exclude data" option for priors determines which
    posterior draws of integrands to add to the child estimations.
 *  The split sex option determines at what level of the location hierarchy
    to solve male and female as separate estimations.

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
