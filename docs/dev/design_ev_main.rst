.. _design-ev-main:

Design EpiViz-AT Main
=====================

The EpiViz main looks like structured programming.

 1. Parse Arguments and read Settings.
 2. Construct a CascadePlan.
 3. Choose a cascade job to do, which will be an estimation job.
     1. Download data with little modification.
     2. Compute results
         1. Clean and construct inputs
         2. Build model
         3. Run fit and simulate.
         4. Gather results
     3. Upload results
 4. Report metrics
