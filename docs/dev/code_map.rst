.. _map-of-the-code:

Map of the Code
===============

Fire Walls in the Code
----------------------

If you're going to modify something, here are the some guides.

 * `/input_data/configuration/raw_input.py` validates the bag of input
   data from the epi-at database and other IHME databases, so it tells you
   the columns and datatypes supplied as input to model building and
   model running.

 * Settings straight from EpiViz-AT are validated in
   :py:class:`cascade.input_data.configuration.form`,
   but they get transformed into a second set of settings, specific
   to this location estimation, by the ``CascadePlan``.

 * Everything that gets written to the Dismod DB file or read from
   it passes through the :py:class:`cascade.model.ObjectWrapper`,
   so that defines types on the other side of where you likely
   need to work.


Quick Context
-------------

There is a separate requirements document. This is a brief
sense of scope.


Usage We Support
^^^^^^^^^^^^^^^^

1. Modeler submits job through EpiViz-AT. (Main success scenario)
   a. It's a connected to a different back-end database. (Faux Dismod-MR)
2. Operations runs Cascade from command-line on cluster.
   a. To run a full model. (Used now to run Dismod-MR during decomps)
   b. To run part of a model after an outage.
3. A modeler runs an IPython notebook to explore Dismod-AT,
   not the full Cascade. (Sasha's group does this now.)
4. A modeler makes other applications using these tools
   to support exploration. (Greg's work on simulation, for instance.)
5. Developer runs a job on the command-line on the cluster against
   production or development databases. (Used by Joe now for testing.)
6. Developer runs on laptop. (For unit tests.)


Separate Installable Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How many times and places do we install something? This is the
definition of a "container" by `C4 Model <http://c4model.com/>`_.


* Cluster installation of dismod_at on cluster using Docker & Singularity

* Cluster installation of Cascade on fair cluster.

   * production runs - production test of install, production live.

   * development runs

* Cluster installation on prod cluster.

* Development installation on local machines.

* Web service to give fast feedback to EpiViz-AT about settings consistency:
  `Cascade Validation Service <https://github.com/ihmeuw/cascade_validation_service>`_.

* IPython notebook use on local machines, for model exploration.

* Github

   * Source repository

   * Unit testing of every code branch by Travis in a Docker image.

   * rocketsonde repository for metrics.

* Documentation at cascade.rtfd.io, built in Docker image.


Components
----------

Each subdirectory in the code, under `src/cascade`, is
treated as a separate component, again thinking of `C4 Model <http://c4model.com/>`_
components. There is a unit test, called
``test_component_dependencies.py``, that ensures there are no dependency
cycles among the components.

The overall approach is to treat this code base as
a framework with which to write multiple, changing ``main()``
functions. To do that we emphasize two pieces: validation on the way
in for settings and data, and creating a well-tested, high-level
interface to the Dismod-AT db file. The Dismod-AT db interface
contained in the ``cascade.model`` component
is, itself, 3000 lines of code that translate SQL tables into
Python objects, and back again. As a result, the current ``main()``
is under 1000 lines, written as a set of functions that manipulate
Model, Var, and Covariate objects.

As of this writing, here's a sense of what's in the components
and where the 7800 lines of code in the source are allocated.


 1. Scripts

    * Build singularity image of Dismod-AT (259 bash and docker)

    * Deploy Dismod-AT to the cluster and acceptance test it. (192 bash)

    * Run from EpiViz-AT grid engine call (147 bash)

    * Run Dismod-AT on local computer. (30 bash)

 2. Documentation - always built, always online, for operations and developers.

    * Text (4000 lines)

    * Build more easily and silencing useless errors: 40 lines.

 3. Core tools

 	* Parsing the wilderness of parameters from EpiViz (294 lines of code)

 	* Deal with Central Comp uisng files for db keys: 10 lines

 	* Redirect connections so that we can have different dev / prod environments (90 lines)

 	* Collect and scan Dismod-AT output while it's running (85 LOC)

 4. Dismod-AT low-level handling

    * Define database schema and cache queries to it (597 lines), so that we
      can protect code from schema changes and so that there is a way to validate data we make.

    * Constants that Dismod-AT uses (97 LOC)

    * Read Dismod-AT stdout to know when it really failed (85 LOC)

    * Parse a Dismod-AT db file to measure how big the problem is for prediction of runtime (141 LOC)

 5. Translate Dismod-AT tables into statistics objects that use IHME location_id, etc. (2214 LOC)

    This is an interface, so that everything below it is really, really tested,
    and when you ask someone to change a statistics behavior, they don't need to think
    about the exact Dismod tables. Makes it much harder to write the wrong thing to the db_file.

    * Define an object to represent a single Fit on all rates.

    * Define an object to represent the priors on all rates.

    * Translate on writing and reading.

 6. IHME Database access for ASDR, CSMR, demographics, locations, bundle, covariates.

    * Initial query, with care for what can go wrong (661 lines)

    * Separate step to
      enforce transformations, interpolate (900 lines).

    * The result is data that has a known set of data types, columns, missingness.

    * Saving results 95 LOC

 7. The Application (Executor)

    * Parse 109 different kinds of parameters from EpiViz (300 LOC).

    * Define global hierarchy separate from other work (179 LOC).

    * Covariates (152 LOC)

    * The bulk of the application: 620 LOC.

      * What data to include and exclude

      * Make the right overall model, with its rates, random effects, and covariates.

      * Run that model, and simulate, and run fits on simulations.

      * Pass data down to the next piece.

 8. Runner

    * Interfaces with the
      `Grid Engine App <https://github.com/ihmeuw/gridengineapp>`_
      repository in order to make the Cascade runnable on the cluster
      and on local machines.

    * Logging - Split into log for modelers and operations, highlighted for web page,
      with defined uses for debug, info, warn, and error. (153 lines)

    * Parse arguments

 9. Storage support

    * File format for concatenating results from locations into HDF that can
      be read by Python and R. 195 LOC

 10. Separate command-line tools for modelers and operations.

    * Get parameters from IHME db to view them (40 LOC)

    * Get model residuals from IHME db (41 LOC)

    * Get model results from IHME db(75 LOC)

    * The main application body, described above (620 LOC)

    * IPython helpers to make data (in stats section): (191 LOC)

 11. Testing - 5500 LOC total

    * Creating fake EpiViz calls: 276 LOC

    * Most of Brad's examples. 392 LOC

    * Make fake data: 105 LOC

    * Compare two Dismod db files : 59 LOC

    * Ensure there are no cycles in software dependency graph. 51 LOC

    * Make a large number of different fits for testing and timing: 383 LOC
