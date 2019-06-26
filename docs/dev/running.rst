.. _running-cascade:

Running Cascade
===============

Requirements
------------
This document describes the different ways to launch the Cascade.

Users
^^^^^

 1. Modelers invoke the Cascade through EpiViz-AT, which
    starts a UGE job.

 2. Modelers, and Greg and Brad, can start the Cascade on the
    command-line on the cluster. These restarts may run only one
    location, if that location had trouble solving.

 3. Testing code will run functions that perform resource-limited
    versions of the code. These should run within a single process.


Backend Support
^^^^^^^^^^^^^^^
This is a set of technical constraints. There are three backends:

 1. Under Grid Engine (UGE, SGE), because this is what we use for production.
 2. As multiple processes, for analysis work done by modelers on laptop
    or a single node on the cluster. This is requested by Greg.
 3. As one process, calling each part of the code as a function, because
    this is used to debug. You see it in Dismod-ODE code as an if-then
    that skips using multiprocessing. This is required for testing.

Choice 2, running as multiple processes, is a lower priority. It will
be a low cost when the other two run.

Given queue lengths on our cluster, it will be necessary to run
each Dismod-AT fit in its own UGE task. The fits can use over
16GB each, so this will also help ensure that a single estimation
recipe doesn't take twice as long as it could from trying to
fit all of its draws onto a single computer with a 256GB cap
on RAM.


How Modelers Specify Work
^^^^^^^^^^^^^^^^^^^^^^^^^
Modelers specify work in terms of statistical steps.
The code should make it possible to compare what modelers
requested with what the code does.

The whole cascade is constructed from these parts:

 * Modelers talk in terms of what we will call **recipes**,
   which can be, for instance, "fit fixed, then fit", or "aggregate".

 * Each recipe contains jobs within it, such as doing a MAP fit,
   doing each of the simulation fits, or summarizing those fits.

Structure of Computation
^^^^^^^^^^^^^^^^^^^^^^^^
The modelers have discussed wanting to run portions of a total
Cascade. These are the portions they have listed so far.

 1. Run everything, known as a **global run**.
 2. Run the **drill**, which is all stages for an estimation and its parent
    estimations.
 3. Run all stages for **part of a drill**. For instance, run one estimation,
    so that we can debug how to improve the model for one estimation.
 4. Run every estimation that depends on a given one. This is a
    **subtree** of the location hierarchy.


Structure of Execution
^^^^^^^^^^^^^^^^^^^^^^
Given the different shapes of computation for the problem on multiple locations,
we usually want to run the whole thing, but there are a
few important cases.

 1. Run the rest of a cascade after some error or restart.

 2. Run one location repeatedly in order to experiment with parameters
    to fit that location. This is requested by Greg and Brad.


Previous Work
-------------

Dismod-ODE does many of these things. Its capabilities are similar.

 * It runs a single estimation, including calculation of uncertainty,
   within a single UGE job.

 * It is set up specifically for UGE to run a Python application within
   an environment. This means the Python application can be run separately
   by hand, if desired, for a single location.

 * Debugging for a single location can use a command-line argument that
   turns off multiprocessing use, making it easier to stop the code
   in any given function.

 * Restarting is handled by Jobmon, which records which processes
   ran successfully and can both restart failed processes and resume
   a workflow to complete the rest in the model.

What's different?

 * Dismod-ODE doesn't split individual draws into separate UGE jobs.
   In fact, it aggregates multiple estimations, for multiple locations,
   into the same job, using Python's multiprocessing library.

 * Dismod-ODE's wrapper isn't set up to run off the cluster.

 * Dismod-ODE doesn't run a drill, a single location, or a
   sub-hierarchy of locations. The structure of computation is baked
   into construction of the Jobmon job objects.

Problems for Dismod-ODE's implementation:

 * It's hard to figure out what happens at what level of the cascade.
   The rules are applied throughout the code. That makes it difficult
   to figure out if the code meets requirements.

 * It's difficult to figure out what input files exist when
   a UGE job runs. It's harder to figure out what's in those files.
   The columns and data types can be different at different levels
   of the cascade.

 * There isn't a way to test that the cascade was constructed
   correctly. Any time you touch the code that makes the Jobmon workflow,
   you have to run an example job, and the only reassurance you have that
   all the pieces connected is that there were no errors when it finished.
   That tests one job, and that one job tends not to throw errors if files
   are missing. We often make the tweak and then don't run the test job.


Proposed Implementation
-----------------------

Computational Structure
^^^^^^^^^^^^^^^^^^^^^^^

I see the computational structure as layers of graphs.

 1. The location hierarchy, which is a tree structure.

 2. The subset of that location hierarchy that will be used for this model.

 3. The graph of recipes constructed from that subsetted location hierarchy.
    That may include a pre-global step to download data from IHME databases
    and save it in a file. It will include an aggregation step at the end,
    so that this graph isn't a tree structure.

 4. The graph of separate tasks that make up the recipes. This is necessary
    because the resource requirements of sub-tasks of a recipe are too
    high for a single UGE job and because there is parallelism in the
    sub-tasks that allows them to be solved faster.

Execution
^^^^^^^^^

Execution of that structure can be separate from the structure itself.
Once we have the graph of tasks, we can hand that to any of the
backends for UGE, for running as functions, or for running as
subprocesses. Each of those backends can handle partial execution
and error-handling in its own way.

The tasks in the task graph could be processes, functions, UGE
jobs, or UGE tasks. (A UGE job has one or more tasks, where a
UGE task array has multiple tasks.) What is the right way to represent
one of those tasks?

 * A task is not a function, because it also has resource requirements
   for memory and run time.

 * A task is not a Unix process because it doesn't always have
   file descriptors, process identifiers, and the other trappings
   of a Unix process.

I'd like to think of a task as a component, the way Szyperski
defines them in *Component Software.* They have

 1. A signature.
 2. Order guarantees for usage of time and memory.
 3. Well-defined exception-handling across the boundary.
 4. Eventual consistency for completion. (I'm fuzzy on whether
    this was one of his rules. It's been a while.)

Out of respect for those guarantees, I propose a design
that implements each task as a class that has a list of
input and output files and tables, and that has a function
that returns expected resource usage as a function of
input data parameters.

Recording inputs and outputs also helps with restarts.
Sometimes the scheduler doesn't tell the truth because of
network partitioning, but the files never lie, so we can use
file presence or absence to determine restarts.


Testing Strategy
^^^^^^^^^^^^^^^^

There are two main collaborators that can be faked for testing.

 1. Model specification, as locations and EpiViz-AT settings.

 2. The execution environment. This could be faked as an entirely
    fake execution environment that does nothing but validate.
    It could also be that we fake qsub or fake at the level
    of tasks, by making each task read and write stub files.

