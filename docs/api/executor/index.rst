.. _executor:


Defining and Sequencing the Work
================================

Cascade-AT performs work by calling scripts
that are included as entry points to the package.

Scripts as Building Blocks
--------------------------
The work in a Cascade-AT model is sequenced like building blocks.
Here is how the work starts out, moving from micro to macro.

The following submodules contain classes and functions
for constructing a job graph that runs Dismod-AT. The smallest
is a cascade operation, which defines one executable task. These
can be stacked together into sequences (stacks), and then recursively
put into a tree structure (dags). The cascade commands are wrappers
around the dags.

.. toctree::
   :maxdepth: 1

   cascade/operations
   cascade/stacks
   cascade/dags
   cascade/commands

Each of these is described briefly below.

* **Scripts**: All potential work starts out as a script.

* :ref:`cascade-operations`: We start with building wrappers around the scripts, and we call these
  wrappers cascade operations. These wrappers are helpful because they
  define the command-line string that will be executed in order to perform the work
  by calling the script with particular arguments, the name of the job if it is
  submitted through a qsub, etc. They also directly interface with ``jobmon``,
  an IHME package that submits and tracks parallel jobs.

* :ref:`cascade-stacks`: There are some sequences of work that often go together, for example
  like running a fit fixed, then a sample, then a predict. These types of sequences
  are called stacks, because they are "stacks" of cascade operations.

* :ref:`cascade-dags`: Once we take many sequences and form them into a tree-like structure
  that traverses a location hierarchy, that's called a DAG or a job graph. The structure
  of this DAG is based off of an IHME location hierarchy, and it defines
  the work for the entire cascade. The DAGs module provides functions to, for example,
  recursively create stacks going down a tree.

* :ref:`cascade-commands`: This is the most "macro" type of work. You say, "I want
  to do a cascade" or "I want to do a drill" by creating a cascade command, and then
  it works its way through DAGs --> Stacks --> Operations --> Scripts to define
  all of the work, with arguments based off of the model version ID's settings that
  you pass to the cascade command.


Arguments
---------
Each of the scripts takes some arguments that are pre-defined using the tools
documented in :ref:`args`.

.. toctree::
   :maxdepth: 1

   args


Jobmon
------
The submitting and tracking of the distributed jobs to do a cascade is done
by the IHME package ``jobmon``. :ref:`cascade-operations` are roughly jobmon
tasks and :ref:`cascade-commands` are roughly jobmon workflows.

We have to convert
between cascade operations and tasks and cascade commands and workflows. Helper functions
to do these conversions are documented in :ref:`jobmon`.

Jobmon uses information from cascade operations and cascade commands to interface
directly with the IHME cluster and the Jobmon databases. See :ref:`jobmon`.

.. toctree::
   :maxdepth: 1

   jobmon

Entry Points
------------

Each of these scripts takes arguments, defined at the top of the scripts.
Here we list the different types of work that are done, and in each section are
three things:

1. The main function in the script, with documentation
2. The cascade operation associated with that script

They are listed in the order that they typically occur to run a Cascade-AT model
from start to finish, with the exception of :ref:`run`, which is how all of this work
is kicked off in the first place.

.. toctree::
   :maxdepth: 1

   run
   scripts/configure-inputs
   scripts/dismod-db
   scripts/sample
   scripts/mulcov-statistics
   scripts/predict
   scripts/upload
