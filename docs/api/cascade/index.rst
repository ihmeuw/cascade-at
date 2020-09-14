.. _cascade:


Cascade Structure
=================

The following submodules contain classes and functions
for constructing a job graph that runs Dismod-AT. The smallest
is a cascade operation, which defines one executable task. These
can be stacked together into sequences (stacks), and then recursively
put into a tree structure (dags). The cascade commands are wrappers
around the dags.

.. toctree::
   :maxdepth: 1

   operations
   stacks
   dags
   commands
