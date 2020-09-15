.. _dismod-run:

Run Dismod Commands
-------------------

To run dismod commands on a database (all possible options are
`here <https://bradbell.github.io/dismod_at/doc/command.htm>`_),
you can use the following helper functions. They will figure out
where your ``dmdismod`` executable is, whether it be installed on your
computer or pulling from docker, based on the installation of ``cascade_at_scripts``.

.. automodule:: cascade_at.dismod.api.run_dismod
   :members:
   :undoc-members:
   :show-inheritance:
