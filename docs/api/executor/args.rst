.. _args:

Argument Parsing
^^^^^^^^^^^^^^^^

Each of the scripts from :ref:`executor` uses argument utilities that are
described here. Arguments are single command like args, that would be passed
in something like ``--do-this-thing`` or ``--location-id 101`` as flags.
We use the ``argparse`` package to interpret these arguments and to define
which arguments are allowed for which scripts.

Arguments are building blocks for argument lists. Each script has an
argument list that defines the arguments that can be passed to it that's
included at the top of the script.

Arguments
"""""""""
There are general arguments and specific arguments that we define
here so we don't have to use them over and over.

.. automodule:: cascade_at.executor.args.args
   :members:
   :undoc-members:
   :show-inheritance:


Argument List
"""""""""""""
Argument lists are made up of arguments, and are defined at the top of each
of the :ref:`executor` scripts. The reason that they're helpful is because
we can then use those lists to parse command line arguments *and* at the same
time use them to validate arguments in :ref:`cascade-operations`. This makes
building new cascade operations much less error-prone. It also has a method
to convert an argument list into a task template command for :ref:`jobmon`.

.. autoclass:: cascade_at.executor.args.arg_utils.ArgumentList


Argument Encoding
"""""""""""""""""
When we are defining arguments to an operation, we don't want to write
as if we were writing something on the command line, especially with things
like dictionaries and lists of dismod database commands.

The following functions are helpful for encoding and decoding dismod option
dictionaries to be used with the dismod database and dismod commands
to run on a dismod database.

.. autofunction:: cascade_at.executor.args.arg_utils.encode_options

.. autofunction:: cascade_at.executor.args.arg_utils.parse_options

.. autofunction:: cascade_at.executor.args.arg_utils.encode_commands

.. autofunction:: cascade_at.executor.args.arg_utils.parse_commands
