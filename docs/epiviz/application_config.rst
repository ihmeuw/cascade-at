.. _application-config:

Application Configuration
=========================

There are a bunch of different kinds of settings and parameters involved.
This section describes those settings that configure directories and
other installation-specific parameters.

The code to get these installation-specific parameters is::

    from pathlib import Path
    from cascade.executor.execution_context import application_config
    config = application_config()
    root_directory = Path(config["DataLayout"]["root-directory"])

These configuration parameters are stored in a TOML file,
which is remarkably like an INI file, but with more regular syntax.

The goal of the ``application_config`` module is to separate
configuration from installation (as recommended in *Release It!*.
There is a TOML configuration file installed
in the main ``cascade`` source tree. There is another one
in an IHME-specific repository called ``cascade_config``.
This means local directories and other configuration are behind
the firewall.

Set configuration parameters by updating the ``cascade_config``
repository and installing it into the same Python environment
as the ``cascade``. The configuration will be found by
using Python's native ``setuptools``.
