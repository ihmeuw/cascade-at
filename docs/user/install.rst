.. _install-api:

Installation
============

Installation of Cascade
-----------------------
Cascade interacts with Dismod-AT underneath. Cascade runs Dismod-AT within
the IHME infrastructure. Clone it from
`Cascade on Github <https://github.com/ihmeuw/cascade>`_.

We recommend you create a virtual environment into which to install
the code. You need python3 to run virtualenv.  You can use our miniconda3 
installed on the cluster::

    /ihme/code/dismod_at/pyenv/miniconda/bin/virtualenv ./env_path
    source ./env_path/bin/activate
    
You can name the environment something happier than env_path.
Then::

    git clone https://github.com/ihmeuw/cascade.git
    # Or use the one below if you have a Github account.
    # git clone git@github.com:ihmeuw/cascade.git
    cd cascade
    pip install .[documentation,ihme_databases,testing]
    python setup.py develop
    cd tests && pytest

Then you should find that the current Python virtual environment
has several new scripts installed.


Installation of Dismod-AT
-------------------------
Dismod-AT is already installed on the cluster. If you've installed the
Cascade as described above, it will have created two commands,
``dmdismod`` and ``dmdismodpy``, which are the application and its
Python helper. You can run, for instance::

    dmdismod data.db set option print_level_fixed 5
    dmdismod data.db set option quasi_fixed false
    dmdismod data.db set option ode_step_size 1
    dmdismod data.db init
    dmdismod data.db fit fixed
    dmdismod data.db predict fit_var
    dmdismodpy data.db db2csv

The value of print_level_fixed is a positive integer between 0 and 12 inclusive,
0 is the default and corresponds to no printing of information about optimizing 
the fixed effects.  You can experiment with this value, to get more information, 
say, if your model is not converging quickly. 

Without the helper, the same commands would be::

    SINGULARITY=/ihme/singularity-images/dismod/current.img
    DMPATH=/home/root/prefix/dismod_at/bin/dismod_at

    singularity exec "${SINGULARITY}" "${DMPATH}" data.db set option print_level_fixed 5
    singularity exec "${SINGULARITY}" "${DMPATH}" data.db set option quasi_fixed false
    singularity exec "${SINGULARITY}" "${DMPATH}" data.db set option ode_step_size 1
    singularity exec "${SINGULARITY}" "${DMPATH}" data.db init
    singularity exec "${SINGULARITY}" "${DMPATH}" data.db fit fixed
    singularity exec "${SINGULARITY}" "${DMPATH}" data.db predict fit_var
    singularity exec "${SINGULARITY}" /home/root/prefix/dismod_at/bin/dismodat.py data.db db2csv

If you are on your local machine, and have installed Docker,
then you can try the same helper scripts, ``dmdismod`` and ``dmdismodpy``.
They will automatically
get the latest Dismod-AT image from the IHME Docker repository
and run that. There is something funny with directories though,
because of Docker mounts, so take a look below to understand the
``dmdismod`` commands.

This describes how to run Dismod-AT on your local computer, not on the
cluster. It uses Docker, which you have to install separately
from `Docker Download <https://www.docker.com/get-started>`_.

Download and run from the IHME Docker registry. This requires
VPN access.::

    docker pull reg.ihme.washington.edu/dismod/dismod_at

Then it's time to run the container. This describes how to share the
current working directory with the container when you run it. It will map
the current directory to the ``/app`` subdirectory in the container.
This example runs init on ``fit.db`` in the current directory::

    docker run -it --mount type=bind,source=${CURDIR},target=/app \
      reg.ihme.washington.edu/dismod/dismod_at \
      /home/root/prefix/dismod_at/bin/dismod_at \
      /app/fit.db init

It's a long command, but it will run Dismod-AT.

If you use the dmdismod commands and have ``data.db`` in the
current directory, you would run::

    dmdismod data.db init
    dmdismod data.db fit
    dmdismod data.db predict
    dmdismodpy data.db db2csv

At least the dmdismod command provides an example to tailor.
