.. _install-api:

Installation
============

Installation of Cascade-AT
--------------------------
Cascade-AT interacts with Dismod-AT underneath. Cascade-AT runs Dismod-AT within
the IHME infrastructure. Clone it from
`Cascade-AT on Github <https://github.com/ihmeuw/cascade-at>`_.

We recommend you create a conda environment into which to install
the code. Then clone the repository and run the tests.::

    git clone https://github.com/ihmeuw/cascade.git
    # Or use the one below if you have a Github account.
    # git clone git@github.com:ihmeuw/cascade.git
    cd cascade
    pip install .[documentation,ihme_databases,testing]
    python setup.py develop
    cd tests && pytest

For instructions on how to install all of the IHME dependencies,
see the internal documentation `here <https://scicomp-docs.ihme.washington.edu/dismod_at/current/install/>`_.

For instructions on how to install ``dismod_at``, see Brad Bell's documentation
`here <https://bradbell.github.io/dismod_at/doc/dock_dismod_at.sh.htm>`_.