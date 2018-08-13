Installation
============

Installation of Cascade
-----------------------
Cascade interacts with Dismod-AT underneath. Cascade runs Dismod-AT within
the IHME infrastructure. Clone it from
`Cascade on Github <https://github.com/ihmeuw/cascade>`_. Then
use ``python setup.py install`` to install it into the current
Python environment or virtual environment.



Installation of Dismod-AT
-------------------------
This describes how to run Dismod-AT on your local computer, not on the
cluster. It uses Docker, which you have to install separately.

Download and run from the IHME Docker registry. This requires
VPN access. Maybe you can pull images without the login?::

    docker login reg.ihme.washington.edu
    docker pull reg.ihme.washington.edu/dismod/dismod_at:0.0.6

Then it's time to run the container. This describes how to share the
current working directory with the container when you run it. It will map
the current directory to the ``/app`` subdirectory in the container.
This example runs init on ``fit.db`` in the current directory::

    docker run reg.ihme.washington.edu/dismod/dismod_at:0.0.6 -it \
      --mount type=bind,source=${CURDIR},target=/app \
      /home/root/prefix/dismod_at/bin/dismod_at \
      /app/fit.db init

It's a long command, but it will run Dismod-AT.
