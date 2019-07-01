.. _docker-container:

Docker Container for Dismod-AT
==============================

Dismod-AT is installed on the cluster using a Singularity image
built from a Docker container.

 * The Dockerfile is in the
   `Dismod Containers Repository <https://stash.ihme.washington.edu/projects/CON/repos/dismod/browse/dismod_at>`_.

 * The singularity container is built with two shell scripts that are under
   ``scripts/automation/dismod``.

You have to

 1. Check out the repository to a machine with Docker, usually a laptop.

 2. Edit the Makefile and build, as described in the README.md.

 3. Push that to the IHME registry, as described in the README.md.

 4. On the cluster, install a new version of the wrapper and run its
    tests.

 5. If those tests pass, then change the symlink in the containers
    directory (described in file locations) to point to the current image.
