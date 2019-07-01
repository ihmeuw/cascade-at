
## Make the Docker

Let's call the Docker registry dockerhub.com.
(The internal one has a different name.)
Let's also assume singularity images are
in $SINGULARITY on the cluster.

Edit the Makefile to increment the version ID of
dockerhub.com/dis/dismod_at:18.12.04

Build the docker image on a local machine with::

    make clean; make

Then push the image to the local dockerhub with
the image number corrected::

    docker login dockerhub.com
    docker push dockerhub.com/dismod/dismod_at:18.12.04
    docker tag dockerhub.com/dismod/dismod_at:18.12.04 \
        dockerhub.com/dismod/dismod_at:latest
    docker push dockerhub.com/dismod/dismod_at:latest

Then clone the repository to the cluster and run
an SGE job to create the singularity image. You have to edit
the version again in the ``create_singualarity.sh``::

    cd cascade_at/automation/dismod
    qsub create_singularity.sh

This will create a file called
`$SINGULARITY/dismod/dismod_at:18.12.04.img`
which you can then run with::

    singularity exec \
       $SINGULARITY/dismod/dismod_at:18.12.04.img \
       /home/root/prefix/dismod_at/bin/dismod_at file.db init

Once it works, there is another step, tagging it as latest.

    docker images ls  # to find the hash
    docker tag <hash> latest
    docker push dockerhub.com/dismod/dismod_at:latest

Login to the cluster and make it current.

    cd $SINGULARITY/dismod
    rm -f current.img && ln -s dismod_at:18.12.04.img current.img


Final note, that I removed mount points from the Docker
for putting this on Github. Haven't tested it without those mounts.
