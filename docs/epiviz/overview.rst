.. _overview-of-epiviz:

Overview of Cascade
===================

EpiViz-AT is a web page on http://internal.ihme.washington.edu/epi-at/.
After you fill out
that web page and hit submit, the web server runs a program on the cluster.
We call that program ``dmcascade``. It is installed on the IHME cluster,
and the code itself is stored at https://github.com/ihmeuw/cascade.
This documentation, that you're reading, is at https://cascade.readthedocs.io.

The Cascade is composed of several systems, some of which are shown in
this partial diagram:

.. image:: cascade_system_diagram.png
    :scale: 25

The `User Manual <user-manual>`_ covers the Cascade API and its
:ref:`dismod-description` is the best
spot to learn about the underlying statistical model.
All detailed questions about Dismod-AT are in the
`online Dismod-AT documentation <https://bradbell.github.io/dismod_at/doc/dismod_at.htm>`_.

This Cascade documentation looks at the problem in parts.

 *  Cascade Plan - decides what to do at each level of the location hierarchy.
 *  EpiViz-AT Inputs - describes input data and transformations to that data.
 *  EpiViz-AT Outputs - describes output data and transformations to that data.
 *  Model Building - How the Cascade uses EpiViz-AT settings and the Cascade API.
 *  Command-line tools - Executable applications on the cluster, including :ref:`dmcascade-command-line`
 *  Operations and Maintenance - Processes to manage the Cascade.
