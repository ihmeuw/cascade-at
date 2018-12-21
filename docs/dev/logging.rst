.. _error-handling-plans:

Error-Handling Plans
====================

.. _exception-handling-plans:

Exception-Handling Proposal
---------------------------

If we look at the layers of the code, we can handle errors in different
ways between and within the layers.

*  EpiViz is one version of the top of this chain.
*  At the top, catch all exceptions and return as strings to EpiViz on initial call.
*  Within processing the settings, any settings that don't
   make sense are returned as a list of errors, not through exception-handling.
*  Below this, assume we are working inside of a UGE job.
   Failure of one job does not kill all jobs b/c people can
   use whatever data they get, often times.
*  The UGE job catches all exceptions and sends them to logs.
   This includes both random exceptions and exceptions that are
   about the more complex construction of the model. An example
   of such an exception is that you created a covariate but
   never set its reference value.
*  Within the code to setup the model, throw exceptions from
   our custom hierarchy when there is something a modeler could do differently.
*  Dismod-AT errors... maybe these are returned as exceptions?

That would be a hierarchy that looks like:

*  CascadeModelError (This is the one that catches more complicated
   model setup faults) and is for the modelers.

   *  Data selection problems.
   *  Algorithm selection problems.
   *  Settings selection problems.

and it is only used within the model construction and serialization,
not during checking of settings.


.. _logging-structure:

Logging
-------
The modelers should be able to see statistical choices, and those can
be separate from debugging statements. Those logs would have separate
lifetimes on disk, too.

*  *Code log* This records regular debugging statements, such as
   function entry and exit. It is kept on the disk.

*  *Math log* This has information about choices the code makes with
   the data. It is shown to the users in EpiViz. All of the Math log
   is always kept.

+------------+------------------------------------------------------------------------+
|**Code Log**                                                                         |
+------------+------------------------------------------------------------------------+
|Debug       | Up to coder. Will be turned off during production runs.                |
+------------+------------------------------------------------------------------------+
|Info        | Kept on in production runs.                                            |
+------------+------------------------------------------------------------------------+
|Warn        | Kept on in production runs. Any warning that fires requires action to  |
|            | to disable it.                                                         |
+------------+------------------------------------------------------------------------+
|Error       | Kept on in production runs, and we read all of these.                  |
+------------+------------------------------------------------------------------------+
|**Math Log**                                                                         |
+------------+------------------------------------------------------------------------+
|Debug       | About choices that are built-in to model logic.                        |
+------------+------------------------------------------------------------------------+
|Info        | About choices where a switch decides what to do.                       |
+------------+------------------------------------------------------------------------+
|Warn        | A problem that needs to be fixed, possibly with another run, but it    |
|            | doesn't make this run completely fail.                                 |
+------------+------------------------------------------------------------------------+
|Error       | Has to be addressed in order to complete this Cascade run.             |
+------------+------------------------------------------------------------------------+

Mathlog statements should have the following.

1.  Put MATHLOG statements in places where you have context on the data
    the function affects. This often means the log statement is in the
    caller.

2.  Include in the log statement summary stats like the number of rows,
    names of variables, things that inform about *this run*.

3.  If something was a choice, indicate how a modeler made that choice,
    and hence how she could unmake it, so refer to the EpiViz selection.


Alec proposes we could construct a hierarchical and narrative MATHLOG
which reads, for the modeler, like::

    Preparing model:
       Downloading input data:
           ...
       Constructing model representation:
           Adding mortality data from GBD:
               Assigning standard error based on bounds
               ...
       Running dismodat

We could write this as a streamable HTML document.

.. _fault-failure:

Faults and Failures
-------------------

Classify failures by the faults that caused them. Highlight to
the modeler the ones they can fix.

*  Model configuration

   *  settings don't meet needs and can be changed.
   *  bundle values don't make sense in some way.

*  IHME Data, maybe modelers know these.

   *  Database doesn't have data we think it should.
   *  IHME Database not responding or otherwise having a problem.

*  All the other faults, not possible modelers will fix these.

   *  Logic errors
   *  Environment errors regarding directories, writability.

*All* errors go to the math log, which also goes to the code log.


.. _logging-usability:

Logging Usability
-----------------
Messages to the GUI user should include

*  The line in the code, with a link to that line in Github.
*  A link to the exception description in the help on docs.
*  A link to the function in which exception occurred.

These would require, for the link to Github, knowing
the git commit so that it links to the right line.
For the URL, it would mean having the refs from the ``objects.inv``
file that sphinx makes when it makes the docs. It has
the mapping from Python entity to its URL and tag in the
documentation.
