.. _epiviz-logs:

Logs
====

There are two sources for logs, the Python wrapper
code, or Dismod-AT itself. Those logs are filtered and stored
in two ways, the math log and the code log.

The logs record what the Cascade actually chose to do when it saw
inputs and settings, although they are difficult to parse. The reason
to parse them is to see how interactions between what the user requested
through the EpiViz-AT user interface and what the databases provide resulted
in a final model. There is one log per estimation run.

These get embedded in the Cascade logs, but it is worth mentioning that
the log-level used for Dismod-AT can make the logs get very large,
because Dismod-AT can log every step of its nonlinear solver. Dismod-AT
logs have two levels, and we really need the top-level log in order
to determine the quality of a fit.

