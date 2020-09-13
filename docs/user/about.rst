.. _about-cascade-api:

About the Cascade API
=====================

Dismod-AT is an executable application, written in C++ and
compiled. It has a Python API already, as demonstrated
by the `Dismod-AT User Examples <https://bradbell.github.io/dismod_at/doc/user.htm>`_.
The Python interface writes a file in the Sqlite format.
That interface in Python asks the user of Dismod-AT to
make lists of ages, priors, covariates, random effects, and more,
all such that they dovetail in the SQL database that it builds.

The goal of this API is to make building a Dismod-AT db
file and reading from that file less error-prone. It achieves
that goal by defining larger data structures to represent
the statistical model, fits, and sets of random variables.
If the data structure is filled out, then all the numbers will
work out in the db file.

The best side-effect of making the interface less delicate
is that it clarifies the statistics. Namely,

 * The statistical Model is a collection of random variables
   that describe rates, random effects, and three kinds of covariates,
   all on age-time grids.

 * A single draw from that Model is a Var, which is a collection of
   draws for all of the random variables in the model. This structure
   is shared by the initial guess, the scale variables, the fit,
   and the "truth var."

The main client for this API is the model builder for the
Cascade, the global model.
