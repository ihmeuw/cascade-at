.. _testing:

Testing
=======

Running Unit Tests
------------------

Unit tests can run from either the root directory or ``tests`` subdirectory
using ``pytest``. Note the following useful options for pytest. The first
couple are custom flags we created.

 * ``pytest --ihme`` This is a flag we created that enables those tests which
   we would run within the IHME environment.

 * ``pytest --dismod`` This is a flag we created that enables those tests which
   require having a command-line Dismod-AT running. Using ``ihme`` turns on
   ``dismod``.

 * ``pytest --signals`` This is a flag we created that enables those tests which
   turns off tests that send UNIX signals to test failure modes. It's useful
   on the Mac, which helpfully offers to inform Apple of application failure.

The rest are standard options, but they are so important that I'm listing them
here.

 * ``pytest --log-level debug`` Captures log messages nicely within unit tests.

 * ``pytest --pdb`` This flag asks to drop into the debugger when an exception
   happens in a unit test. Very helpful when using tests for test-driven development.

 * ``pytest --capture=no`` This allows stdout, stderr, and logging to be printed
   when running tests.

 * ``pytest -k partial_name`` This picks out all tests whose names contain the
   letters "partial_name".

 * ``pytest --lf`` Run the last set of failing tests.

 * ``pytest --ff`` Run the last set of failing tests, and then run the rest
   of the tests.

 * ``pytest -x`` Die on the first failure.


In order to make a test that relies on IHME databases, use the global fixture
called ``ihme``::

    def test_get_ages(ihme):
        gbd_round = 12
        ages = get_gbd_age_groups(gbd_round)

This test will automatically be disabled until the ``--ihme`` flag is on.


Running Acceptance Tests
------------------------
There is a separate directory for acceptance tests. It's called ``acceptance``
in the unit directory. Here, too, run ``pytest``, but it will take longer
and do thread tests, which are tests from one interaction to a response.
