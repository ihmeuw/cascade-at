.. _testing:

Testing
=======

Running Unit Tests
------------------

Unit tests can run from either the root directory or ``tests`` subdirectory
using ``pytest``. Note the following useful options for pytest.

 * ``pytest --pdb`` This flag asks to drop into the debugger when an exception
   happens in a unit test. Very helpful when using tests for test-driven development.

 * ``pytest --capture=no`` This allows stdout, stderr, and logging to be printed
   when running tests.

 * ``pytest -k partial_name`` This picks out all tests whose names contain the
   letters "partial_name".

 * ``pytest --ihme`` This is a flag we created that enables those tests which
   we would run within the IHME environment.

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
