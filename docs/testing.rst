.. _testing:

Testing
=======

Running Tests
-------------

Running Unit Tests
^^^^^^^^^^^^^^^^^^

Unit tests can run from either the root directory or ``tests`` subdirectory
using ``pytest``. Note the following useful options for pytest. The first
couple are custom flags we created.

 * ``pytest --ihme`` This is a flag we created that enables those tests which
   we would run within the IHME environment. If you write a test
   that calls IHME databases, you must include the `ihme` fixture in order
   for that unit test to run. This guarantees that when Jenkins runs without
   the `--ihme` flag, none of the tests it runs require the IHME databases.

 * ``pytest --dismod`` This is a flag we created that enables those tests which
   require having a command-line Dismod-AT running. Using ``ihme`` turns on
   ``dismod``.

 * ``pytest --signals`` This is a flag we created that enables those tests which
   turns off tests that send UNIX signals to test failure modes. It's useful
   on the Mac, which helpfully offers to inform Apple of application failure.

The rest are standard options, but they are so important that I'm listing them
here.

 * ``pytest ---log-cli-level debug`` Captures log messages nicely within unit tests.

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
^^^^^^^^^^^^^^^^^^^^^^^^
There is a separate directory for acceptance tests. It's called ``acceptance``
in the unit directory. Here, too, run ``pytest``, but it will take longer
and do thread tests, which are tests from one interaction to a response.

Unit and acceptance tests are run with the ``--ihme`` flag
turned on, just before the end of installation. If they fail, then
installation fails. Be sure to run unit tests on the cluster
with ``--ihme``, even if they pass in Tox, which runs a subset
of tests.


Structure of Tests
------------------

Testing structure follows the component structure of the code,
but there are a few tests that outweigh others in importance
because they are system integration tests. If we look at the
larger architectural parts, those system integration tests
mock out different pieces. The larger architectural
parts are:

 1. Main success scenario (MSS), that does a fit and simulate
    with Dismod-AT.

 2. Input data of various kinds

    a. Bundle data records

    b. IHME databases of mortality.

    c. EpiViz-AT settings.

 3. Interface with the Dismod-AT db file


Main Success Scenario
^^^^^^^^^^^^^^^^^^^^^

There is a single file that runs the core set of steps
for the wrapper, using no inputs from external sources.
It does the *first two* of these steps. As we work through the
main success scenario, we should make it do all of the steps.

 1. Generate input data with Dismod-AT predict.
 2. Fit that data.
 3. Generate simulations.
 4. Fit those simulations.
 5. Summarize simulation outputs.
 6. Create posterior data.

It's in ``tests/model/main_success_scenario.py``.
It's set up to run through different types of models
and different combinations of input parameters. It does
a fractional-factorial experiment on those parameters, working
up to seeing how two parameters interact, and whether the
code still runs.

This same script generates files with timings on how
long it takes Dismod-AT to do a fit, for a given set of
parameters and data.


Test Settings Parsing
^^^^^^^^^^^^^^^^^^^^^

This mocks the creation of EpiViz-AT settings and
then runs stochastically-generated settings through
the builder for models, all the way to writing a Dismod-AT db file.
It's in ``test_construct_model.py``


Live Tests against Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These use a real MVID, and pull settings and data
for it in order to build a database, in ``test_estimate_locations.py``.


Testing the Dismod-AT DB File Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These tests skip any IHME database interaction.
The redo the extensive tests included with Dismod-AT,
but they do it using the internal interface.
This is what tells us our internal interface works.
In ``test_dismod_examples.py``.


What's Missing
^^^^^^^^^^^^^^

There should be a test that creates settings and input
data, and runs completely through the main scenario.
This would save us from waiting for the IHME databases
to send data and would exercise the later part of the main
success scenario, which isn't covered enough yet.
