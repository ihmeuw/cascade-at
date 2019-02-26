.. _epiviz-settings:

EpiViz-AT Settings
==================

The EpiViz-AT Settings are the set of all choices a user
makes in the EpiViz-AT user interface. This is how the interface
sends those choices to the command-line EpiViz-AT.

You can download and examine those settings for any model
version ID, as described in `get-settings`.

The list of all possible settings is in
https://github.com/ihmeuw/cascade/blob/develop/src/cascade/input_data/configuration/form.py
where any setting with the word ``Dummy`` is being ignored.

Any setting that is unset, meaning the user has used the
close box to ensure it is greyed-out in the EpiViz-AT user interface,
will be missing from the EpiViz-AT settings sent to the command-line
program, and the program understands that it should use a default.
