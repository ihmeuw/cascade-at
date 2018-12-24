.. _versioning:

Versioning
==========

We are using the date to version. This means tagging versions with git, and then
the rest should work. The command is::

    git tag -a v18.12.23 -m "Release with uncertainty in results"

After the tag is merged, the docs and application, itself, will update.
