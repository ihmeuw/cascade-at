.. _versioning:

Versioning
==========

We are using the date to version and ``setuptools_scm`` to create the version.
This means tagging versions with git, and then
the rest should work. You will see it mentioned in ``setup.py`` and
``cascade.core.__init__.py``, where ``__version__`` is defined. It's not
defined in ``cascade.__init__.py`` because this uses a namespace package,
so there isn't an init there.


A new version is created with each install and tagged
by date, as enforced by the install script (``install.sh``). Were you to do it by hand,
the command is::

    git tag -a v18.12.23 -m "Release with uncertainty in results"

After the tag is merged, the docs and application, itself, will update.
