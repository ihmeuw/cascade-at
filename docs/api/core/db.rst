.. _db:

Shared Functions
================

The central computation team and the scientific computing team at IHME maintain several
packages that Cascade-AT relies on. These are not open source, so we can't use them in Travis CI,
or in building the docs. In order to get around this, there is a class
that wraps a module and will only use it if its importable.

.. autoclass:: cascade_at.core.db.ModuleProxy
   :members: __init__, __getattr__, __dir__