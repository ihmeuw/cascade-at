.. _dismod:

Dismod Database API
===================

This module describes the interface for reading and writing
from dismod databases. Dismod-AT works on SQLite databases,
and we need a user-friendly way to write data to and extract data
from these databases. It is also important to make sure
we have all of the correct columns and column types.

The input tables and column types are explained
`here <https://bradbell.github.io/dismod_at/doc/input.htm>`_,
and the output tables and column types are explained
`here <https://bradbell.github.io/dismod_at/doc/data_flow.htm>`_.

We mimic that table metadata
`here <https://github.com/ihmeuw/cascade-at/blob/develop/src/cascade_at/dismod/api/table_metadata.py>`_,
and then build an interface on top of it for easy reading and writing.



