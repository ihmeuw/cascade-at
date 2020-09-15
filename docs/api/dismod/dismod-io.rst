.. _dismod-interface:

Interface
---------

The base interface is ``DismodSQLite``, and the input and output class has getters and setters
for each of the tables (``DismodIO``, not documented here).

.. autoclass:: cascade_at.dismod.api.dismod_sqlite.DismodSQLite
   :members:
   :undoc-members:
   :show-inheritance:


To use a ``DismodIO(DismodSQLite)`` interface, you can do

.. code-block:: python

   from cascade_at.dismod.api.dismod_io import DismodIO
   file = 'my_database.db'
   db = DismodIO(file)

   # Tables are stored as attributes, e.g.
   db.data
   db.age
   db.time
   db.prior
