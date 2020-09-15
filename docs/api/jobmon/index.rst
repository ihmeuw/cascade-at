.. _jobmon:


Utilizing Jobmon
^^^^^^^^^^^^^^^^

Unfortunately, we can't document these functions because ``jobmon`` is not yet open source and
the ``sphinx-autodoc`` extension won't work. To be continued once it's released... but for now
please see the source code directly `here <https://github.com/ihmeuw/cascade-at/blob/develop/src/cascade_at/jobmon/workflow.py>`_.


Jobmon Workflows
""""""""""""""""
At the highest level, we need to make a workflow from a
:ref:`cascade-commands`.
This utilizes the Jobmon Guppy version, which allows us to create
"task templates". In the Guppy terminology, a Cascade-AT workflow is considered
to come from a ``dismod-at`` "tool".


Resources
"""""""""
Using jobmon requires some knowledge of the amount of cluster resources that a job
will use. Right now, there is no resource prediction algorithm implemented in Cascade-AT.
The base resources are the same for all jobs, and then some are increased or decreased
depending on the specific task, as options
passed to :py:class:`~cascade_at.cascade.cascade_operations._CascadeOperation`.

