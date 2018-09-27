.. _meid-quota:

Update the modelable_entity_id quota in the db
===============================================

By default the ``modelable_entity_id`` can have at most 10 models defined.
This cap is stored in the db table ``epi.modelable_entity_quota`` as ``quota``.
The quota can be updated by calling the stored procedure, ``epi_model_quota_add``.
As an example, this SQL CALL statement will change the quota value for meid=1744 to 20::

    call epi.epi_model_quota_add(1744, 20, 'at_cascade');

The user ``at_cascade`` must have "execute" permission on the stored procedure.
The db team has given this permission to user ``at_cascade``:: 

    GRANT EXECUTE ON PROCEDURE `epi`.`epi_model_quota_add` TO 'at_cascade'@'%';  

in both of the dismod_at db's, dev and prod: ``epi-db-d01`` and ``epiat-db-p01``



