"""
Tests writing to and reading from a Dismod db file.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Float, Integer, String, MetaData, ForeignKey
from sqlalchemy.sql import select, update, join, bindparam

from cascade.testing_utilities import make_execution_context
from cascade.model.grids import AgeTimeGrid, PriorGrid
from cascade.model.rates import Smooth
import cascade.model.priors as priors
from cascade.dismod.db.wrapper import _get_engine, DismodFile


def test_interactive_smooth():
    ec = make_execution_context()
    dismod_db = DismodFile(_get_engine(None))

    grid = AgeTimeGrid([0, 1, 5, 10, 20, 50, 100], [1990, 1995, 2000])
    d_time = PriorGrid(grid)
    d_age = PriorGrid(grid)
    value = PriorGrid(grid)
    d_age[:, :].prior = priors.Uniform(float("-inf"), float("inf"), 0)
    d_time[:, :].prior = priors.Uniform(float("-inf"), float("inf"), 0)
    value[:, :].prior = priors.Uniform(0, 1, 0)
    smooth0 = Smooth(value, d_age, d_time)


    grid1 = AgeTimeGrid([0, 0.19, 1, 5, 10, 20, 50, 120], [1995, 2000, 2005])
    value1 = PriorGrid(grid1)
    value1[:,:].prior = priors.Gaussian(0.1, 0.05)
    smooth1 = Smooth(value1, d_age, d_time)

    # CHOICE: Does the smooth have a name or get a name from its owner at
    # the moment of serialization.
    smooth0.write(dismod_db.ostream(), "smooth0")
    smooth1.write(dismod_db.ostream(), "smooth1")
    dismod_db.order_ages_and_times()

    smooth2 = Smooth.read(dismod_db.istream(), "smooth0")
    smooth3 = Smooth.read(dismod_db.istream(), "smooth1")
    assert smooth0 == smooth2
    assert smooth1 == smooth3
    assert smooth2 != smooth3


metadata = MetaData()
age_table = Table('age', metadata,
    Column('age_id', Integer, primary_key=True),
    Column('age', Float),
)
grid_table = Table('grid', metadata,
  Column('grid_id', Integer, primary_key=True),
  Column('age_id', None, ForeignKey('age.age_id')),
  Column('prior', String, nullable=False),
)
age_translation = Table("age_translation", metadata,
  Column("old_id", Integer, primary_key=True),
  Column("new_id", Integer)
)


def test_fix_ages_times_additive():
    """
    This is a strategy for getting ages and times correct.
    Every time it saves, it adds to the list of ages and times,
    and then uses the id of those. Then it fixes them later.

    This uses an update command to fix the ages, instead of writing
    new tables. That choice is made b/c it seems less possible to
    reorder the records.
    """
    mem = True
    engine_loc = ":memory:" if mem else "zz.db"
    if not mem and os.path.exists(engine_loc):
        os.unlink(engine_loc)

    engine = create_engine(f"sqlite:///{engine_loc}", echo=False)
    metadata.bind = engine
    metadata.create_all()
    with engine.connect() as conn:
        ages = [0.0, 0.19, 5.0, 10.0, 2.0, 0.0, 5.0, 10.0]
        make_age = pd.DataFrame({
            "age_id": np.array(list(range(len(ages))), dtype=np.int),
            "age": np.array(ages, dtype=np.double),
        })
        age_converted = [
            {"age_id": int(arec["age_id"]), "age": float(arec["age"])}
            for arec
            in make_age.to_dict(orient="records")
        ]
        conn.execute(age_table.insert(), age_converted)

        make_grid = pd.DataFrame({
            "grid_id": list(range(len(ages))),
            "age_id": list(range(len(ages))),
            "prior": list(map(str, list(range(len(ages)))))
        })
        conn.execute(grid_table.insert(), make_grid.to_dict(orient="records"))

    # Fix the age table itself so it has only the few values it needs.
    with engine.connect() as read:
        all_ages = pd.read_sql_table("age", read)
        age_table.drop(read)

    unique_ages_np = all_ages["age"].unique()
    unique_ages_np.sort()
    age_to_write = pd.DataFrame({
        "new_id": list(range(len(unique_ages_np))),
        "age": unique_ages_np,
    })

    with engine.connect() as new_age_conn:
        age_table.create(new_age_conn)
        new_age_conn.execute(
            age_table.insert(),
            age_to_write.rename(columns={"new_id": "age_id"}).to_dict(orient="records"))

    # Create an id translation table from the two age dfs.
    # age_merge is length of all_ages
    age_merge = pd.merge_asof(all_ages.sort_values("age"), age_to_write, on="age").drop(columns=["age"])
    age_merge = age_merge.sort_values("age_id").rename(columns={"age_id": "old_id"}).reset_index(drop=True)

    with engine.connect() as write_translation:
        ok_to_write = [{str(k): int(v) for (k, v) in age_translation_record.items()}
                       for age_translation_record in age_merge.to_dict(orient="records")]
        age_translation_insert = age_translation.insert().compile()
        write_translation.execute(age_translation_insert, ok_to_write)

        # Once that is in place, it's one command to update any table.
        # stmt = grid_table.update().where(grid_table.c.age_id==select([age_translation.c.new_id]).where(
        #     age_translation.c.old_id==grid_table.c.age_id
        # ).as_scalar())
        # write_translation.execute(stmt)
        write_translation.execute(f"update grid set age_id=(select new_id from age_translation where old_id=age_id);")

    if not mem:
        engine.dispose()
        engine = create_engine(f"sqlite:///{engine_loc}", echo=True)

    with engine.connect() as checker:
        with_age = select([grid_table.c.grid_id, age_table.c.age, grid_table.c.prior]).where(
            grid_table.c.age_id == age_table.c.age_id
        )
        ans = dict()
        for row in checker.execute(with_age):
            ans[row[2]] = row[1]
        key = {f"{idx}": v for (idx, v) in enumerate(ages)}
        for k, v in key.items():
            assert np.isclose(ans[k], v)


def test_fix_ages_times_by_value():
    """
    This is a strategy for getting ages and times correct.
    It saves with an extra column that is the floating point
    value of age and time. Then it sets the IDs at the end.
    """
    engine = create_engine("sqlite:///:memory:", echo=True)


if __name__ == "__main__":
    test_fix_ages_times_additive()
