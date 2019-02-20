from itertools import product

import numpy as np

from cascade.model.var import Var


class SerializationError(Exception):
    """An error serializing data."""


def write_var(hdf_group, var, name):
    ages = var.ages
    times = var.times

    ds = hdf_group.create_dataset(name, (len(ages), len(times)), dtype=np.float)
    for aidx, tidx in product(range(len(ages)), range(len(times))):
        ds[aidx, tidx] = var[ages[aidx], times[tidx]]

    for scale_idx, kind in enumerate(["age", "time"]):
        hdf_group[f"{name}_{kind}"] = [ages, times][scale_idx]
        ds.dims.create_scale(hdf_group[f"{name}_{kind}"], kind)
        ds.dims[scale_idx].attach_scale(hdf_group[f"{name}_{kind}"])

    ds.attrs["cascade_type"] = "Var"

    return None


def read_var(hdf_group, name):
    ds = hdf_group[name]
    if "cascade_type" not in ds.attrs or ds.attrs["cascade_type"] != "Var":
        raise SerializationError(f"Expected {name} to be a var")

    ages = ds.dims[0][0]
    times = ds.dims[1][0]
    var = Var(ages, times)
    for aidx, tidx in product(range(ds.shape[0]), range(ds.shape[1])):
        var[ages[aidx], times[tidx]] = ds[aidx, tidx]
    return var
