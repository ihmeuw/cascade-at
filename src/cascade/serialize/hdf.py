from itertools import product

import numpy as np

from cascade.model.dismod_groups import DismodGroups
from cascade.model.var import Var


class SerializationError(Exception):
    """An error serializing data."""


def write_group(hdf_group, dismod_group):
    """
    Writes a DismodGroups object into an HDF Group. Assumes that there
    is nothing in the HDF Group. This arranges all the names within that
    HDF Group.

    Args:
        hdf_group (h5py.Group): The HDF Group into which to write.
        dismod_group (DismodGroups): The Dismod objects to write.
    """
    for group_name, group in dismod_group.items():
        for key, item in group.items():
            if isinstance(key, str):
                key_name = key
                description = dict(dismod_group=group_name, key0=key, key1=None)
            elif key[1] is None:
                key_name = key[0]
                description = dict(dismod_group=group_name, key0=key[0], key1=None)
            else:
                key_name = "_".join(str(k) for k in key)
                description = dict(dismod_group=group_name, key0=key[0], key1=key[1])
            name = f"{group_name}_{key_name}"
            ds = write_var(hdf_group, item, name)
            # Add to the dataset the keys that DismodGroups uses to track
            # each value.
            for desc_key, desc_value in description.items():
                if desc_value is not None:
                    try:
                        ds.attrs[desc_key] = desc_value
                    except TypeError:
                        raise RuntimeError(f"Could not write {desc_value} as attr of type {type(desc_value)}")


def read_group(hdf_group):
    """
    Reads a DismodGroup of Var.

    Args:
        hdf_group (h5py.Group): The HDF Group into which to write.

    Returns:
        DismodGroups
    """
    dismod_group = DismodGroups()
    for group_name, group in dismod_group.items():
        datasets = [ds_name for ds_name in hdf_group.keys() if ds_name.startswith(group_name)]
        for ds_name in datasets:
            ds = hdf_group[ds_name]
            if "dismod_group" in ds.attrs:
                key0 = ds.attrs["key0"]
                if "key1" in ds.attrs:
                    try:
                        key = (str(key0), int(ds.attrs["key1"]))
                    except ValueError:
                        key = (str(key0), str(ds.attrs["key1"]))
                elif group_name == "random_effect":
                    key = (str(key0), None)
                else:  # Rate
                    key = str(key0)
                group[key] = read_var(ds)
    return dismod_group


def write_dimension(hdf_group, dim_type, values):
    """Given an HDF group, write a set of ages or times,
    possibly sharing a set of ages or times that already exist."""
    base_name = f"{dim_type}{len(values)}"
    dim_idx = 0
    dim_name = f"{base_name}_{dim_idx}"
    while dim_name in hdf_group:
        compare_axis = hdf_group[dim_name]
        if np.allclose(values, compare_axis):
            return compare_axis
        dim_idx += 1
        dim_name = f"{base_name}_{dim_idx}"

    dim_value = hdf_group.create_dataset(dim_name, (len(values),), dtype=np.float)
    dim_value[:] = values
    return dim_value


def write_var(hdf_group, var, name):
    """Writes a var into an HDF group by writing an age, time, and
    values datasets."""
    ages = var.ages
    times = var.times

    ds = hdf_group.create_dataset(name, (len(ages), len(times)), dtype=np.float)
    for aidx, tidx in product(range(len(ages)), range(len(times))):
        ds[aidx, tidx] = var[ages[aidx], times[tidx]]

    for scale_idx, kind in enumerate(["age", "time"]):
        dimension = write_dimension(hdf_group, kind, [ages, times][scale_idx])
        ds.dims.create_scale(dimension, kind)
        ds.dims[scale_idx].attach_scale(dimension)

    ds.attrs["cascade_type"] = "Var"
    return ds


def read_var(ds):
    if "cascade_type" not in ds.attrs or ds.attrs["cascade_type"] != "Var":
        raise SerializationError(f"Expected {ds} to be a var")

    ages = ds.dims[0][0]
    times = ds.dims[1][0]
    var = Var(ages, times)
    for aidx, tidx in product(range(ds.shape[0]), range(ds.shape[1])):
        var[ages[aidx], times[tidx]] = ds[aidx, tidx]

    return var
