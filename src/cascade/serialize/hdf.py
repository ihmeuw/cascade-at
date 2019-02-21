from itertools import product

import numpy as np
from numpy import nan

from cascade.model.dismod_groups import DismodGroups
from cascade.model.smooth_grid import SmoothGrid
from cascade.model.var import Var
from cascade.model.priors import DENSITY_ID_TO_PRIOR


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
    # h5py recommends using S for arrays of fixed-size strings.
    if isinstance(values, np.ndarray) and not np.issubdtype(values.dtype, np.number):
        values = values.astype("S")
    else:
        if len(values) > 0 and isinstance(values[0], str):
            values = np.array(values, dtype="S")
        else:
            values = np.array(values, dtype=np.float)

    base_name = f"{dim_type}{len(values)}"
    dim_idx = 0
    dim_name = f"{base_name}_{dim_idx}"
    while dim_name in hdf_group:
        compare_axis = hdf_group[dim_name]
        if np.allclose(values, compare_axis):
            return compare_axis
        dim_idx += 1
        dim_name = f"{base_name}_{dim_idx}"

    dim_value = hdf_group.create_dataset(dim_name, data=values)
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


PRIOR_KINDS = ["value", "dage", "dtime"]
PRIOR_NAMES = ["density", "mean", "std", "lower", "upper", "eta", "nu"]
DENSITY_TO_ID = {di.density: di_key for (di_key, di) in DENSITY_ID_TO_PRIOR.items()}


def write_smooth_grid(hdf_group, smooth_grid, name):
    ages = smooth_grid.ages
    times = smooth_grid.times

    data = np.zeros((len(PRIOR_KINDS), len(ages), len(times), len(PRIOR_NAMES)), dtype=np.float)
    for kind_idx, kind in enumerate(["value", "dage", "dtime"]):
        one_prior = getattr(smooth_grid, kind).grid
        for row_idx, row in one_prior.iterrows():
            aidx = np.where(ages == row.age)[0]
            tidx = np.where(times == row.time)[0]
            density_id = DENSITY_TO_ID.get(row.density, nan)
            to_write = [density_id, row["mean"], row["std"], row.lower, row.upper, row.eta, row.nu]
            data[kind_idx, aidx, tidx, :] = to_write

    ds = hdf_group.create_dataset(name, data=data)

# The dict is ordered, so this order is the same as the shape when
    # calling create_dataset.
    scales = dict(
        prior_kind=PRIOR_KINDS,
        age=ages,
        time=times,
        prior=PRIOR_NAMES,
    )
    for scale_idx, kind in enumerate(scales.keys()):
        dimension = write_dimension(hdf_group, kind, scales[kind])
        ds.dims.create_scale(dimension, kind)
        ds.dims[scale_idx].attach_scale(dimension)

    ds.attrs["cascade_type"] = "SmoothGrid"
    return ds


def read_smooth_grid(ds):
    if ds.attrs["cascade_type"] != "SmoothGrid":
        raise SerializationError(f"Expected {ds} to be a SmoothGrid")

    # The 1 and 2 change when the shape of create_dataset changes.
    ages = ds.dims[1][0][:]
    times = ds.dims[2][0][:]
    smooth = SmoothGrid(ages, times)
    for kind_idx, kind in enumerate(PRIOR_KINDS):
        grid = getattr(smooth, kind).grid
        for aidx, tidx in product(range(len(ages)), range(len(times))):
            age = ages[aidx]
            time = times[tidx]
            to_set = dict(zip(PRIOR_NAMES, ds[kind_idx, aidx, tidx, :]))
            to_set["density"] = DENSITY_ID_TO_PRIOR.get(int(to_set["density"]), nan).density
            grid.loc[(grid.age == age) & (grid.time == time), list(to_set.keys())] = to_set.values()

    return smooth
