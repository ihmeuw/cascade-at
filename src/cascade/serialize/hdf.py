from h5py import File


class axis_cache:
    def __init__(self, hdf_destination, cache_name):
        self._file_group = hdf_destination.create_group(cache_name)

    def get_axis(self, ages, times):
        pass  # either return a known one or make one.



def save_dismod_groups(hdf_destination, dg):
    for group_name, dismod_group in dg.items():
        sub_hdf_group = hdf_destination.create_group(group_name)
        for key, grid in dismod_group.items():
            if group_name == "rate":
                keys = dict(rate=key)
            elif group_name == "random_effect":
                keys = dict(rate=key[0], location=key[1])
            elif group_name == "alpha":
                keys = dict(covariate=key[0], rate=key[1])
            elif group_name == "beta":
                keys = dict(covariate=key[0], integrand=key[1])
            elif group_name == "alpha":
                keys = dict(covariate=key[0], integrand=key[1])
            else:
                raise RuntimeError(f"Unknown group name {group_name}.")
            # priors have 3 grids per group.
            save_single_grid(sub_hdf_group, keys, grid)


def save_single_grid(hdf_destination, keys, dismod_age_time_grid):
    if the age_time_grid has 1 dimension, then save in 2d, else 3d.
