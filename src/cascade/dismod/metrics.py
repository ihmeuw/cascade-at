"""
Given a DismodFile object, collect numbers that characterize the size
of the work.
"""
import logging
from argparse import ArgumentParser
from inspect import getdoc
from pathlib import Path
from textwrap import indent

from cascade.core.log import getLoggers
from .constants import INTEGRAND_COHORT_COST
from .db.wrapper import DismodFile, get_engine

CODELOG, MATHLOG = getLoggers(__name__)
METRICS = list()


def metric(retrieval):
    """Decorator records which functions are measuring the db_file."""
    METRICS.append((retrieval.__name__, retrieval))
    return retrieval


@metric
def age_integration_points(db_file):
    """This table is re-created by every Dismod-AT function to contain
    the total number of age integration points."""
    return len(db_file.age_avg)


@metric
def age_extent(db_file):
    """Maximum age minus minimum age."""
    age = db_file.age
    return age.age.max() - age.age.min()


@metric
def time_extent(db_file):
    """Maximum time minus minimum time."""
    time = db_file.time
    return time.time.max() - time.time.min()


@metric
def smooth_count(db_file):
    """Total number of Smooth Grids, which are grids of prior distributions."""
    return len(db_file.smooth)


@metric
def children(db_file):
    """Count of the number of child locations."""
    option = db_file.option
    # The file may have either the parent_node_id or parent_name set,
    # and it might be "none" as a string, or "", or None.
    parent_id = option[option.option_name == "parent_node_id"].option_value.iloc[0]
    parent_name = option[option.option_name == "parent_node_name"].option_value.iloc[0]
    node = db_file.node
    try:
        parent_node_id = int(parent_id)
    except ValueError:
        parent_node_id = None
    if parent_node_id is None:
        # Then try the other one.
        parent_node_record = node[node.node_name == parent_name]
        if len(parent_node_record) == 1:
            try:
                parent_node_id = int(parent_node_record)
            except ValueError:
                parent_node_id = None
    if parent_node_id is not None:
        return len(node[node.parent == parent_node_id])
    else:
        return 0


@metric
def rate_count(db_file):
    """How many rates are nonzero."""
    return len(db_file.rate[db_file.rate.parent_smooth_id.notnull()])


@metric
def random_effect_points(db_file):
    """How many grid points have random effects. This takes each smooth
    grid, multiplies it by the number of children, and counts every
    age-time point in the grid. It's the number of variables that come
    from random effects."""
    smooth = db_file.smooth.reset_index(drop=True)
    rate = db_file.rate[db_file.rate.child_smooth_id.notnull()]
    rate = rate.assign(child_smooth_id=rate.child_smooth_id.astype(int))
    child_random = rate.merge(
        smooth, left_on="child_smooth_id", right_on="smooth_id", how="inner"
    )
    child_cnt = (child_random.n_age * child_random.n_time).sum()
    location_children = children(db_file)
    if location_children > 0:
        # Each child smooth is used once for each of the children.
        child_cnt = child_cnt * location_children
    nslist_pair = db_file.nslist_pair[db_file.nslist_pair.smooth_id.notnull()]
    nslist_pair = nslist_pair.assign(smooth_id=nslist_pair.smooth_id.astype(int))
    smooth = db_file.smooth.reset_index(drop=True)
    many_random = nslist_pair.merge(smooth, on="smooth_id", how="inner")
    child_cnt += (many_random.n_age * many_random.n_time).sum()
    return child_cnt


@metric
def variables(db_file):
    """Total number of variables to solve for."""
    return len(db_file.var)


@metric
def avgint(db_file):
    """How many predictions to make."""
    return len(db_file.avgint)


def data_records(db_file):
    """Data records counts. Extent marks those that have either age
    or time extent, and cohort marks those that are more expensive
    than the primary rates."""
    integrand = db_file.integrand.reset_index(drop=True)
    data = db_file.data.merge(integrand, on="integrand_id", how="left")
    has_extent = ((data.age_upper != data.age_lower) |
                  (data.time_lower != data.time_upper))
    cohort_cost = {k for (k, v) in INTEGRAND_COHORT_COST.items() if v}
    has_cohort_cost = data.integrand_name.isin(cohort_cost)
    by_integrand = data.groupby(by="integrand_name").size().to_dict()
    return dict(
        data_cnt=len(data),
        data_extent_cohort=(has_extent & has_cohort_cost).sum(),
        data_point_cohort=(~has_extent & has_cohort_cost).sum(),
        data_extent_primary=(has_extent & ~has_cohort_cost).sum(),
        data_point_primary=(~has_extent & ~has_cohort_cost).sum(),
        **by_integrand,
    )


def options(db_file):
    """Several entries are from the options table."""
    relevant = {
        "zero_sum_random", "derivative_test_fixed", "derivative_test_random",
        "max_num_iter_fixed", "max_num_iter_random", "tolerance_fixed",
        "tolerance_random", "quasi_fixed", "bound_frac_fixed",
        "limited_memory_max_fixed", "bound_random",
    }
    opt = db_file.option[["option_name", "option_value"]]
    return dict(opt[opt.option_name.isin(relevant)].to_records(index=False))


def gather_metrics(db_file):
    # This code is a terrible reason to kill a job, so catch all
    # exceptions, but report them.
    try:
        name_to_value = options(db_file)
    except Exception:
        CODELOG.exception(f"Could not collect options metrics")
        name_to_value = dict()
    try:
        name_to_value.update(data_records(db_file))
    except Exception:
        CODELOG.exception(f"Could not collect data records metrics")
    for metric_name, retrieval in METRICS:
        try:
            name_to_value[metric_name] = retrieval(db_file)
        except Exception:
            CODELOG.exception(f"Could not collect metric {metric_name}")
    return name_to_value


def parser():
    parse = ArgumentParser(
        description="Measure quantities to characterize a db_file."
    )
    parse.add_argument("db_file", type=Path, nargs="?")
    parse.add_argument("--list-metrics", action="store_true",
                       description="Tell me about the metrics.")
    parse.add_argument("-v", action="store_true")
    return parse


def entry():
    """This is installed as a script in the Python environment
    so that you can print metrics on any db file."""
    args = parser().parse_args()
    if args.v:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    if args.list_metrics:
        metrics = METRICS + [
            ("data_records", data_records),
            ("options", options)
        ]
        for metric_name, retrieval in metrics:
            print(f"{metric_name}\n{indent(getdoc(retrieval), '    ')}")
        exit(0)
    if not args.db_file:
        parser().print_help()
    if not args.db_file.exists():
        print(f"File {args.db_file} not found")
        exit(1)
    db_file = DismodFile(get_engine(args.db_file))
    values = gather_metrics(db_file)
    key_len = max([len(key) for key in values.keys()])
    for metric_name, value in values.items():
        print(f"{metric_name:{key_len + 1}s}{value}")
