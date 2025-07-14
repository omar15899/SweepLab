from pathlib import Path
import re
from itertools import product

from firedrake import SpatialCoordinate, sin, cos, exp, pi, UnitIntervalMesh
from src.analysis import ConvergenceAnalyser

# Define paths from which we are retrieving the results
# and the structure of the files
PATH = "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/solver/tests/heatfiles/HE"

PATTERN = re.compile(
    r"^heat_n(?P<n>\d+)"
    r"_dt(?P<dt>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"_sw(?P<sw>\d+)"
    r"_nodes(?P<nodes>\d+)"
    r"_degreepol(?P<degreepol>\d+)"
    r"_prectype(?P<prectype>[A-Za-z0-9\-]+)"
    r"_tfinal(?P<tfinal>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:_(?P<idx>\d+))?"
    r"\.h5$"
)


KEYS = ["n", "dt", "sw", "nodes", "degreepol", "prectype", "tfinal"]
KEYS_TYPE = [int, float, int, int, int, str, float]


# We need to create an an exact mesh in order to write in UFL
def U_EXACT(t: int | float, a: SpatialCoordinate):
    x = a[0]
    return sin(pi * x) * exp(x * t)


# INSTANTIATE CONVERGENCE ANALYSER
analyser = ConvergenceAnalyser(
    file_path=PATH,
    pattern=PATTERN,
    keys=KEYS,
    keys_type=KEYS_TYPE,
    f_exact_ufl=U_EXACT,
    function_names="u",
)


# SPATIAL CONVERGENCE STUDY
DT_REF = 1e-5
SW_REF = 1
analyser.spatial_error_convergence(
    spatial_key="n",
    spatial_lower_bound=4,
    temporal_key="dt",
    temporal_val=DT_REF,
    sweep_key="sw",
    sweep_val=SW_REF,
    degreepol=1,
)

# TEMPORAL CONVERGENCE STUDY
N_REF = 400
SW_REF = 1
analyser.temporal_error_convergence(
    temporal_key="dt",
    spatial_key="n",
    spatial_val=N_REF,
    sweep_key="sw",
    sweep_val=SW_REF,
    degreepol=1,
)

# SWEEP CONVERGENCE STUDY
# for dt_ref in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
N_REF = 400
DT_REF = 1e-1
for dt in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    analyser.sweep_error_convergence(
        sweep_key="sw",
        spatial_key="n",
        spatial_val=N_REF,
        temporal_key="dt",
        temporal_val=dt,
        degreepol=1,
    )
