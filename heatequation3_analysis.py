from pathlib import Path
import re
from itertools import product

from firedrake import SpatialCoordinate, sin, cos, exp, pi, UnitIntervalMesh
from src.analysis import ConvergenceAnalyser

# Define paths from which we are retrieving the results
# and the structure of the files
PATH = (
    "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/"
    "Project/programming/solver/tests/heatfiles/HE"
)

PATTERN = (
    rf"heat_n(?P<n>\d+)"
    rf"_dt(?P<dt>[0-9eE+\-.]+)"
    rf"_sw(?P<sw>\d+)"
    rf"(?:_(?P<idx>\d+))?"
    rf"\.h5$"
)


KEYS = ["n", "dt", "sw"]
KEYS_TYPE = [int, float, int]

# We need to create an an exact mesh in order to write in UFL
x = SpatialCoordinate(UnitIntervalMesh(1))[0]
U_EXACT = lambda t: sin(pi * x) * exp(x * t)

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
    temporal_key="dt",
    temporal_val=DT_REF,
    sweep_key="sw",
    sweep_val=SW_REF,
)

# TEMPORAL CONVERGENCE STUDY
N_REF = 100
for sw in (1, 2, 3, 4, 5, 6):
    analyser.temporal_error_convergence(
        temporal_key="dt",
        spatial_key="n",
        spatial_val=N_REF,
        sweep_key="sw",
        sweep_val=sw,
    )

# SWEEP CONVERGENCE STUDY
for dt_ref in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    analyser.sweep_error_convergence(
        sweep_key="sw",
        spatial_key="n",
        spatial_val=N_REF,
        temporal_key="dt",
        temporal_val=dt_ref,
    )
