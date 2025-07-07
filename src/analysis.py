import re
from pathlib import Path
from firedrake import *
from firedrake.checkpointing import CheckpointFile

base_dir = Path(
    "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/solver/tests/heatfiles/HE"
)

# general pattern ‘heat_n{n}_dt{dt}_sw{sw}_{idx?}.h5’
PAT = re.compile(
    r"heat_n(?P<n>\d+)_dt(?P<dt>[0-9eE\+\-]+)_sw(?P<sw>\d+)" r"(?:_(?P<idx>\d+))?\.h5$"
)

file_name = "heat_n"
pattern = f"{file_name}_n(?P<n>\d+)_dt(?P<dt>\d*(?:\d[eE]-\d+)?)_sw(?P<sw>\d+).*(?:_(?P<idx>\d+))?\.h5$"
deg = 4


class ConvergenceAnalyser:

    def __init__(self, file_name, pattern, function_names):
        self.name = file_name
        self.pattern = pattern

    latest_chk = {}
    for f in base_dir.glob("*.h5"):
        m = PAT.match(f.name)
        if not m:
            continue
        key = (int(m["n"]), float(m["dt"]), int(m["sw"]))
        idx = int(m["idx"] or 0)
        # we save the biggest number
        if key not in latest_chk or idx > latest_chk[key][1]:
            latest_chk[key] = (f, idx)

    def file_extractor(self, path: Path) -> float:
        with CheckpointFile(str(path), "r") as afile:
            mesh = afile.load_mesh()
            hist = afile.get_timestepping_history(mesh, "u")
            tfin = hist["time"][-1]
            idx = hist["index"][-1]
            u_num = afile.load_function(mesh, "u", idx=idx)

        return (mesh, hist, tfin, idx, u_num)

    def L2_error(self):
        x = SpatialCoordinate(mesh)[0]
        V = FunctionSpace(mesh, "CG", degree=deg)
        u_exact = Function(V).interpolate(sin(pi * x) * exp(x * tfin))
        return errornorm(u_exact, u_num, norm_type="L2")

    print(
        l2_error(
            "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/solver/tests/heatfiles/HE/heat_equation_sdc_3.h5"
        )
    )
