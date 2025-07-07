import re
from pathlib import Path
from typing import List
from firedrake import *
from firedrake.checkpointing import CheckpointFile


# # general pattern ‘heat_n{n}_dt{dt}_sw{sw}_{idx?}.h5’
# PAT = re.compile(
#     r"heat_n(?P<n>\d+)_dt(?P<dt>[0-9eE\+\-]+)_sw(?P<sw>\d+)" r"(?:_(?P<idx>\d+))?\.h5$"
# )

base_dir = Path(
    "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/solver/tests/heatfiles/HE"
)
file_name = "heat_n"
pattern = f"{file_name}_n(?P<n>\d+)_dt(?P<dt>\d*(?:\d[eE]-\d+)?)_sw(?P<sw>\d+).*(?:_(?P<idx>\d+))?\.h5$"
deg = 4


class ConvergenceAnalyser:
    """
    In this class we will be addign one file_name, one pattern the function names
    which we have defined in our simulations and the real solution to those functions,
    and we will retrieve all the informatio concerning to convergence and so on.
    """

    def __init__(
        self,
        fun_sol: Function | List[Function],
        function_names: str | List[str],
        file_name: str,
        file_path: Path,
        pattern,
    ):
        self.real_f = [fun_sol] if not isinstance(fun_sol, list) else fun_sol
        self.name = file_name
        self.pattern = pattern
        self.function_names = (
            [function_names] if not isinstance(function_names, List) else function_names
        )
        self.file_path = file_path
        self.len_function_names = len(self.function_names)
        self.file_list = self._file_list()

    def _file_list(self) -> dict:
        file_list = {}
        for f in self.file_path.glob("*.h5"):
            m = self.file_name.match(f.name)
            if not m:
                continue
            key = (int(m["n"]), float(m["dt"]), int(m["sw"]))
            idx = int(m["idx"] or 0)
            # we save the biggest number
            if key not in file_list or idx > file_list[key][1]:
                file_list[key] = (f, idx)
        return file_list

    def _file_extractor(self, path: Path) -> tuple:
        with CheckpointFile(str(path), "r") as afile:
            mesh = afile.load_mesh()
            hist = afile.get_timestepping_history(mesh, "u")
            tfin = hist["time"][-1]
            idx = hist["index"][-1]
            u_num = afile.load_function(mesh, "u", idx=idx)

        return (mesh, hist, tfin, idx, u_num)

    def _group_file_extractor(self, path: Path) -> dict:
        """
        Might not be very usefull as if we want to upload several
        functions at the same time it might use a lot of ram that
        is actually not needed, the best thing is import one by one
        and then deliting that function in order to retrieve the next
        one. Also this might be helpfull in case we have defined
        several functions, so we access the path
        """
        data = {}
        with CheckpointFile(str(path), "r") as afile:
            data["mesh"] = afile.load_mesh()
            for i, fun_name in enumerate(self.function_names):
                data["hist"] = data.setdefault("hist", []).append(
                    afile.get_timestepping_history(data.get("mesh"), fun_name)
                )
                data["tfin"] = data.setdefault("tfin", []).append(
                    data.get("hist")[i]["time"][-1]
                )
                data["idx"] = data.setdefault("idx", []).append(
                    data.get("hist")[i]["index"][-1]
                )
                u_num = afile.load_function(
                    data.get("mesh"), fun_name, idx=data["idx"][i]
                )

        return data

    @staticmethod
    def L2_error(mesh, tfin, f_exact, f_num):
        x = SpatialCoordinate(mesh)[0]
        V = FunctionSpace(mesh, "CG", degree=deg)
        u_exact = Function(V).interpolate(sin(pi * x) * exp(x * tfin))
        return errornorm(f_exact, f_num, norm_type="L2")
