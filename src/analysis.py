import re
from pathlib import Path
from typing import List
from firedrake import *
from firedrake.checkpointing import CheckpointFile


# # general pattern ‘heat_n{n}_dt{dt}_sw{sw}_{idx?}.h5’
# PAT = re.compile(
#     r"heat_n(?P<n>\d+)_dt(?P<dt>[0-9eE\+\-]+)_sw(?P<sw>\d+)" r"(?:_(?P<idx>\d+))?\.h5$"
# )

file_path = Path(
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
        pattern: str,
        file_path: Path,
    ):
        self.real_f = [fun_sol] if not isinstance(fun_sol, list) else fun_sol
        self.pattern = pattern
        self.function_names = (
            [function_names] if not isinstance(function_names, List) else function_names
        )
        self.file_path = file_path
        self.len_function_names = len(self.function_names)
        self.file_list = self._file_list()

    @staticmethod
    def L2_error(mesh, tfin, f_exact, f_num):
        x = SpatialCoordinate(mesh)[0]
        V = FunctionSpace(mesh, "CG", degree=deg)
        u_exact = Function(V).interpolate(sin(pi * x) * exp(x * tfin))
        return errornorm(f_exact, f_num, norm_type="L2")


analyer = ConvergenceAnalyser()
