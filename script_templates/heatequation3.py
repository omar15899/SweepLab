import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from firedrake import *
from itertools import product
from src.sdc import SDCSolver
from src.specs import PDESystem
from datetime import datetime

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


def solve_heat_pde(
    dt: float,
    n_cells: int,
    nsweeps: int,
    M: int,
    Tfinal: float,
    *,
    is_parallel: bool = True,
    prectype: str = "MIN-SR-FLEX",
    degree: int = 4,
    analysis: bool = False,
    mode: str = "vtk",
    folder_name: str | None = None,
    path_name: str | None = None,
):

    dt_str = f"{dt:.2e}".replace(".", "p")
    Tfinal_str = f"{Tfinal:.2e}".replace(".", "p")
    file_name = (
        f"heat_n{n_cells}_dt{dt_str}_sw{nsweeps}_nodes{M}_deg{degree}_"
        f"prectype{prectype}_tfinal{Tfinal_str}_par{str(is_parallel)}"
    )

    folder_name = folder_name or "HE_" + time_str
    path_name = path_name or (
        "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/"
        "programming/solver_scripts_results/tests/heatfiles"
    )

    mesh = IntervalMesh(n_cells, length_or_left=0.0, right=1.0)
    X = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", degree=degree)

    def u_exact(Xcoord, t):
        return sin(pi * Xcoord[0]) * exp(Xcoord[0] * t)

    f_obtained = lambda t: (X[0] + pi**2 - t**2) * sin(pi * X[0]) * exp(
        X[0] * t
    ) - 2 * pi * t * cos(pi * X[0]) * exp(X[0] * t)

    def f_heat(t, u, v):
        return -inner(grad(u), grad(v)) + f_obtained(t) * v

    u0 = Function(V, name="u0").interpolate(u_exact(X, 0.0))
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    pde = PDESystem(
        mesh=mesh,
        V=V,
        coord=X,
        f=f_heat,
        u0=u0,
        boundary_conditions=(bc,),
        name="Heat1D_MS",
    )

    solver = SDCSolver(
        mesh=mesh,
        PDEs=pde,
        M=M,
        dt=dt,
        is_parallel=is_parallel,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_rtol": 1e-14,
            "snes_atol": 1e-16,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        prectype=prectype,
        file_name=file_name,
        folder_name=folder_name,
        path_name=path_name,
        analysis=analysis,
        mode=mode,
    )

    real_exp = u_exact if analysis else None
    solver.solve(Tfinal, nsweeps, real_exp)


N_CELLS = [8]
DT_LIST = [1e-2]
SWEEPS = [6]
DEGREE = [1]
M = 6
TFINAL = 0.1

for n, dt, sw, deg in product(N_CELLS, DT_LIST, SWEEPS, DEGREE):
    solve_heat_pde(
        dt=dt,
        n_cells=n,
        nsweeps=sw,
        M=M,
        Tfinal=TFINAL,
        is_parallel=True,
        prectype="MIN-SR-FLEX",
        degree=deg,
        analysis=True,
        mode="vtk",
        folder_name=None,
        path_name=None,
    )
