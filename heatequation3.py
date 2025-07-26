from firedrake import *
from itertools import product
from src.sdc import SDCSolver
from src.specs import PDESystem
from datetime import datetime

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


def solve_heat_pde(
    dt,
    n_cells,
    nsweeps,
    M,
    Tfinal,
    is_parallel=True,
    prectype="MIN-SR-FLEX",
    degree=4,
    full_collocation=False,
):

    dt_str = f"{dt:.2e}".replace(".", "p")
    Tfinal_str = f"{Tfinal:.2e}".replace(".", "p")
    file_name = f"heat_n{n_cells}_dt{dt_str}_sw{nsweeps}_nodes{M}_degreepol{degree}_prectype{prectype}_tfinal{Tfinal_str}_is_parallel{str(is_parallel)}"

    mesh = IntervalMesh(n_cells, length_or_left=0, right=1)
    x = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", degree=degree)

    u_exact = lambda x, t: sin(pi * x[0]) * exp(x[0] * t)
    f_obtained = lambda t: (x[0] + pi**2 - t**2) * sin(pi * x[0]) * exp(
        x[0] * t
    ) - 2 * pi * t * cos(pi * x[0]) * exp(x[0] * t)

    def f_heat(t, u, v):
        # return -inner(grad(u), grad(v))
        # return v
        return -inner(grad(u), grad(v)) + f_obtained(t) * v

    u0 = Function(V, name="u0").interpolate(u_exact(x, 0.0))
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    pde = PDESystem(
        mesh=mesh,
        V=V,
        coord=SpatialCoordinate(mesh),
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
        folder_name=f"HE5_{time_str}",
        path_name=(
            "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/"
            "programming/solver/tests/heatfiles"
        ),
        full_collocation=full_collocation,
    )

    solver.solve(Tfinal, nsweeps, u_exact, analysis=True)
    solver.solve(Tfinal, nsweeps)


# N_CELLS = [4, 8, 16, 25, 50, 100, 200, 400, 800]
# DT_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# SWEEPS = [1, 2, 3, 4, 5, 6]
# DEGREE = [1, 2, 3, 4]


N_CELLS = [8]
DT_LIST = [1e-2]
SWEEPS = [6]
DEGREE = [1]
M = 6
TFINAL = 0.5


for n, dt, sw, deg in product(N_CELLS, DT_LIST, SWEEPS, DEGREE):
    solve_heat_pde(
        dt=dt,
        n_cells=n,
        nsweeps=sw,
        M=M,
        Tfinal=TFINAL,
        is_parallel=False,
        prectype="MIN-SR-FLEX",
        degree=deg,
        full_collocation=True,
    )
