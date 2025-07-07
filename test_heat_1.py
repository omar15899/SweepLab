from firedrake import *
from itertools import product
from src.sdc import SDCSolver
from src.specs import PDESystem


Tfinal = 0.5
dt = 1e-4
M = 4
nsweeps = 3
n_cells = 100


def solve_heat_pde1(dt, n_cells, nsweeps, M, Tfinal, degree=4):

    dt_str = f"{dt:.0e}"
    file_name = f"heat_n{n_cells}_dt{dt_str}_sw{nsweeps}"

    mesh = IntervalMesh(n_cells, length_or_left=0, right=1)
    x = SpatialCoordinate(mesh)[0]

    V = FunctionSpace(mesh, "CG", degree=degree)

    u_exact = lambda t: sin(pi * x) * exp(x * t)
    f_obtained = lambda t: (x + pi**2 - t**2) * sin(pi * x) * exp(
        x * t
    ) - 2 * pi * t * cos(pi * x) * exp(x * t)

    def f_heat(t, u, v):
        return -inner(grad(u), grad(v)) + f_obtained(t) * v

    u0 = Function(V, name="u0").interpolate(u_exact(0.0))
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
        prectype="MIN-SR-FLEX",
        file_name=file_name,
        folder_name="HE",
        path_name=(
            "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/"
            "programming/solver/tests/heatfiles"
        ),
    )

    solver.solve(Tfinal, nsweeps)


N_CELLS = [25, 50, 100, 200, 400, 800]
DT_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
SWEEPS = [1, 2, 3, 4, 5, 6]

for n, dt, sw in product(N_CELLS, DT_LIST, SWEEPS):
    solve_heat_pde1(dt=dt, n_cells=n, nsweeps=sw, M=4, Tfinal=0.5)
