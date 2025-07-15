from firedrake import *
from itertools import product
from src.sdc import SDCSolver
from src.specs import PDESystem
from datetime import datetime

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


def solve_heat_pde1(
    dt,
    n_cells,
    nsweeps,
    M,
    Tfinal,
    prectype="MIN-SR-FLEX",
    degree=4,
):

    dt_str = f"{dt:.0e}"
    Tfinal_str = f"{Tfinal:.0e}"
    file_name = f"heat_n{n_cells}_dt{dt_str}_sw{nsweeps}_nodes{M}_degreepol{degree}_prectype{prectype}_tfinal{Tfinal}"

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
        prectype=prectype,
        file_name=file_name,
        folder_name=f"HE5_{time_str}",
        path_name=(
            "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/"
            "programming/solver/tests/heatfiles"
        ),
    )

    solver.solve(Tfinal, nsweeps)


# N_CELLS = [1, 4, 8, 16, 25, 50, 100, 200, 400, 800]
# DT_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# SWEEPS = [1, 2, 3, 4, 5, 6]
# DEGREE = [1, 2, 3, 4]


N_CELLS = [10]
DT_LIST = [1e-2]
SWEEPS = [4]
DEGREE = [4]


TFINAL = 0.1
M = 4

for n, dt, sw, deg in product(N_CELLS, DT_LIST, SWEEPS, DEGREE):
    solve_heat_pde1(
        dt=dt,
        n_cells=n,
        nsweeps=sw,
        M=M,
        Tfinal=TFINAL,
        prectype="MIN-SR-FLEX",
        degree=deg,
    )


"""

vale, creo que hay un problema serio, y ese problema tiene que ver con cómo se plotea cada error, me gustaría que modificaras la función de plotting para generar las curvas correctas para cada uno de los errores, se muy consciente de que tiene que verse bien y tiene que maximizar la información que me puedes 

"""
