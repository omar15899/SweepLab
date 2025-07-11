from firedrake import *
from src.sdc import SDCSolver
from src.specs import PDESystem

Tfinal = 0.5
dt = 1e-4
M = 4
nsweeps = 3

mesh = UnitSquareMesh(100, 100)
x = SpatialCoordinate(mesh)
xx, yy = x

V = FunctionSpace(mesh, "CG", 1)


u0_expr = exp(-150 * ((xx - 0.25) ** 2 + (yy - 0.75) ** 2)) + 0.6 * sin(pi * xx) * sin(
    2 * pi * yy
)
u0 = Function(V, name="u0").interpolate(u0_expr)


def f_heat(t, u, v, k=1.0, Q1=2.0, Q2=1.5):
    src1 = Q1 * sin(2 * pi * t) * sin(pi * xx) * sin(pi * yy)
    src2 = Q2 * exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)
    return -k * inner(grad(u), grad(v)) + (src1 + src2) * v


bc = DirichletBC(V, Constant(0.0), "on_boundary")


pde = PDESystem(
    mesh=mesh,
    V=V,
    coord=x,
    f=f_heat,
    u0=u0,
    boundary_conditions=[bc],
    name="heat_equation",
)

solver = SDCSolver(
    mesh=mesh,
    PDEs=pde,
    M=M,
    dt=dt,
    is_local=True,
    prectype="MIN-SR-FLEX",
    file_name="heat_equation_sdc",
    folder_name="HE",
    path_name=(
        "/Users/omarkhalil/Desktop/Universidad/"
        "ImperialCollege/Project/programming/solver/heatSDC"
    ),
    is_vtk=True,
)
solver.solve(T=Tfinal, sweeps=nsweeps)
print("Simulación terminada")
