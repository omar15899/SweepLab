from firedrake import *
from src.sdc import SDCSolver

Tfinal = 0.5
dt = 1e-4
M = 4
nsweeps = 3

mesh = UnitSquareMesh(100, 100)
x = SpatialCoordinate(mesh)
xx, yy = x[0], x[1]
V = FunctionSpace(mesh, "CG", 1)

u0_expr = exp(-150 * ((xx - 0.25) ** 2 + (yy - 0.75) ** 2)) + 0.6 * sin(pi * xx) * sin(
    2 * pi * yy
)


def f0(t, u, v, k=1.0, Q=2.0, w=pi):
    # La v hay que meterla después dentro
    source = Q * sin(w * t) * sin(pi * xx) * sin(pi * yy)
    # source = Constant(0)
    return -k * inner(grad(u), grad(v)) + source * v


def f1(t, u, v, k=1.0, Q1=2.0, Q2=1.5):
    src1 = Q1 * sin(2 * pi * t) * sin(pi * xx) * sin(pi * yy)
    src2 = Q2 * exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)
    return -k * inner(grad(u), grad(v)) + (src1 + src2) * v


bcs = DirichletBC
solver = SDCSolver(
    mesh,
    V,
    f=f1,
    u0=u0_expr,
    boundary_conditions=bcs,
    M=M,
    dt=dt,
    is_parallel=True,
    prectype="MIN-SR-FLEX",
    file_name="heat_equation_sdc",
    folder_name="HE",
    path_name="/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/solver/heatSDC",
    is_vtk=True,
)

uT = solver.solve(T=Tfinal, sweeps=nsweeps)
print("donessiiuuu")
