from firedrake import *
from src.sdc import SDCSolver

Tfinal = 10.0
dt = 1e-2
M = 4
nsweeps = 3

mesh = UnitSquareMesh(100, 100)
x = SpatialCoordinate(mesh)
xx, yy = x[0], x[1]
V = FunctionSpace(mesh, "CG", 1)

u0_expr = exp(-150 * ((xx - 0.25) ** 2 + (yy - 0.75) ** 2)) + 0.6 * sin(pi * xx) * sin(
    2 * pi * yy
)


def f(t, u, v, k=1.0, Q=2.0, w=pi):
    # La v hay que meterla después dentro
    # source = Q * sin(w * t) * sin(pi * xx) * sin(pi * yy)
    source = Constant(0)
    return -k * inner(grad(u), grad(v)) + source * v


bcs = DirichletBC
solver = SDCSolver(
    mesh,
    V,
    f=f,
    u0=u0_expr,
    bcs=bcs,
    M=M,
    dt=dt,
    is_paralell=True,
    prectype="MIN-SR-NS",
)

uT = solver.solve(T=Tfinal, sweeps=nsweeps)
print("donessiiuuu")
