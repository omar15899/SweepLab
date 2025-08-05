import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from firedrake import *
from itertools import product
from src.sdc import SDCSolver  # tus módulos
from src.specs import PDESystem
from datetime import datetime

# ────────────────────── datos de la barrida ────────────────────── #
N_CELLS = [8]  # nº de celdas 1‑D
DT_LIST = [1e-2]  # paso temporal
SWEEPS = [4]  # barridos SDC por paso
DEGREE = [1]  # grado polinómico FEM
M = 4  # nodos de colocalización
T_FINAL = 0.05
PREC_TYPE = "MIN-SR-NS"


# ────────────────────── solución exacta fabricada ───────────────── #
#   u(x,t) =  sin(πx)      e^{-t}
#   v(x,t) =  sin(2πx)     e^{-t}
#  ∂u/∂t  = ∆u + f₁   →   f₁ = (−1 + π²) sin(πx) e^{-t}
#  ∂v/∂t  = ∆v + f₂   →   f₂ = (−1 + (2π)²) sin(2πx) e^{-t}
#
#  Ambos campos se anulan en x = 0,1  ⇒  C.C. de Dirichlet homogéneas
#
def u_exact_expr(x, t):  # expresión UFL
    return sin(pi * x[0]) * exp(-t)


def v_exact_expr(x, t):
    return sin(2 * pi * x[0]) * exp(-t)


def f1_obtained(x, t):
    return (-1.0 + pi**2) * sin(pi * x[0]) * exp(-t)


def f2_obtained(x, t):
    return (-1.0 + (2 * pi) ** 2) * sin(2 * pi * x[0]) * exp(-t)


# RHS en forma débil  (f_i(t, u, v_test))
def rhs_u(t, u, v):
    return -inner(grad(u), grad(v)) + f1_obtained(x, t) * v


def rhs_v(t, u, v):
    return -inner(grad(u), grad(v)) + f2_obtained(x, t) * v


# ------------------------------------------------------------------


def solve_mixed_heat_system(
    dt,
    n_cells,
    nsweeps,
    M,
    Tfinal,
    degree=1,
    is_parallel=True,
    prectype="MIN-SR-NS",
    full_collocation=True,
):
    # 1. malla y espacios
    mesh = IntervalMesh(n_cells, 0.0, 1.0)
    global x  # <- usado dentro de los rhs
    x = SpatialCoordinate(mesh)

    V1 = FunctionSpace(mesh, "CG", degree)
    V2 = FunctionSpace(mesh, "CG", degree)
    V = V1 * V2  # MixedFunctionSpace ⟨u,v⟩

    # 2. condiciones iniciales (único Function sobre V)
    u0 = Function(V, name="u0")
    u0.subfunctions[0].interpolate(u_exact_expr(x, 0.0))
    u0.subfunctions[1].interpolate(v_exact_expr(x, 0.0))

    # 3. B.C. de Dirichlet homogéneas sobre cada sub‑espacio
    bc_u = DirichletBC(V.sub(0), Constant(0.0), "on_boundary")
    bc_v = DirichletBC(V.sub(1), Constant(0.0), "on_boundary")

    # 4. empaquetamos el sistema
    pde = PDESystem(
        mesh=mesh,
        V=V,
        coord=x,
        f=[rhs_u, rhs_v],  # lista de RHS
        u0=u0,
        boundary_conditions=(bc_u, bc_v),
        name="Heat1D_Mixed",
    )

    # 5. solver SDC
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"mixedHeat_n{n_cells}_dt{dt:.2e}_sw{nsweeps}_nodes{M}_deg{degree}"

    solver = SDCSolver(
        mesh=mesh,
        PDEs=pde,
        M=M,
        dt=dt,
        is_parallel=is_parallel,
        prectype=prectype,
        file_name=file_name,
        folder_name=f"MHE_{now}",
        path_name="/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/"
        "programming/solver/tests/heatfiles",  # cambia a tu ruta
        analysis=full_collocation,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    # Nota: no pasamos solución exacta porque la rutina interna asume
    #       un solo campo y fallaría con Mixed; bastará ver convergencia.
    solver.solve(Tfinal, nsweeps)


# ────────────────────── barrido de casos ───────────────────────── #
for n, dt, sw, deg in product(N_CELLS, DT_LIST, SWEEPS, DEGREE):
    solve_mixed_heat_system(
        dt=dt,
        n_cells=n,
        nsweeps=sw,
        M=M,
        Tfinal=T_FINAL,
        degree=deg,
        is_parallel=True,
        prectype=PREC_TYPE,
        full_collocation=True,
    )
