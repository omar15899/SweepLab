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
DT_LIST = [1e-4]  # paso temporal
SWEEPS = [6]  # barridos SDC por paso
DEGREE = [1]  # grado polinómico FEM
M = 6  # nodos de colocalización
T_FINAL = 0.05
PREC_TYPE = "MIN-SR-FLEX"


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

    # def _define_node_time_boundary_setup(self):
    #     if not self.PDEs.boundary_conditions:
    #         return [], {}

    #     bcs = []
    #     local_bcs = {}
    #     for bc in self.PDEs.boundary_conditions:
    #         if isinstance(bc, DirichletBC):
    #             bc_function_space = bc.function_space()
    #             # Índice del bloque p y, si procede, subcomponente vectorial
    #             if bc_function_space.value_shape == ():
    #                 p_idx, sub_idx = (bc_function_space.index or 0), None
    #             else:
    #                 p_idx, sub_idx = (*bc._indices, None)[
    #                     :2
    #                 ]  # sub_idx solo informativo

    #             for m_node in range(self.M):
    #                 # slot temporal correspondiente en W
    #                 idx_in_W = p_idx + m_node * self.lenV
    #                 V_slot = self.W.sub(idx_in_W)

    #                 # ⚠️ reconstruimos SIEMPRE sobre el PADRE del slot (V_slot),
    #                 # no sobre V_slot.sub(sub_idx). Firedrake mantiene internamente el índice.
    #                 local_bc = bc.reconstruct(V=V_slot)
    #                 bcs.append(local_bc)
    #                 local_bcs.setdefault(idx_in_W, []).append(local_bc)

    #         elif isinstance(bc, EquationBC):
    #             pass
    #         else:
    #             raise Exception("your bc is not accepted.")

    #     return tuple(bcs), local_bcs

    # @staticmethod
    # @PETSc.Log.EventDecorator("sweep_unique_execution")
    # def _sweep(s):
    #     apply_to = getattr(s, "_apply_to", None)  # ← el padre MixedFunction
    #     if apply_to is None:
    #         raise RuntimeError("Internal error: missing _apply_to in solver.")

    #     # Aplica BCs en el PADRE (Mixed), no en la subfunción
    #     for bc in getattr(s, "_my_bcs", ()):
    #         bc.apply(apply_to)

    #     s.solve()

    #     # Reaplícalas por si el solver las “rompe” (opcional pero seguro)
    #     for bc in getattr(s, "_my_bcs", ()):
    #         bc.apply(apply_to)

    # @staticmethod
    # @PETSc.Log.EventDecorator("sweep_unique_execution")
    # def _sweep(s):
    #     # Aplica BCs "manuales" sobre la incógnita local (u_m) si las hay
    #     for bc in getattr(s, "_manual_bcs", ()):
    #         bc.apply(getattr(s, "_my_u", None))
    #     s.solve()
    #     for bc in getattr(s, "_manual_bcs", ()):
    #         bc.apply(getattr(s, "_my_u", None))

    ### Opt 2 - neww
    # @PETSc.Log.EventDecorator("_setup_paralell_sweep_solver")
    # def _setup_paralell_sweep_solver(self):
    #     """
    #     Construye los solvers de barrido en paralelo.

    #     Claves:
    #     - Para cada (p, m) usamos SIEMPRE el subespacio exacto de W: V_slot = self.W.sub(idx)
    #     y tanto la incógnita u_m como el test v_m viven en ese mismo objeto.
    #     - Las BCs NO se pasan al NonlinearVariationalProblem para evitar el error de
    #     "bc space does not match the test function space" en el Jacobiano global.
    #     En su lugar, se crean copias "limpias" sobre V_slot (sin índices de Mixed)
    #     y se aplican manualmente antes y después de cada s.solve() en _sweep.
    #     - Jacobiano explícito J_sweep para asegurar que los espacios de prueba/ensayo
    #     son exactamente V_slot.
    #     """
    #     deltat = self.deltat
    #     tau = self.tau
    #     t0 = self.t_0_subinterval
    #     f = self.PDEs.f
    #     Q = self.Q
    #     Q_D = self.Q_D

    #     self.sweep_solvers = []
    #     self.R_sweep = []

    #     for p, f_i in enumerate(f):
    #         for m in range(self.M):
    #             idx = p + m * self.lenV
    #             V_slot = self.W.sub(idx)
    #             u_m = self.u_k_act.subfunctions[idx]
    #             v_m = TestFunction(V_slot)

    #             left = (
    #                 inner(u_m, v_m)
    #                 - deltat
    #                 * self.scale
    #                 * Q_D[m, m]
    #                 * f_i(t0 + tau[m] * deltat, u_m, v_m)
    #             ) * dx
    #             right = inner(self.u_0.subfunctions[idx], v_m)
    #             for j in range(self.M):
    #                 jdx = p + j * self.lenV
    #                 coeff = Q[m, j] - self.scale * Q_D[m, j]
    #                 right += (
    #                     deltat
    #                     * coeff
    #                     * f_i(
    #                         t0 + tau[j] * deltat, self.u_k_prev.subfunctions[jdx], v_m
    #                     )
    #                 )
    #             right = right * dx

    #             R_sweep = left - right
    #             self.R_sweep.append(R_sweep)

    #             # du_m = TrialFunction(V_slot)
    #             # J_sweep = derivative(R_sweep, u_m, du_m)

    #             bcs_local_list = tuple(self.bcs_V_2.get(idx, []))

    #             problem_m = NonlinearVariationalProblem(
    #                 R_sweep,
    #                 u_m,
    #                 bcs=None,
    #             )
    #             solver = NonlinearVariationalSolver(
    #                 problem_m,
    #                 solver_parameters=(
    #                     {"snes_type": "newtonls", "snes_rtol": 1e-8, "ksp_type": "cg"}
    #                     if not self.solver_parameters
    #                     else self.solver_parameters
    #                 ),
    #             )

    #             # Guarda referencias para aplicar BCs correctamente
    #             solver._apply_to = self.u_k_act  # <— PADRE (MixedFunction)
    #             solver._my_bcs = bcs_local_list  # <— BCs ya “indexadas” a W.sub(idx)
    #             self.sweep_solvers.append(solver)

    # @PETSc.Log.EventDecorator("_setup_paralell_sweep_solver")
    # def _setup_paralell_sweep_solver(self):
    #     deltat, tau, t0 = self.deltat, self.tau, self.t_0_subinterval
    #     f, Q, Q_D = self.PDEs.f, self.Q, self.Q_D

    #     self.sweep_solvers = []
    #     self.R_sweep = []

    #     for p, f_i in enumerate(f):
    #         for m in range(self.M):
    #             idx = p + m * self.lenV

    #             V_slot = self.W.sub(idx)
    #             u_m = self.u_k_act.subfunctions[idx]
    #             v_m = TestFunction(V_slot)

    #             left = (
    #                 inner(u_m, v_m)
    #                 - deltat
    #                 * self.scale
    #                 * Q_D[m, m]
    #                 * f_i(t0 + tau[m] * deltat, u_m, v_m)
    #             ) * dx

    #             right = inner(self.u_0.subfunctions[idx], v_m)
    #             for j in range(self.M):
    #                 jdx = p + j * self.lenV
    #                 coeff = Q[m, j] - self.scale * Q_D[m, j]
    #                 right += (
    #                     deltat
    #                     * coeff
    #                     * f_i(
    #                         t0 + tau[j] * deltat, self.u_k_prev.subfunctions[jdx], v_m
    #                     )
    #                 )
    #             right *= dx

    #             R_sweep = left - right
    #             self.R_sweep.append(R_sweep)

    #             du_m = TrialFunction(V_slot)
    #             J_sweep = derivative(R_sweep, u_m, du_m)

    #             # BCs reconstruidas para este slot
    #             bcs_local_all = tuple(self.bcs_V_2.get(idx, []))

    #             # Particiona: las que viven exactamente en V_slot vs. subespacios
    #             bcs_direct = tuple(
    #                 bc for bc in bcs_local_all if bc.function_space() == V_slot
    #             )
    #             bcs_manual = tuple(
    #                 bc for bc in bcs_local_all if bc.function_space() != V_slot
    #             )

    #             problem_m = NonlinearVariationalProblem(
    #                 R_sweep, u_m, bcs=bcs_direct, J=J_sweep
    #             )
    #             solver = NonlinearVariationalSolver(
    #                 problem_m,
    #                 solver_parameters=(
    #                     {"snes_type": "newtonls", "snes_rtol": 1e-8, "ksp_type": "cg"}
    #                     if not self.solver_parameters
    #                     else self.solver_parameters
    #                 ),
    #             )

    #             # Guarda referencias para aplicar las BCs "manuales" a u_m
    #             solver._my_u = u_m
    #             solver._manual_bcs = bcs_manual

    #             self.sweep_solvers.append(solver)
