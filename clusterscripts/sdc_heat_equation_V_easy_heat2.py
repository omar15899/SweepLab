import os
import sys
import time
from pathlib import Path
from typing import Literal, Callable
from firedrake import *
from firedrake.petsc import OptionsManager
from dataclasses import dataclass
from typing import Union, List
from dataclasses import dataclass
from typing import Literal
import numpy as np
from FIAT.quadrature import (
    GaussLobattoLegendreQuadratureLineRule,
    RadauQuadratureLineRule,
)
from FIAT.reference_element import DefaultLine

from typing import Iterable, Literal, List
from pathlib import Path
import json
import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from datetime import datetime
from itertools import product


class FileNamer:
    extensions = {
        "checkpoint": ".h5",
        "vtk": ".pvd",
        "pdf": ".pdf",
        "png": ".png",
        "txt": ".txt",
        "json": ".json",
    }

    def __init__(
        self,
        file_name: str = "solution",
        folder_name: str | None = None,
        path_name: str | None = None,
        mode: Literal["checkpoint", "vtk", "pdf"] = "checkpoint",
    ):

        # File saving attributes
        self.file_name = os.path.splitext(file_name)
        self.folder_name = folder_name if folder_name else "solution"
        self.path_name = path_name if path_name else os.getcwd()
        self.mode = mode
        self.file = self._create_unique_path()

        # Asegura la carpeta del fichero principal (.h5/.pvd/…)
        Path(self.file).parent.mkdir(parents=True, exist_ok=True)

        # Log PETSc
        self._log_txt = Path(self.file).with_suffix("").as_posix() + "_log.txt"
        Path(self._log_txt).parent.mkdir(parents=True, exist_ok=True)  # <--- AQUI
        PETSc.Options().setValue("log_view", f":{self._log_txt}")
        if "log_view" not in OptionsManager.commandline_options:
            PETSc.Log.begin()

        # if self.mode not in self.extensions.values():
        #     raise Exception("Invalid mode.")

    def _create_unique_path(self):
        """
        Create correct folder organisation.
        if vtk, we store the solution in different folders
        if chekcpoint, we store the solution in only one folder
        """
        base_name, _ = self.file_name
        base_dir = os.path.join(self.path_name, self.folder_name)
        os.makedirs(base_dir, exist_ok=True)
        if self.mode != "vtk":
            # files with no extension
            all_files = {
                os.path.splitext(name)[0]
                for name in os.listdir(os.path.join(self.path_name, self.folder_name))
            }

            # If the file is a checkpoint, we enumerate the files
            i = 0
            while True:
                file_name = f"{base_name}_{i}"
                if file_name not in all_files:
                    break
                i += 1

            return os.path.join(base_dir, file_name + self.extensions[self.mode])

        else:
            all_folders = {
                name
                for name in os.listdir(self.path_name)
                if os.path.isdir(os.path.join(self.path_name, name))
            }

            i = 0
            while True:
                folder_name = f"{self.folder_name}_{i}"
                if folder_name not in all_folders:
                    break
                i += 1

            vtk_path = os.path.join(self.path_name, folder_name, base_name + ".pvd")
            os.makedirs(os.path.dirname(vtk_path), exist_ok=True)  # <--- AQUI
            return vtk_path


@dataclass
class SDCPreconditioners:
    M: float
    prectype: int | str = "MIN-SR-FLEX"
    tau: np.ndarray | None = None
    tau_type: Literal["lobatto", "radau-left", "radau-right"] | None = "radau-right"

    """
    Optional to use a personal quadrature rule, I have to add more options to 
    the Gauss Lobatto one
    """

    def __post_init__(self):

        if self.tau is None:
            # Calculate collocation nodes in [-1,1] (main parameter in collocation problem)
            if self.tau_type == "lobatto":
                rule = GaussLobattoLegendreQuadratureLineRule(DefaultLine(), self.M)

            elif self.tau_type == "radau-left":
                # Includes the extreme x = −1  ->  tau = 0
                rule = RadauQuadratureLineRule(DefaultLine(), self.M, right=False)

            elif self.tau_type == "radau-right":
                # Includes the extreme x = 1  ->  tau = 1
                rule = RadauQuadratureLineRule(DefaultLine(), self.M, right=True)
            else:
                raise ValueError(f"Unknown quadrature: {self.tau_type!r}")

            self.tau = 0.5 * (
                np.asarray(rule.get_points()).flatten() + 1.0
            )  # Change to [0,1]

        # INstantiate the collocation matrix and the Q_Delta
        self.Q = self._buildQ()
        self.Q_D = self._Q_Delta()

    def _buildQ(self):
        tau = self.tau
        M = self.M

        # Create Vandermonde matrix mxm
        V = np.vander(tau, N=M, increasing=True)

        # Create the integrals of monomials by broadcasting
        exps = np.arange(1, M + 1)
        integrals = tau[:, None] ** exps / exps

        # Calculate lagrange coef
        coef = np.linalg.solve(V, np.eye(M))
        Q = integrals @ coef

        return Q

    # We will include all preconditioners here Q_delta. (MIN-RES)
    def _Q_Delta(self):
        if self.prectype in {0, "DIAG1"}:
            return np.diag(np.diag(self.Q))
        elif self.prectype == "MIN-SR-NS":
            return np.diag(self.tau) / self.M
        elif self.prectype in {"MIN-SR-S", "MIN-SR-FLEX"}:
            # Special case tau0=0
            if np.isclose(self.tau[0], 0.0):
                D = np.diag(self.tau)
                # Added remark 2.11 from paper
                D[0, 0] = 0.0
                return D
            return np.diag(self.tau)
        else:
            raise Exception("there's no other preconditioners defined")


@dataclass
class PDESystem(object):
    """
    V must be the definitive space for that PDE conforming the system.
    If we have one scalar pde, just use FunctionSpace, if we have a
    vector PDE use VectorFunctionSpace, if we have a mixed system use MixedFunctionSpace.
    Could we use also this object to compact the whole potential system?
    ej, we have a system of mixed scalar and vector pde's, could we incorpore
    them inmediately here or better to do it separated


    As Colin said, it's important to take advantage of all Firedrake properties
    and build upon it. So in this case we need to prepare all of the boundary conditions,
    the mixed space and everything of the system just before. We could do it by parts
    instantiating different PDEobj but just for cases in which the boundary conditions
    are not connected. But the recommendation is to just use one object with everything
    setted from before (EVERYTHING = V MIXEDSPACE, ALL BOUNDARY CONDITIONS OVER ALL THE SYSTEM,
    ALL INITIAL CONDITIONS IN JUST ONE FIREDRAKE VECTOR UO, ALL THE BOUNDARY CONDITIONS...)

    Params:
    -------

    IT'S VERY IMPORTANT TO HAVE ALL THE MATHEMATICS AND THE BOUNDARY CONDITIONS PERFECTLY
    DEFINED BEFORE WRITING ANY CODE, THEN, CAREFULLY SHAPE THE PDE'S AS IN DOCUMENTATION.
    IT HAS TO BE DEFINED IN THE MOST GENERAL WAY BEFORE INCORPORATING THEM HERE!

    Mesh: Mesh
        The mesh over which the PDE is defined.

    V: FunctionSpace | VectorFunctionSpace | MixedFunctionSpace
        The function space over which the PDE is defined. If a system of PDEs, use MixedFunctionSpace.
        If we have a system of mixed scalar and vectorial PDE's, we need to create the
        whole MixedFunctionSpace

    coord: SpatialCoordinate
        The spatial coordinates of the mesh, used to define the PDEs.

    f: Function
        The right-hand side function of the PDE, which can depend on time and space. All pde's we want
        to solve must be time dependant (if not SDC makes nonsense), so we should be able to
        describe de ecuation as delta u / delta t = f(u, ...). If we have a system of ecuations,
        we have to define a vectorial function f over the whole mixed space V and define al the functions
        as subfunctions. Also include the names and everything. We will take advantage that all
        Function.subfunctions is an iterable (even if it has just one element. )

    u0: Function
        The initial condition of the PDE, which can be a scalar or vectorial function depending on V.
        If we have a system of mixed PDE's, we can use Function(V) and specify the subfunctions
        as the initial conditions for each subspace.

    boundary_conditions: callable | Iterable[callable]


    IMPORTANT:

    1. Thanks to the way firedrake works and how mixedfunctionspaces are flattened and the
        conceptual equivalence between VFS and FS, the way that we have to input the PDESystem
        is:
        a) mesh: Unique Mesh for all the system.
        b) V: Or 1 FunctionSpace, or 1 VectorFunctionSpace or if several from the previous 2,
            a MixedFunctionSpace.
        c) coord: Unique parametrisation of the mesh
        d) f: Just 1 if V is a 1 FunctionSpace or 1 VectorFunctionSpace (in this second case
            it will have several subfunctions, but VFS is ment to be defined over a same f function
            as even its a VFS with dim 5 for instance, it will only have
            one subfunction as the 5 differents ones are considered by the program
            as just one single function object (staked contigiously in physical memory), thats
            why we have this 1 to 1 correspondence in VFS). In the complementary case (MFS), it needs to be
            a lists of f defined with a 1 to 1 correspondence with each subFS or subVFS of V. the
            u / delta t = f(u, ...). without the dx) and each subspace
            (either FunctionSpace or VectorFunctionSpace), impossible to be
            a nested MixedFunctionSpace.
            --- IS A NORMAL FUNCTION OR LIST OF FUNCTIONS EXPRESSED IN UFL
            IS NOT EVEN A FIREDRAAKES FUNCTION.
        d1) u0: Function defined over all V, here we define the initial conditions
            for all the pde's of the system at the same time, each with one subfunction
            associated. Again, if we have a MFS composed of FS and VFS, as VFS do
            also have just 1 subfunctions, we would not have to think about nested
            loops over this subfunctions. It is some sense also flattened (same behaviour
            as what I have in my notes about it.)
        e) boundary conditions: A non sorted list with all of them. The program will automatically
            handle it.
        d) time_dependent_constants_bts: Same
        e) name: Name of the general system.

    2. At the end, as V defines the whole system
    """

    mesh: Mesh
    V: Union[FunctionSpace, VectorFunctionSpace, MixedFunctionSpace]
    coord: SpatialCoordinate
    f: Callable | List[Callable]
    u0: Function  # (if system is mixed, use Function(V) withc .subfunctions[] already specified)
    boundary_conditions: tuple[DirichletBC | EquationBC]
    time_dependent_constants_bts: Constant | tuple[Constant] | None = None
    name: str | None = None

    def __post_init__(self):
        self._is_Mixed = len(self.V.subspaces) > 1
        # Define boundary conditions in an homogenieus way (always a list)
        self.boundary_conditions = (
            (self.boundary_conditions,)
            if not isinstance(self.boundary_conditions, tuple)
            else self.boundary_conditions
        )
        # check if exactly equivalent to previous.
        if self.time_dependent_constants_bts is None:
            self.time_dependent_constants_bts = tuple()
        elif isinstance(self.time_dependent_constants_bts, Constant):
            self.time_dependent_constants_bts = (self.time_dependent_constants_bts,)
        else:
            self.time_dependent_constants_bts = tuple(self.time_dependent_constants_bts)
        # functions to list
        self.f = [self.f] if not isinstance(self.f, list) else self.f


class SDCSolver(FileNamer, SDCPreconditioners):
    """
    Specific solver for SDC (versión unificada: paralelo sobre V, W sólo si hace falta).
    """

    # ====== __init__ (nuevo) ======
    def __init__(
        self,
        mesh: Mesh,
        PDEs: PDESystem,
        M: int = 4,
        dt: int | float = 1e-3,
        is_parallel: bool = True,
        solver_parameters: dict | None = None,
        prectype: int | str = 0,
        tau: np.ndarray | None = None,
        file_name: str = "solution",
        folder_name: str | None = None,
        path_name: str | None = None,
        mode: Literal["checkpoint", "vtk", "pdf"] = "checkpoint",
        analysis: bool = False,
    ):
        # I/O y precondicionadores
        FileNamer.__init__(
            self,
            file_name=file_name,
            folder_name=folder_name,
            path_name=path_name,
            mode=mode,
        )
        SDCPreconditioners.__init__(self, M=M, prectype=prectype, tau=tau)

        # Atributos básicos
        self.mesh = mesh
        self.PDEs = PDEs
        self.deltat = float(dt)
        self.deltatC = Constant(self.deltat)
        self.is_parallel = is_parallel
        self.solver_parameters = solver_parameters
        self.analysis = analysis

        # Timings/meta
        self._sweep_meta: list[dict] = []
        self._timings_buffer: list[dict] = []

        # Espacios y dimensiones
        self.V = self.PDEs.V
        self.lenV = len(self.V.subspaces) if self.PDEs._is_Mixed else 1

        # BCs sobre V (tal cual definidas por el usuario)
        self.bcs_V = self.PDEs.boundary_conditions

        # --- ¿Necesitamos W? (global o análisis) ---
        self.uses_W = (not self.is_parallel) or self.analysis

        # --- Representación sobre W (opcional) ---
        if self.uses_W:
            self.W = MixedFunctionSpace([self.V] * self.M)
            self.bcs_W, self.bcs_V_2 = self._define_node_time_boundary_setup()

            self.R_sweep = []
            self.R_coll = 0

            self.u_0 = Function(self.W, name="u_0")
            self.u_k_prev = Function(self.W, name="u_k")
            self.u_k_act = Function(self.W, name="u_{k+1}")

            if self.analysis:
                self.u_0_collocation = Function(self.W, name="u_0 collocation")
                self.u_collocation = Function(self.W, name="u_collocation")
                self._init_collocation_from_u0()
            else:
                self.u_0_collocation = None
                self.u_collocation = None
        else:
            self.W = None
            self.bcs_W, self.bcs_V_2 = (), {}
            self.R_sweep = []
            self.R_coll = 0
            self.u_0 = None
            self.u_k_prev = None
            self.u_k_act = None
            self.u_0_collocation = None
            self.u_collocation = None

        # --- Listas nodales sobre V (siempre) ---
        self.U0 = [Function(self.V, name=f"U0[{m}]") for m in range(self.M)]
        self.Uk_prev = [Function(self.V, name=f"Uk_prev[{m}]") for m in range(self.M)]
        self.Uk_act = [Function(self.V, name=f"Uk_act[{m}]") for m in range(self.M)]

        # Condición inicial sobre V (+ sincronización a W si existe)
        for m in range(self.M):
            for p in range(self.lenV):
                self.U0[m].subfunctions[p].interpolate(self.PDEs.u0.subfunctions[p])
                self.Uk_prev[m].subfunctions[p].interpolate(
                    self.PDEs.u0.subfunctions[p]
                )
                self.Uk_act[m].subfunctions[p].interpolate(self.PDEs.u0.subfunctions[p])

        if self.uses_W:
            self._sync_W_from_V()

        # Tiempo y escala
        self.t_0_subinterval = Constant(0.0)
        self.scale = Constant(1.0)

        # Construcción de solvers
        if not self.is_parallel:
            # Global necesita W
            assert self.uses_W, "Global solver requires W."
            self._setup_global_sweep_solver()
        else:
            # Paralelo sobre V (sin construir W para barridos)
            self._setup_parallel_sweep_solver_V()

        if self.analysis:
            # El solver de colocación se formula sobre W
            if not self.uses_W:
                self.W = MixedFunctionSpace([self.V] * self.M)
                self.u_0 = Function(self.W, name="u_0")
                self.u_k_prev = Function(self.W, name="u_k")
                self.u_k_act = Function(self.W, name="u_{k+1}")
                self.u_0_collocation = Function(self.W, name="u_0 collocation")
                self.u_collocation = Function(self.W, name="u_collocation")
                self.bcs_W, self.bcs_V_2 = self._define_node_time_boundary_setup()
                self._sync_W_from_V()
                self._init_collocation_from_u0()
                self.uses_W = True
            self._setup_full_collocation_solver()

    # ====== utilidades de estado ======
    def _init_collocation_from_u0(self):
        if (
            not self.analysis
            or self.u_0_collocation is None
            or self.u_collocation is None
        ):
            return
        for i, (u0c, uc) in enumerate(
            zip(self.u_0_collocation.subfunctions, self.u_collocation.subfunctions)
        ):
            base = self.PDEs.u0.subfunctions[i % self.lenV]
            u0c.interpolate(base)
            uc.interpolate(base)

    def _sync_W_from_V(self):
        """Copiar Uk_* (listas sobre V) hacia subfunciones de u_* (sobre W)."""
        if getattr(self, "W", None) is None or any(
            getattr(self, nm, None) is None for nm in ("u_k_act", "u_k_prev", "u_0")
        ):
            return
        for m in range(self.M):
            for p in range(self.lenV):
                idx = p + m * self.lenV
                self.u_k_act.subfunctions[idx].assign(self.Uk_act[m].subfunctions[p])
                self.u_k_prev.subfunctions[idx].assign(self.Uk_prev[m].subfunctions[p])
                self.u_0.subfunctions[idx].assign(self.U0[m].subfunctions[p])

    def _sync_all_nodes_to_last(self):
        if self.is_parallel:
            last_V = self.Uk_act[-1]
            for coll in (self.Uk_act, self.Uk_prev, self.U0):
                for u in coll:
                    u.assign(last_V)
            return
        last_W = self.u_k_act.subfunctions[-1]
        for sub in (
            *self.u_k_act.subfunctions,
            *self.u_k_prev.subfunctions,
            *self.u_0.subfunctions,
        ):
            sub.assign(last_W)

    def _last_block_as_V(self, *, prev: bool = False) -> Function:
        if self.is_parallel:
            return self.Uk_prev[-1] if prev else self.Uk_act[-1]
        out = Function(self.V)
        src_vec = self.u_k_prev if prev else self.u_k_act
        for p in range(self.lenV):
            idx = p + (self.M - 1) * self.lenV
            out.subfunctions[p].assign(src_vec.subfunctions[idx])
        return out

    def _get_last_state_view(self):
        return self.Uk_act[-1] if self.is_parallel else self.u_k_act.subfunctions[-1]

    def _total_residual_sweep_aggregated(self) -> float:
        if not self.R_sweep:
            return 0.0
        if len(self.R_sweep) == 1:
            vec = assemble(self.R_sweep[0]).riesz_representation()
            return float(norm(vec, norm_type="L2"))
        total = 0.0
        for Rm in self.R_sweep:
            vec = assemble(Rm).riesz_representation()
            total += float(norm(vec, norm_type="L2"))
        return total

    def _write_and_close_log(self):
        """Dump PETSc log al fichero y cerrar el viewer explícitamente (seguro en cluster)."""
        try:
            viewer = getattr(self, "_petsc_viewer", None)
            if viewer is None:
                viewer = PETSc.Viewer().createASCII(
                    self._log_txt, comm=PETSc.COMM_WORLD
                )
            PETSc.Log.view(viewer)
            try:
                viewer.flush()
            except Exception:
                pass
            try:
                viewer.destroy()
            except Exception:
                pass
            try:
                self._petsc_viewer = None
            except Exception:
                pass
        except Exception:
            pass

    # ====== BCs sobre W (idéntico concepto al tuyo; conserva reconstruct) ======
    def _define_node_time_boundary_setup(self):
        if not self.PDEs.boundary_conditions:
            return ([], {})
        bcs = []
        local_bcs = {}
        for bc in self.PDEs.boundary_conditions:
            if isinstance(bc, DirichletBC):
                bc_function_space = bc.function_space()
                idx, sub_idx = (
                    (
                        (
                            (
                                bc_function_space.index
                                if bc_function_space.index is not None
                                else 0
                            ),
                            None,
                        )
                    )
                    if bc_function_space.value_shape == ()
                    else (*bc._indices, None)[:2]
                )
                for m_node in range(self.M):
                    subspace = self.W.sub(idx + m_node * self.lenV)
                    subspace = subspace if sub_idx is None else subspace.sub(sub_idx)
                    local_bc = bc.reconstruct(V=subspace)
                    bcs.append(local_bc)
                    local_bcs.setdefault(idx + m_node * self.lenV, []).append(local_bc)
            elif isinstance(bc, EquationBC):
                pass
            else:
                raise Exception("your bc is not accepted.")
        return tuple(bcs), local_bcs

    # ====== bucle de barridos ======
    @PETSc.Log.EventDecorator("sweep_loop_execution")
    def _sweep_loop(self):
        self._timings_buffer.clear()
        for i, s in enumerate(self.sweep_solvers):
            t0 = time.perf_counter()
            SDCSolver._sweep(s)
            dt = time.perf_counter() - t0
            self._timings_buffer.append({"solver_index": i, "wall_time": dt})

    @staticmethod
    @PETSc.Log.EventDecorator("sweep_unique_execution")
    def _sweep(s):
        s.solve()

    # ====== métricas de análisis ======
    def _compute_analysis_metrics(
        self,
        real_u: Function | None,
        use_collocation: bool,
        use_exact: bool,
    ) -> dict[str, float | None]:
        total_residual_sweep = self._total_residual_sweep_aggregated()

        if use_collocation:
            vec_coll = assemble(self.R_coll).riesz_representation()
            total_residual_collocation = norm(vec_coll, norm_type="L2")
        else:
            total_residual_collocation = 0.0

        def _diff_fn(a: Function, b: Function) -> Function:
            out = Function(a.function_space())
            out.assign(a)
            out -= b
            return out

        def _l2_space(f: Function) -> float:
            return float((assemble(inner(f, f) * dx)) ** 0.5)

        def _h1_semi_of_fn(f: Function) -> float:
            return float((assemble(inner(grad(f), grad(f)) * dx)) ** 0.5)

        def _time_L2(err_nodes_flat: list[Function]) -> float:
            w = np.asarray(self.Q[-1, :], dtype=float)
            M = self.M
            P = self.lenV
            vals = np.empty(M, dtype=float)
            for m in range(M):
                block = err_nodes_flat[m * P : (m + 1) * P]
                s2 = 0.0
                for e in block:
                    s2 += _l2_space(e) ** 2
                vals[m] = s2**0.5
            return float(float(self.deltatC) * float(np.dot(w, vals**2))) ** 0.5

        if use_collocation:
            sweep_vs_coll_err = errornorm(
                self.u_collocation.subfunctions[-1],
                self.u_k_act.subfunctions[-1],
                norm_type="L2",
            )
            sweep_vs_coll_comp = sum(
                errornorm(u_c, u_k, norm_type="L2")
                for u_c, u_k in zip(
                    self.u_collocation.subfunctions, self.u_k_act.subfunctions
                )
            )
            e_nodes = [
                _diff_fn(u_c, u_k)
                for u_c, u_k in zip(
                    self.u_collocation.subfunctions, self.u_k_act.subfunctions
                )
            ]
            sweep_vs_coll_H1 = (
                sum(_h1_semi_of_fn(e) ** 2 for e in e_nodes[-self.lenV :])
            ) ** 0.5
            sweep_vs_coll_timeL2 = _time_L2(e_nodes)
        else:
            sweep_vs_coll_err = None
            sweep_vs_coll_comp = None
            sweep_vs_coll_H1 = None
            sweep_vs_coll_timeL2 = None

        if use_exact and real_u is not None:
            sweep_vs_real_err = errornorm(
                real_u.subfunctions[-1],
                self.u_k_act.subfunctions[-1],
                norm_type="L2",
            )
            sweep_vs_real_comp = sum(
                errornorm(r_u, u_k, norm_type="L2")
                for r_u, u_k in zip(real_u.subfunctions, self.u_k_act.subfunctions)
            )
            e_nodes_real = [
                _diff_fn(r, u_k)
                for r, u_k in zip(real_u.subfunctions, self.u_k_act.subfunctions)
            ]
            sweep_vs_real_H1 = (
                sum(_h1_semi_of_fn(e) ** 2 for e in e_nodes_real[-self.lenV :])
            ) ** 0.5
            sweep_vs_real_timeL2 = _time_L2(e_nodes_real)
        else:
            sweep_vs_real_err = None
            sweep_vs_real_comp = None
            sweep_vs_real_H1 = None
            sweep_vs_real_timeL2 = None

        if use_collocation and use_exact and real_u is not None:
            coll_vs_real_err = errornorm(
                self.u_collocation.subfunctions[-1],
                real_u.subfunctions[-1],
                norm_type="L2",
            )
            coll_vs_real_comp = sum(
                errornorm(u_c, r_u, norm_type="L2")
                for u_c, r_u in zip(
                    self.u_collocation.subfunctions, real_u.subfunctions
                )
            )
        else:
            coll_vs_real_err = None
            coll_vs_real_comp = None

        return {
            "total_residual_sweep": total_residual_sweep,
            "total_residual_collocation": total_residual_collocation,
            "sweep_vs_collocation_errornorm": sweep_vs_coll_err,
            "sweep_vs_collocation_compound_norm": sweep_vs_coll_comp,
            "sweep_vs_real_errornorm": sweep_vs_real_err,
            "sweep_vs_real_compound_norm": sweep_vs_real_comp,
            "collocation_vs_real_errornorm": coll_vs_real_err,
            "collocation_vs_real_compound_norm": coll_vs_real_comp,
            "sweep_vs_collocation_H1seminorm": sweep_vs_coll_H1,
            "sweep_vs_real_H1seminorm": sweep_vs_real_H1,
            "sweep_vs_collocation_timeL2": sweep_vs_coll_timeL2,
            "sweep_vs_real_timeL2": sweep_vs_real_timeL2,
        }

    # ====== solver de colocación (W) ======
    @PETSc.Log.EventDecorator("_setup_full_collocation_solver")
    def _setup_full_collocation_solver(self):
        deltat, tau, t0, f = self.deltatC, self.tau, self.t_0_subinterval, self.PDEs.f
        Q = self.Q
        w = TestFunction(self.W)
        u_c = self.u_collocation
        u0_c = self.u_0_collocation
        u_c_split = split(u_c)
        u0_c_split = split(u0_c)
        w_split = split(w)
        R_coll = 0
        for p, f_i in enumerate(f):
            for m in range(self.M):
                idx = p + m * self.lenV
                Rm = inner(u_c_split[idx] - u0_c_split[idx], w_split[idx])
                for j in range(self.M):
                    jdx = p + j * self.lenV
                    Rm -= (
                        deltat
                        * Q[m, j]
                        * f_i(t0 + tau[j] * deltat, u_c_split[jdx], w_split[idx])
                    )
                R_coll += Rm * dx
        self.R_coll = R_coll
        problem = NonlinearVariationalProblem(R_coll, u_c, bcs=self.bcs_W)
        self.collocation_solver = NonlinearVariationalSolver(
            problem,
            solver_parameters=self.solver_parameters
            or {
                "snes_type": "newtonls",
                "snes_rtol": 1e-8,
                "ksp_type": "gmres",
                "pc_type": "hypre",
            },
        )

    # ====== paralelo sobre V (sin W) ======
    @PETSc.Log.EventDecorator("_setup_parallel_sweep_solver_V")
    def _setup_parallel_sweep_solver_V(self):
        deltat = self.deltatC
        tau = self.tau
        t0 = self.t_0_subinterval
        f_list = self.PDEs.f
        Q = self.Q
        Q_D = self.Q_D

        self.sweep_solvers = []
        self.R_sweep = []
        self._sweep_meta.clear()

        for m in range(self.M):
            u_m = self.Uk_act[m]
            vV = TestFunction(self.V)

            # Nota: en tu caso de calor 1D (V escalar) esto coincide con la versión previa.
            u_split = split(u_m)
            v_split = split(vV)
            u0_split = split(self.U0[m])
            u_prev_splits = [split(self.Uk_prev[j]) for j in range(self.M)]

            Rm_int = 0
            t_m = t0 + tau[m] * deltat
            for p, f_i in enumerate(f_list):
                left_p = inner(u_split[p], v_split[p]) - deltat * self.scale * Q_D[
                    m, m
                ] * f_i(t_m, u_split[p], v_split[p])
                right_p = inner(u0_split[p], v_split[p])
                for j in range(self.M):
                    t_j = t0 + tau[j] * deltat
                    coeff = Q[m, j] - self.scale * Q_D[m, j]
                    right_p += (
                        deltat * coeff * f_i(t_j, u_prev_splits[j][p], v_split[p])
                    )
                Rm_int += left_p - right_p

            Rm = Rm_int * dx
            self.R_sweep.append(Rm)

            problem_m = NonlinearVariationalProblem(Rm, u_m, bcs=self.bcs_V)
            solver_m = NonlinearVariationalSolver(
                problem_m,
                solver_parameters=self.solver_parameters
                or {
                    "snes_type": "newtonls",
                    "snes_rtol": 1e-8,
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                },
            )
            self._sweep_meta.append(
                {"solver_index": len(self.sweep_solvers), "node": m}
            )
            self.sweep_solvers.append(solver_m)

    # ====== paralelo sobre W (si quieres usar W en paralelo) ======
    @PETSc.Log.EventDecorator("_setup_parallel_sweep_solver_W")
    def _setup_parallel_sweep_solver_W(self):
        deltat, tau, t0 = self.deltatC, self.tau, self.t_0_subinterval
        f_list, Q, Q_D = self.PDEs.f, self.Q, self.Q_D

        self.sweep_solvers = []
        self.R_sweep = []
        self._sweep_meta.clear()

        for p, f_i in enumerate(f_list):
            for m in range(self.M):
                idx = p + m * self.lenV
                u_m = self.u_k_act.subfunctions[idx]
                v_m = TestFunction(u_m.function_space())

                left = (
                    inner(u_m, v_m)
                    - deltat
                    * self.scale
                    * Q_D[m, m]
                    * f_i(t0 + tau[m] * deltat, u_m, v_m)
                ) * dx

                right = inner(self.u_0.subfunctions[idx], v_m)
                for j in range(self.M):
                    jdx = p + j * self.lenV
                    coeff = Q[m, j] - self.scale * Q_D[m, j]
                    right += (
                        deltat
                        * coeff
                        * f_i(
                            t0 + tau[j] * deltat, self.u_k_prev.subfunctions[jdx], v_m
                        )
                    )
                right *= dx
                R_sweep = left - right
                self.R_sweep.append(R_sweep)

                # BCs locales correctas (ya reconstruidas)
                bcs_local = tuple(self.bcs_V_2.get(idx, []))
                problem_m = NonlinearVariationalProblem(R_sweep, u_m, bcs=bcs_local)

                self._sweep_meta.append(
                    {
                        "solver_index": len(self.sweep_solvers),
                        "comp": p,
                        "node": m,
                        "flat_idx": idx,
                        "lenV": self.lenV,
                    }
                )
                self.sweep_solvers.append(
                    NonlinearVariationalSolver(
                        problem_m,
                        solver_parameters=(
                            {
                                "snes_type": "newtonls",
                                "snes_rtol": 1e-8,
                                "ksp_type": "cg",
                            }
                            if not self.solver_parameters
                            else self.solver_parameters
                        ),
                    )
                )

    # ====== global sobre W ======
    @PETSc.Log.EventDecorator("_setup_global_sweep_solver")
    def _setup_global_sweep_solver(self):
        deltat = self.deltatC
        tau = self.tau
        t0 = self.t_0_subinterval
        u_0 = self.u_0
        u_k_prev = self.u_k_prev
        u_k_act = self.u_k_act
        f = self.PDEs.f

        v = TestFunction(self.W)
        v_split = split(v)
        u_k_act_tup = split(u_k_act)
        R_sweep = 0

        Q = self.Q
        Q_D = self.Q_D

        self.sweep_solvers = []
        self._sweep_meta = [{"solver_index": 0, "global": True}]
        self.R_sweep = []

        for p, f_i in enumerate(f):
            for i_m in range(self.M):
                m = p + i_m * self.lenV
                v_m = v_split[m]
                u_m = u_k_act_tup[m]

                left = (
                    inner(u_m, v_m)
                    - deltat
                    * self.scale
                    * Q_D[i_m, i_m]
                    * f_i(t0 + tau[i_m] * deltat, u_m, v_m)
                ) * dx

                right = inner(u_0.subfunctions[m], v_m)
                for j in range(self.M):
                    jdx = p + j * self.lenV
                    coeff = Q[i_m, j] - self.scale * Q_D[i_m, j]
                    right += (
                        deltat
                        * coeff
                        * f_i(
                            t0 + tau[j] * deltat,
                            u_k_prev.subfunctions[jdx],
                            v_m,
                        )
                    )
                right *= dx
                R_sweep += left - right

        self.R_sweep = [R_sweep]

        problem_m = NonlinearVariationalProblem(R_sweep, u_k_act, bcs=self.bcs_W)
        self.sweep_solvers.append(
            NonlinearVariationalSolver(
                problem_m,
                solver_parameters=(
                    {
                        "snes_type": "newtonls",
                        "snes_rtol": 1e-8,
                        "ksp_type": "preonly",
                        "pc_type": "lu",
                    }
                    if not self.solver_parameters
                    else self.solver_parameters
                ),
            )
        )

    # ====== solve (con cierre de log garantizado) ======
    def solve(
        self,
        T,
        sweeps,
        real_solution_exp: Function | None = None,
        max_diadic: int = 10000,
    ):
        """
        Ejecuta time-stepping SDC. Funciona con layout paralelo "V" o "W",
        y global. Si analysis=True, se usa W para métricas/colocación.
        """
        try:

            def _update_exact_field_W(t_now: float):
                if real_solution_exp is None:
                    return
                for i, ru in enumerate(real_u.subfunctions):
                    i_node = i // self.lenV  # ← nodo temporal (0..M-1)
                    local_t = t_now + float(self.tau[i_node]) * float(
                        self.deltatC
                    )  # ← usa deltatC
                    self._t_exact.assign(local_t)
                    ru.interpolate(self._exact_expr)

            def _set_scale(k: int):
                if self.prectype == "MIN-SR-FLEX":
                    self.scale.assign(1.0 / k)
                else:
                    self.scale.assign(1.0)

            analysis = self.analysis
            use_exact = real_solution_exp is not None
            write_vtk = self.mode == "vtk"

            t, step = 0.0, 0
            err_intra = []

            if analysis or use_exact:
                assert self.uses_W, "Analysis or exact solution requires W."
                use_exact = real_solution_exp is not None
                if use_exact:
                    real_u = Function(self.W, name="u_exact")

                    X0 = self.PDEs.coord
                    # --- clave: tiempo como Constant reutilizable ---
                    self._t_exact = Constant(0.0)
                    self._exact_expr = real_solution_exp(X0, self._t_exact)

                    # Inicializa u_exact con t inicial sin compilar mil veces
                    for u in real_u.subfunctions:
                        u.interpolate(self._exact_expr)

            convergence_results = {
                "INFO": [
                    "Total residual collocation",
                    "Total residual sweep",
                    "Sweep vs collocation error norm",
                    "Sweep vs collocation compound norm",
                    "Sweep vs real error norm",
                    "Sweep vs real compound norm",
                    "Collocation vs real error norm",
                    "Collocation vs real compound norm",
                ]
            }

            # Estride diádico
            save_max = max_diadic
            N_steps_total = int(np.ceil(T / float(self.deltat)))
            base_stride = max(1, int(np.ceil((N_steps_total + 1) / float(save_max))))
            SAVE_STRIDE = 1 << (base_stride - 1).bit_length()

            save_idx = 0
            wrote_T = False

            def _maybe_save_checkpoint(afile, t_now: float):
                nonlocal save_idx, wrote_T
                tau_last = float(self.tau[-1]) if len(self.tau) > 0 else 1.0
                t_tag = float(t_now + float(self.deltatC) * tau_last)
                if abs(t_tag - T) <= 1e-12:
                    wrote_T = True

                last_fn = self._last_block_as_V(prev=False)
                u_save = Function(last_fn.function_space(), name="u")
                u_save.assign(last_fn)
                afile.save_function(
                    u_save, idx=save_idx, timestepping_info={"time": t_tag}
                )

                if use_exact:
                    real_last = real_u.subfunctions[-1]
                    real_save = Function(real_last.function_space(), name="u_exact")
                    real_save.assign(real_last)
                    afile.save_function(
                        real_save, idx=save_idx, timestepping_info={"time": t_tag}
                    )

                if analysis:
                    ucoll_last = self.u_collocation.subfunctions[-1]
                    ucoll_save = Function(ucoll_last.function_space(), name="u_coll")
                    ucoll_save.assign(ucoll_last)
                    afile.save_function(
                        ucoll_save, idx=save_idx, timestepping_info={"time": t_tag}
                    )

                save_idx += 1

            def _maybe_save_vtk(vtk, vtk_coll, vtk_exact, t_now: float):
                nonlocal save_idx, wrote_T
                tau_last = float(self.tau[-1]) if len(self.tau) > 0 else 1.0
                t_tag = float(t_now + float(self.deltatC) * tau_last)  # ← deltatC
                vtk.write(self._get_last_state_view(), time=t_tag)
                if analysis and vtk_coll is not None:
                    vtk_coll.write(self.u_collocation.subfunctions[-1], time=t_tag)
                if use_exact and vtk_exact is not None:
                    vtk_exact.write(real_u.subfunctions[-1], time=t_tag)
                if abs(t_tag - T) <= 1e-12:  # ← compara t_tag
                    wrote_T = True
                save_idx += 1

            if not write_vtk:
                with CheckpointFile(self.file, "a") as afile:
                    afile.save_mesh(self.mesh)

                    while t < T:
                        dt_eff = min(self.deltat, T - t)
                        self.deltatC.assign(dt_eff)
                        self.t_0_subinterval.assign(t)
                        if self.PDEs.time_dependent_constants_bts:
                            for ct in self.PDEs.time_dependent_constants_bts:
                                ct.assign(t)
                        delta_prev = None
                        rho_seq = []
                        delta_seq = []

                        if analysis:
                            if self.is_parallel:
                                self._sync_W_from_V()
                            last = self.u_k_act.subfunctions[-1]
                            for u in self.u_0_collocation.subfunctions:
                                u.assign(last)
                            t0_wall = time.perf_counter()
                            self.collocation_solver.solve()
                            if use_exact:
                                _update_exact_field_W(t)
                            collocation_wall_time = time.perf_counter() - t0_wall
                            convergence_results[
                                f"{step},{t},full_collocation_timing"
                            ] = [
                                {
                                    "solver_index": "full_collocation",
                                    "wall_time": collocation_wall_time,
                                }
                            ]
                            analysis_metrics_0 = self._compute_analysis_metrics(
                                real_u if use_exact else None, True, use_exact
                            )
                            convergence_results[f"{step},{t},0"] = [
                                analysis_metrics_0["total_residual_collocation"],
                                analysis_metrics_0["total_residual_sweep"],
                                analysis_metrics_0["sweep_vs_collocation_errornorm"],
                                analysis_metrics_0[
                                    "sweep_vs_collocation_compound_norm"
                                ],
                                analysis_metrics_0["sweep_vs_real_errornorm"],
                                analysis_metrics_0["sweep_vs_real_compound_norm"],
                                analysis_metrics_0["collocation_vs_real_errornorm"],
                                analysis_metrics_0["collocation_vs_real_compound_norm"],
                            ]
                        for k in range(1, sweeps + 1):
                            _set_scale(k)
                            if self.is_parallel:
                                for m in range(self.M):
                                    self.Uk_prev[m].assign(self.Uk_act[m])
                            else:
                                self.u_k_prev.assign(self.u_k_act)

                            if analysis:
                                self._sweep_loop()
                            else:
                                for s in self.sweep_solvers:
                                    s.solve()

                            if analysis:
                                if self.is_parallel:
                                    self._sync_W_from_V()

                                try:
                                    curr = self._last_block_as_V(prev=False)
                                    prevv = self._last_block_as_V(prev=True)
                                    du = Function(self.V)
                                    du.assign(curr)
                                    du -= prevv
                                    delta = float(norm(du, norm_type="L2"))
                                    delta_seq.append(delta)
                                    eps = 1e-14
                                    if delta_prev is not None and delta_prev > eps:
                                        rho_seq.append(float(delta / delta_prev))
                                    delta_prev = delta
                                except Exception:
                                    pass

                                if use_exact:
                                    _update_exact_field_W(t)

                                analysis_metrics = self._compute_analysis_metrics(
                                    real_u if use_exact else None, analysis, use_exact
                                )
                                convergence_results[f"{step},{t},{k}"] = [
                                    analysis_metrics["total_residual_collocation"],
                                    analysis_metrics["total_residual_sweep"],
                                    analysis_metrics["sweep_vs_collocation_errornorm"],
                                    analysis_metrics[
                                        "sweep_vs_collocation_compound_norm"
                                    ],
                                    analysis_metrics["sweep_vs_real_errornorm"],
                                    analysis_metrics["sweep_vs_real_compound_norm"],
                                    analysis_metrics["collocation_vs_real_errornorm"],
                                    analysis_metrics[
                                        "collocation_vs_real_compound_norm"
                                    ],
                                ]
                                convergence_results[f"{step},{t},contraction"] = {
                                    "delta_last": (
                                        float(delta_prev)
                                        if delta_prev is not None
                                        else None
                                    ),
                                    "rho_seq": rho_seq[:],
                                    "delta_seq": delta_seq[:],
                                }

                                timings = []
                                for row in self._timings_buffer:
                                    meta = next(
                                        (
                                            m
                                            for m in self._sweep_meta
                                            if m.get("solver_index")
                                            == row["solver_index"]
                                        ),
                                        {},
                                    )
                                    timings.append({**meta, **row})
                                convergence_results[f"{step},{t},{k}_timings"] = timings

                                err_intra.append(
                                    analysis_metrics[
                                        "sweep_vs_collocation_compound_norm"
                                    ]
                                    if analysis
                                    else None
                                )

                                print(
                                    f"step {step}  t={t:.4e}  "
                                    f"res_sweep={analysis_metrics['total_residual_sweep']:.3e}  "
                                    f"err_coll={analysis_metrics['sweep_vs_collocation_errornorm']}"
                                )

                        print("\n\n\n")

                        if (step % SAVE_STRIDE) == 0:
                            _maybe_save_checkpoint(afile, t)

                        self._sync_all_nodes_to_last()

                        # t += self.deltat
                        t += float(self.deltatC)
                        # self.t_0_subinterval.assign(t)

                        step += 1

                    if not wrote_T:
                        tau_last = float(self.tau[-1]) if len(self.tau) > 0 else 1.0
                        _maybe_save_checkpoint(
                            afile, T - float(self.deltatC) * tau_last
                        )

                    convergence_results_path = (
                        Path(self.file).with_suffix("").as_posix()
                        + "_convergence_results.json"
                    )
                    with open(str(convergence_results_path), "w") as f:
                        json.dump(convergence_results, f, indent=2)
                    return step - 1
            else:
                vtk = VTKFile(self.file)
                vtk_coll = (
                    VTKFile(Path(self.file).with_suffix("").as_posix() + "_ucoll.pvd")
                    if analysis
                    else None
                )
                vtk_exact = (
                    VTKFile(Path(self.file).with_suffix("").as_posix() + "_uexact.pvd")
                    if use_exact
                    else None
                )

                while t < T:
                    print(f"step {step}  t={t:.4e}")
                    dt_eff = min(self.deltat, T - t)
                    self.deltatC.assign(dt_eff)
                    self.t_0_subinterval.assign(t)
                    if self.PDEs.time_dependent_constants_bts:
                        for ct in self.PDEs.time_dependent_constants_bts:
                            ct.assign(t)

                    if analysis:
                        if self.is_parallel:
                            self._sync_W_from_V()
                        last = self.u_k_act.subfunctions[-1]
                        for u in self.u_0_collocation.subfunctions:
                            u.assign(last)
                        self.collocation_solver.solve()

                    if use_exact:
                        _update_exact_field_W(t)

                    for k in range(1, sweeps + 1):
                        _set_scale(k)
                        if self.is_parallel:
                            for m in range(self.M):
                                self.Uk_prev[m].assign(self.Uk_act[m])
                        else:
                            self.u_k_prev.assign(self.u_k_act)

                        if analysis:
                            self._sweep_loop()
                            if self.is_parallel:
                                self._sync_W_from_V()
                            analysis_metrics = self._compute_analysis_metrics(
                                real_u if use_exact else None, analysis, use_exact
                            )
                            convergence_results[f"{step},{t},{k}"] = [
                                analysis_metrics["total_residual_collocation"],
                                analysis_metrics["total_residual_sweep"],
                                analysis_metrics["sweep_vs_collocation_errornorm"],
                                analysis_metrics["sweep_vs_collocation_compound_norm"],
                                analysis_metrics["sweep_vs_real_errornorm"],
                                analysis_metrics["sweep_vs_real_compound_norm"],
                                analysis_metrics["collocation_vs_real_errornorm"],
                                analysis_metrics["collocation_vs_real_compound_norm"],
                            ]

                            timings = []
                            for row in self._timings_buffer:
                                meta = next(
                                    (
                                        m
                                        for m in self._sweep_meta
                                        if m.get("solver_index") == row["solver_index"]
                                    ),
                                    {},
                                )
                                timings.append({**meta, **row})
                            convergence_results[f"{step},{t},{k}_timings"] = timings

                            print(
                                f"step {step}  t={t:.4e}  "
                                f"res_sweep={analysis_metrics['total_residual_sweep']:.3e}  "
                                f"err_coll={analysis_metrics['sweep_vs_collocation_errornorm']}"
                            )
                        else:
                            for s in self.sweep_solvers:
                                s.solve()

                    if (step % SAVE_STRIDE) == 0:
                        _maybe_save_vtk(vtk, vtk_coll, vtk_exact, t)

                    self._sync_all_nodes_to_last()

                    # t += self.deltat
                    t += float(self.deltatC)
                    # self.t_0_subinterval.assign(t)
                    if self.PDEs.time_dependent_constants_bts:
                        for ct in self.PDEs.time_dependent_constants_bts:
                            ct.assign(t)
                    step += 1

                if not wrote_T:
                    tau_last = float(self.tau[-1]) if len(self.tau) > 0 else 1.0
                    _maybe_save_vtk(
                        vtk, vtk_coll, vtk_exact, T - float(self.deltatC) * tau_last
                    )

                convergence_results_path = (
                    Path(self.file).with_suffix("").as_posix()
                    + "_convergence_results.json"
                )
                with open(str(convergence_results_path), "w") as f:
                    json.dump(convergence_results, f, indent=2)
                return step - 1
        finally:
            # CIERRE IMPRESCINDIBLE DEL LOG (también si hay excepción)
            self._write_and_close_log()


# ================= Helpers y launcher (después de SDCSolver) =================
import os
import platform
from datetime import datetime
from pathlib import Path
from itertools import product

# Fecha para etiquetar si lo necesitas en file_name
now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


def _is_cluster_env() -> bool:
    """Heurística robusta para distinguir cluster vs Mac."""
    if os.environ.get("PBS_JOBID"):
        return True
    if Path("/home/clustor2").exists():
        return True
    return platform.system() != "Darwin"


def _resolve_base_output_dir(path_name: str | None) -> Path:
    """
    Prioridad:
      1) path_name explícito (argumento)
      2) SDC_OUTPUT_DIR (env)
      3) clustor2 por defecto (/home/clustor2/ma/<u>/<user>/solver_results/heatfiles) si existe
      4) Mac: ~/solver_results/heatfiles
      5) Genérico: ~/sdc_runs
    """
    if path_name:
        return Path(path_name)

    env_out = os.environ.get("SDC_OUTPUT_DIR")
    if env_out:
        return Path(env_out)

    # Preferir clustor2 si existe
    user = os.environ.get("USER") or os.path.basename(str(Path.home()))
    if user and len(user) > 0:
        cl2_base = Path(f"/home/clustor2/ma/{user[0]}/{user}")
        if cl2_base.exists():
            return cl2_base / "solver_results" / "heatfiles"

    # macOS: ~/solver_results/heatfiles
    if platform.system() == "Darwin":
        return Path.home() / "solver_results" / "heatfiles"

    # Fallback genérico
    return Path.home() / "sdc_runs"


def _next_run_subfolder(base: Path, group: str) -> str:
    """
    Devuelve '<group>/run_XXXX' con índice creciente.
    Crea el directorio de grupo si no existe.
    """
    run_root = base / group
    run_root.mkdir(parents=True, exist_ok=True)
    existing = [
        p
        for p in run_root.iterdir()
        if p.is_dir() and p.name.startswith("run_") and p.name[4:].isdigit()
    ]
    last = max((int(p.name[4:]) for p in existing), default=0)
    return str(Path(group) / f"run_{last+1:04d}")


# ===================== Ejemplo: problema del calor 1D (HOMOGÉNEO f=0) =====================
import math


def heat_rhs(t, u, v):
    # u_t = Δu  ⇒  <u_t, v> = -<∇u, ∇v>    (Dirichlet homogénea)
    return -inner(grad(u), grad(v))


def run_order3_test(
    N=256,
    deg=4,  # malla y grado espacial (espacial << temporal)
    M=2,
    prectype="MIN-SR-FLEX",  # Radau-right (lo fija el test más abajo)
    tfinal=0.1,  # T pequeño → menos pasos y mismo orden
    dts=(1e-2, 5e-3, 2.5e-3, 1.25e-3),  # 10, 20, 40, 80 pasos
    verbose=True,
):
    mesh = UnitIntervalMesh(N)
    V = FunctionSpace(mesh, "CG", deg)
    (x,) = SpatialCoordinate(mesh)

    # Solución exacta y u0
    def u_exact_expr(t):
        return sin(math.pi * x) * math.exp(-(math.pi**2) * t)

    u0 = Function(V, name="u0")
    u0.interpolate(u_exact_expr(0.0))

    # BCs homogéneas en todo el borde
    bc = DirichletBC(V, 0.0, "on_boundary")

    # PDESystem minimal
    P = PDESystem(
        mesh=mesh,
        V=V,
        coord=SpatialCoordinate(mesh),
        f=heat_rhs,  # tu SDCSolver espera f(t,u,v)
        u0=u0,
        boundary_conditions=(bc,),
        time_dependent_constants_bts=None,
        name="heat_1d",
    )

    errs = []
    for dt in dts:
        if verbose:
            print(f"\n[Δt={dt:g}] montando solver (M=2 Radau-right)...", flush=True)

        # Radau-right: en tu SDCPreconditioners, usa tau_type="radau-right"
        solver = SDCSolver(
            mesh=mesh,
            PDEs=P,
            M=M,
            dt=dt,
            is_parallel=True,  # da igual; vamos a usar solo colocación
            solver_parameters={
                "snes_type": "newtonls",
                "snes_rtol": 1e-12,
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
            prectype=prectype,
            tau=None,  # que lo genere con radau-right por defecto
            analysis=True,  # <— IMPORTANTE: crea el solver de colocación
            file_name="noop",
            folder_name="noop",
            mode="checkpoint",
        )

        # (re)inicializa los estados de colocación desde u0
        # (analysis=True ya lo hace en __init__, lo repetimos por claridad)
        if hasattr(solver, "_init_collocation_from_u0"):
            solver._init_collocation_from_u0()

        t = 0.0
        nsteps = int(round(tfinal / dt))
        t0_const = solver.t_0_subinterval
        Mloc = solver.M
        lenV = solver.lenV

        tic = time.perf_counter()
        for n in range(nsteps):
            t0_const.assign(t)  # paso local
            # resuelve el sistema de colocación del subintervalo
            solver.collocation_solver.solve()

            # prepara el siguiente subintervalo: copiar el último nodo → u0 de todos los nodos
            for p in range(lenV):
                last = solver.u_collocation.subfunctions[p + (Mloc - 1) * lenV]
                for m in range(Mloc):
                    solver.u_0_collocation.subfunctions[p + m * lenV].assign(last)

            t += dt
        toc = time.perf_counter()

        # error final (L2) vs exacta en t=tfinal (último nodo de la malla temporal)
        u_end = solver.u_collocation.subfunctions[(Mloc - 1) * lenV]  # lenV=1 aquí
        u_ex = Function(V)
        u_ex.interpolate(u_exact_expr(tfinal))
        err = errornorm(u_ex, u_end, norm_type="L2")

        errs.append(float(err))
        if verbose:
            print(
                f"  pasos: {nsteps:4d}   tiempo: {toc - tic:.2f}s   ||e||_L2 = {err:.3e}"
            )

    # Ajuste en log–log: e ≈ C * Δt^p
    x = np.array(dts, dtype=float)
    y = np.array(errs, dtype=float)
    m, b = np.polyfit(np.log(x), np.log(y), 1)
    p = float(m)
    if verbose:
        print("\n==== Resultado ====")
        print("dts:", "  ".join(f"{v:g}" for v in x))
        print("err:", "  ".join(f"{v:.3e}" for v in y))
        print(f"orden temporal p ≈ {p:.3f}  (esperado ≈ 3.0 para Radau-right M=2)")

    return p, list(zip(dts, errs))


if __name__ == "__main__":
    p, pairs = run_order3_test()
    # Test “blando”: no fallar duro; solo avisa si se sale de banda
    if 2.7 <= p <= 3.3:
        print("✅ OK: orden ~3")
    else:
        print(
            "⚠️ aviso: p≈{:.3f} (fuera de [2.7, 3.3]) — revisa que el error espacial no sature (sube N o deg) o usa dts más grandes.".format(
                p
            )
        )
