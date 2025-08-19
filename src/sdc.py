from typing import Iterable, Literal, List
from pathlib import Path
import json
import numpy as np
import time
from firedrake import *
from firedrake.output import VTKFile
from .preconditioners import SDCPreconditioners
from .filenamer import FileNamer
from .specs import PDESystem


class SDCSolver(FileNamer, SDCPreconditioners):
    """
    Specific solver for SDC
    """

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
        """
        Mesh : Predermined mesh
        f: python function class object where the expression of the
            f(t, x, u(t, x)) part of the heat equation is written in
            UFL. It needs to be already written in weak form and
            after integration by parts.
        V: Initial basis space
        bcs: python function calls object where its
        prectype : MIN-SR-FLEX, MIN-SR-S, DIAG1, ...,
        tau: personalised nodes

        -----
        All the iterables are ment to solve systems of equations.

        From what I heard we can combine and nest different function spaces
        over the same mesh and they will be flattened. Also we can mix
        MixedFunctionSpace with VectorFunctionSpace

        """
        # Initialise FileNamer
        FileNamer.__init__(
            self,
            file_name=file_name,
            folder_name=folder_name,
            path_name=path_name,
            mode=mode,
        )
        # Initialise preconditioner infrastructure
        SDCPreconditioners.__init__(
            self,
            M=M,
            prectype=prectype,
            tau=tau,
        )

        self.mesh = mesh
        self.PDEs = PDEs
        self.deltat = dt
        self.is_parallel = is_parallel
        self.solver_parameters = solver_parameters
        self.analysis = analysis

        # For time measuring, we define two list of dictionaries
        self._sweep_meta: list[dict] = []
        self._timings_buffer: list[dict] = []

        # Dealing with the whole system of pdes
        # Create the mixed Function space of all of them
        self.V = self.PDEs.V
        self.lenV = len(self.V.subspaces) if self.PDEs._is_Mixed else 1

        # In order to match spatial and temporal discretisation,
        # we create a MixedFunctionSpace in order to have a bag
        # of individual function space objects, so when we create
        # a function in this space we are creating M functions, one
        # for each node M defined
        self.W = MixedFunctionSpace([self.V] * self.M)
        # Also we could use the * operator to create the MixedFunctionSpace,
        # self.W is a flat list of subspaces, theres no nested subspaces
        # except for the vectorfunctionspaces. self.V is flattened before
        # being instantiated in PDESystem.

        # Instantiate boundary conditions and test functions:
        self.bcs_V = self.PDEs.boundary_conditions
        self.bcs_W, self.bcs_V_2 = self._define_node_time_boundary_setup()

        # Define the residuals
        self.R_sweep = []
        self.R_coll = 0

        # Define the actual functions, if we want to retrieve
        # the list of functions for each coordinate use split.
        self.u_0 = Function(self.W, name="u_0")
        self.u_k_prev = Function(self.W, name="u_k")
        self.u_k_act = Function(self.W, name="u_{k+1}")

        # In order to compare the collocation problem with the solutions,
        # we need to create
        self.u_0_collocation = Function(self.W, name="u_0 collocation")
        self.u_collocation = Function(self.W, name="u_collocation")

        # Debemos pensar que lo que se define en el espacio finito es la base
        # del espacio finito ghlobal, nada más, justamente lo que hace Function
        # es definir las coordenadas de la función con respecto a esa base (y
        # bueno más cosas). Para más información mirar mi librería de finite
        # elements.

        # As all the functions are vectorial in the codomain due
        # to the nodal discretisation of the temporal axis
        for i, (
            subfunction_0,
            subfunction_k_prev,
            subfunction_k_act,
            subfunction_0_collocation,
            subfunction_collocation,
        ) in enumerate(
            zip(
                self.u_0.subfunctions,
                self.u_k_prev.subfunctions,
                self.u_k_act.subfunctions,
                self.u_0_collocation.subfunctions,
                self.u_collocation.subfunctions,
            ),
            0,
        ):
            u0 = self.PDEs.u0.subfunctions[i % self.lenV]
            subfunction_0.interpolate(u0)
            subfunction_k_prev.interpolate(u0)
            subfunction_k_act.interpolate(u0)
            subfunction_0_collocation.interpolate(u0)
            subfunction_collocation.interpolate(u0)

        # Initial time and instantiate the solvers
        self.t_0_subinterval = Constant(0.0)
        self.scale = Constant(1.0)
        (
            self._setup_paralell_sweep_solver()
            if is_parallel
            else self._setup_global_sweep_solver()
        )

        self._setup_full_collocation_solver() if self.analysis else None

    def _define_node_time_boundary_setup(self):
        """
        Need to be very awayre of how Firedrake flattens
        MixedFunctionSpace, we have a lot of tests in
        boundary_conditions.py. Also have a look to my notes
        in Notability.

        1. Firedrake flattens any nested MixedFunctionSpaces, so
            at the end what we always have with the subspaces atribute
            is that gives us an iterator with the flattened result
            (FunctionSpace, VectorFunctionSpace, FunctionSpace, FunctionSpace...)
            but inside this iterable we wont find MixedFunctionSpaces.
            The value_size attr will give the whole dimension of the MixedFS.
            For instance if the previous tupple has dimensions (1, 4, 1, 1)
            respectively value_size will be 7
        2. Now, to deal with the VectorFunctionSpaces (they are like MixedFunctionSpaces
            but in contiguous memory space) is a little different, as
            VectorFunctions will have always len() = 1 subspaces iterable.
            Also it does not accept a nested VFS over other VFS, it must be
            seen as an individual FunctionSpace.
        3. Based on the structure of our code, there's no possible BC defined
            over the whole MFS, as the later is created within the PDESystem
            class, so the user is not allowed to define any bc over self.W,
            and thus we can ignore this case in the code.
        4. When a MFS is created, the subspaces that conforms it are copied,
            so the boundary conditions need to be also reassign to this new
            copies.

        +. No matter if in _setup_paralell_sweep_solver we are working with
            self.V test,

        """

        if not self.PDEs.boundary_conditions:
            return ([], {})
        # if self.is_parallel:
        #     return list(self.PDEs.boundary_conditions)

        bcs = []
        local_bcs = {}
        for bc in self.PDEs.boundary_conditions:
            if isinstance(bc, DirichletBC):
                # First, we need to retrieve the index of the subspace
                # for which the boundary condition acts
                # and, if it is a VFS, the sub_idx. We use the
                # internal attribute _indices in order to retrieve both of
                # them in the second case. To characterise the VFS we know
                # that it's value_shape attribute will have a.l 1 parameter.
                ## CAN WE HAVE A 1D VECTORFUNCTIONSPACE? HOW CAN I CHARACTERISE
                # A VECTORFUNCTIONSPACE
                bc_function_space = (
                    bc.function_space()
                )  # it gives us the subspace in which vs is defined.
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
                    # As Firedrake flattens the MixedFunctionSpace,
                    # we cannot have another MixedFunctionSpace nested!
                    local_bc = bc.reconstruct(V=(subspace))
                    # local_bc = bc
                    bcs.append(local_bc)
                    local_bcs.setdefault(idx + m_node * self.lenV, []).append(local_bc)

            elif isinstance(bc, EquationBC):
                pass
            else:
                raise Exception("your bc is not accepted.")

        return tuple(bcs), local_bcs

    @PETSc.Log.EventDecorator("sweep_loop_execution")
    def _sweep_loop(self):
        self._timings_buffer.clear()
        for i, s in enumerate(self.sweep_solvers):
            t0 = time.perf_counter()
            SDCSolver._sweep(s)
            dt = time.perf_counter() - t0
            self._timings_buffer.append(
                {
                    "solver_index": i,
                    "wall_time": dt,
                }
            )

    @staticmethod
    @PETSc.Log.EventDecorator("sweep_unique_execution")
    def _sweep(s):
        s.solve()

    def _compute_analysis_metrics(
        self,
        real_u: Function | None,
        use_collocation: bool,
        use_exact: bool,
    ) -> dict[str, float | None]:
        """ """
        residual_sweep_vecs = (
            assemble(Rm).riesz_representation() for Rm in self.R_sweep
        )
        total_residual_sweep = sum(norm(r) for r in residual_sweep_vecs)

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

        def _time_L2(err_nodes: list[Function]) -> float:
            w = np.asarray(self.Q[-1, :], dtype=float)  # pesos b_j (fila última de Q)
            vals = np.array([_l2_space(e) for e in err_nodes], dtype=float)
            return float(self.deltat * float(np.dot(w, vals**2))) ** 0.5

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
            sweep_vs_coll_H1 = _h1_semi_of_fn(e_nodes[-1])
            sweep_vs_coll_timeL2 = _time_L2(e_nodes)
        else:
            sweep_vs_coll_err = None
            sweep_vs_coll_comp = None
            sweep_vs_coll_H1 = None
            sweep_vs_coll_timeL2 = None

        # Errores vs solución exacta
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
            sweep_vs_real_H1 = _h1_semi_of_fn(e_nodes_real[-1])
            sweep_vs_real_timeL2 = _time_L2(e_nodes_real)
        else:
            sweep_vs_real_err = None
            sweep_vs_real_comp = None
            sweep_vs_real_H1 = None
            sweep_vs_real_timeL2 = None

        # Collocation vs real (como antes)
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

    @PETSc.Log.EventDecorator("_setup_full_collocation_solver")
    def _setup_full_collocation_solver(self):
        deltat, tau, t0, f = self.deltat, self.tau, self.t_0_subinterval, self.PDEs.f
        Q = self.Q
        w = TestFunction(self.W)
        # We could use the mixed space but it's nonsense, as we don't have coupling
        # among the different finite element subspaces.
        # We store the solvers
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
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )

    @PETSc.Log.EventDecorator("_setup_parallel_sweep_solver_V")
    def _setup_parallel_sweep_solver_V(self):
        """
        Un solver por nodo m, pero cada solver vive en el espacio mixto completo V.
        Pasamos self.bcs_V sin tocarlas.
        """
        deltat = self.deltat
        tau = self.tau
        t0 = self.t_0_subinterval
        f_list = self.PDEs.f
        Q = self.Q
        Q_D = self.Q_D

        self.sweep_solvers = []
        self.R_sweep = []
        self._sweep_meta.clear()

        for m in range(self.M):
            u_m = self.Uk_act[m]  # incógnita en V
            vV = TestFunction(self.V)  # test en V

            u_split = split(u_m)
            v_split = split(vV)
            u0_split = split(self.U0[m])

            Rm_int = 0
            t_m = t0 + tau[m] * deltat
            u_prev_splits = [split(self.Uk_prev[j]) for j in range(self.M)]
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

            # ¡BCs originales sobre V.sub(i)! — sin reconstrucciones locales
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

    @PETSc.Log.EventDecorator("_setup_paralell_sweep_solver")
    def _setup_paralell_sweep_solver(self):
        """
        Compute the solvers for the parallel case.

        + IMPORTANT: We create the test function over V because each
        subfunction of W has as function_space() attribute which
        points to the subspace of W where the basis of the function
        is defined. As there's no coupled systems, we just need to
        work with the basis of that finite_element space, ignoring
        the rest of W.

        """
        deltat = self.deltat
        tau = self.tau
        t0 = self.t_0_subinterval
        f = self.PDEs.f
        Q = self.Q
        Q_D = self.Q_D
        # We could use the mixed space but it's nonsense, as we don't have coupling
        # among the different finite element subspaces.
        # We store the solvers
        self.sweep_solvers = []
        self.R_sweep = []

        # We now instantiate also de residual of the original collocation problem
        # in order to study the convergence of the sweeps.
        for p, f_i in enumerate(f):
            for m in range(self.M):
                idx = p + m * self.lenV
                u_m = self.u_k_act.subfunctions[idx]
                v_m = TestFunction(u_m.function_space())
                #  assemble the part with u^{k+1}. We have to be very carefull as
                # v_m will be included in the function f.
                left = (
                    inner(u_m, v_m)
                    - deltat
                    * self.scale
                    * Q_D[m, m]
                    * f_i(t0 + tau[m] * deltat, u_m, v_m)
                ) * dx  # f need to be composed with the change of variables

                # assemble part with u^{k}
                right = inner(self.u_0.subfunctions[idx], v_m)
                for j in range(self.M):
                    jdx = p + j * self.lenV
                    coeff = Q[m, j] - self.scale * Q_D[m, j]
                    f_value = f_i(
                        t0 + tau[j] * deltat,
                        self.u_k_prev.subfunctions[jdx],
                        v_m,
                    )

                    right += deltat * coeff * f_value

                right = right * dx

                # Define the functional for that specific node
                R_sweep = left - right

                self.R_sweep.append(R_sweep)

                # Rebuild BCs on the exact trial space of this node/component
                u_space = u_m.function_space()
                # bcs_local = tuple(
                #     bc.reconstruct(V=u_space) for bc in self.bcs_V_2.get(idx, [])
                # )
                bcs_local = self.bcs_V

                # Colin asked me to use Nonlinear instead of Solve, is there any specific reason?
                problem_m = NonlinearVariationalProblem(R_sweep, u_m, bcs=bcs_local)

                # Add some parameters for analysis.
                self._sweep_meta.append(
                    {
                        "solver_index": len(self.sweep_solvers),
                        "comp": p,
                        "node": m,
                        "flat_idx": idx,
                        "lenV": self.lenV,
                    }
                )

                # Append the solver
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

    def _setup_upper_triangular_sweep_solver(self):
        pass

    def _setup_lower_triangular_sweep_solver(self):
        pass

    @PETSc.Log.EventDecorator("_setup_global_sweep_solver")
    def _setup_global_sweep_solver(self):
        """
        Here we use the accumulated residual over W. This is
        a global solver, meaning that can accept any kind of
        preconditioner. But please be aware that it wont calculate
        the optimal
        """

        deltat = self.deltat
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
        # Registrar meta único para el solver global
        self._sweep_meta = [{"solver_index": 0, "global": True}]
        self.R_sweep = []

        for p, f_i in enumerate(f):
            for i_m in range(self.M):
                m = p + i_m * self.lenV
                # As in my notes, each test function is independemt from the rest
                v_m = v_split[m]
                # retrieve m-coordinate of the vector function
                u_m = u_k_act_tup[m]
                #  assemble the part with u^{k+1}. We have to be very carefull as
                # v_m will be included in the function f.
                left = (
                    inner(u_m, v_m)
                    - deltat
                    * self.scale
                    * Q_D[i_m, i_m]
                    * f_i(t0 + tau[i_m] * deltat, u_m, v_m)
                ) * dx  # f need to be composed with the change of variables

                # assemble part with u^{k}
                right = inner(u_0.subfunctions[m], v_m)
                for j in range(self.M):
                    # we need to select the correct functions.
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
                right = right * dx

                # Add to general residual functional
                R_sweep += left - right

        self.R_sweep = [R_sweep]

        # Colin asked me to use Nonlinear instead of Solve, is there any specific reason?
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

    def solve(
        self,
        T,
        sweeps,
        real_solution_exp: Function | None = None,
        max_diadic: int = 10000,
    ):
        """
        real_solution_exp = ufl expression to be projected over the W space dependent
        on space and time if needed. X will be inputed considering that they
        are going to be the coordinates OBJECT over the mesh.

        """

        def _update_exact_field(t_now: float):
            """
            interpolates the exact solution over all of the subtime intervals
            tau defined in the collocation problem.
            """
            if real_solution_exp is None:
                return
            x = self.PDEs.coord
            for m, ru in enumerate(real_u.subfunctions):
                local_t = t_now + self.tau[m] * self.deltat
                ru.interpolate(real_solution_exp(x, local_t))

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

        if use_exact:
            real_u = Function(self.W, name="u_exact")
            X0 = self.PDEs.coord
            for u in real_u.subfunctions:
                u.interpolate(real_solution_exp(X0, t))

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
                "T_seq",
                "T_max",
            ]
        }

        # Choose a stride as a power of 2 so that output times align between runs
        # with dyadic time steps. Ensures consistent output times for comparisons.
        save_max = max_diadic
        N_steps_total = int(np.ceil(T / float(self.deltat)))  # nº total de pasos
        base_stride = max(1, int(np.ceil((N_steps_total + 1) / float(save_max))))
        # potencia de 2 mínima >= base_stride
        SAVE_STRIDE = 1 << (base_stride - 1).bit_length()

        # Compact index for saved states (separate from time step index)
        # and flag to mark if the final T snapshot has been saved
        save_idx = 0
        wrote_T = False

        def _maybe_save_checkpoint(afile, t_now: float):
            """
            Save solution (and exact/collocation fields if applicable)
            at the current time 't_now' to a checkpoint file.
            - Uses the compact save index 'save_idx'
            - Marks if the final time T is saved
            """
            nonlocal save_idx, wrote_T

            # Determine physical time of the last collocation node; usually tau[-1] = 1.0
            tau_last = (
                float(self.tau[-1])
                if hasattr(self, "tau") and len(self.tau) > 0
                else 1.0
            )
            t_tag = float(t_now + self.deltat * tau_last)

            # Save a fresh Function to avoid renaming side-effects on live fields
            u_last_node = self.u_k_act.subfunctions[-1]
            u_save = Function(u_last_node.function_space(), name="u")
            u_save.assign(u_last_node)
            afile.save_function(u_save, idx=save_idx, timestepping_info={"time": t_tag})

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

            if abs(t_now - T) <= 1e-12:
                wrote_T = True
            save_idx += 1

        def _maybe_save_vtk(vtk, vtk_coll, vtk_exact, t_now: float):
            """
            Save solution (and exact/collocation fields if applicable)
            at the current time 't_now' to VTK files.
            - Uses the compact save index 'save_idx'
            - Marks if the final time T is saved
            """
            nonlocal save_idx, wrote_T

            # Determine physical time of the last collocation node for VTK time tag
            tau_last = (
                float(self.tau[-1])
                if hasattr(self, "tau") and len(self.tau) > 0
                else 1.0
            )
            t_tag = float(t_now + self.deltat * tau_last)

            vtk.write(self.u_k_act.subfunctions[-1], time=t_tag)
            if analysis and vtk_coll is not None:
                vtk_coll.write(self.u_collocation.subfunctions[-1], time=t_tag)
            if use_exact and vtk_exact is not None:
                vtk_exact.write(real_u.subfunctions[-1], time=t_tag)

            if abs(t_now - T) <= 1e-12:
                wrote_T = True
            save_idx += 1

        if not write_vtk:
            with CheckpointFile(self.file, "a") as afile:
                # Save the mesh
                afile.save_mesh(self.mesh)

                while t < T:

                    # Contraction metrics initialization
                    delta_prev = None
                    rho_seq = []
                    delta_seq = []

                    # Solve the full collocation solver
                    if analysis:
                        for u in self.u_0_collocation.subfunctions:
                            u.assign(self.u_collocation.subfunctions[-1])

                        t0 = time.perf_counter()
                        self.collocation_solver.solve()
                        collocation_wall_time = time.perf_counter() - t0

                        convergence_results[f"{step},{t},full_collocation_timing"] = [
                            {
                                "solver_index": "full_collocation",
                                "wall_time": collocation_wall_time,
                            }
                        ]

                    # Apply the sweep
                    for k in range(1, sweeps + 1):
                        _set_scale(k)
                        self.u_k_prev.assign(self.u_k_act)
                        # Calculate the new values in the efficient way if
                        # no analytics happening, if not use call stack
                        if analysis:
                            self._sweep_loop()
                        else:
                            for s in self.sweep_solvers:
                                s.solve()

                        if analysis:
                            try:
                                du = Function(
                                    self.u_k_act.subfunctions[-1].function_space()
                                )
                                du.assign(self.u_k_act.subfunctions[-1])
                                du -= self.u_k_prev.subfunctions[-1]
                                delta = float(norm(du, norm_type="L2"))
                                delta_seq.append(delta)

                                eps = 1e-14
                                if delta_prev is not None and delta_prev > eps:
                                    rho_seq.append(float(delta / delta_prev))

                                delta_prev = delta

                            except Exception:
                                pass

                            _update_exact_field(t) if use_exact else None
                            analysis_metrics = self._compute_analysis_metrics(
                                real_u if use_exact else None,
                                analysis,
                                use_exact,
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
                                        if m.get("solver_index") == row["solver_index"]
                                    ),
                                    {},
                                )
                                timings.append({**meta, **row})

                            convergence_results[f"{step},{t},{k}_timings"] = timings

                            err_intra.append(
                                analysis_metrics["sweep_vs_collocation_compound_norm"]
                                if analysis
                                else None
                            )

                            print(
                                f"step {step}  t={t:.4e}  "
                                f"res_sweep={analysis_metrics['total_residual_sweep']:.3e}  "
                                f"err_coll={analysis_metrics['sweep_vs_collocation_errornorm']}"
                            )
                    print("\n\n\n")

                    # --- Dyadic save: only when stride matches ---
                    if (step % SAVE_STRIDE) == 0:
                        _maybe_save_checkpoint(afile, t)

                    # Synchronize states across subfunctions
                    u_last_node = self.u_k_act.subfunctions[-1]
                    for sub in (
                        *self.u_k_act.subfunctions,
                        *self.u_k_prev.subfunctions,
                        *self.u_0.subfunctions,
                    ):
                        sub.assign(u_last_node)

                    # Advance physical time and book-keeping
                    t += self.deltat
                    self.t_0_subinterval.assign(t)
                    if self.PDEs.time_dependent_constants_bts:
                        for ct in self.PDEs.time_dependent_constants_bts:
                            ct.assign(t)
                    step += 1

                if not wrote_T:
                    _maybe_save_checkpoint(afile, T)

                convergence_results_path = (
                    Path(self.file).with_suffix("").as_posix()
                    + "_convergence_results.json"
                )
                with open(str(convergence_results_path), "w") as f:
                    json.dump(convergence_results, f, indent=2)
                return step - 1

        else:
            vtk = VTKFile(self.file)
            u_out = self.u_k_act.subfunctions[-1]

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

                if analysis:
                    self.collocation_solver.solve()
                    for u in self.u_0_collocation.subfunctions:
                        u.assign(self.u_collocation.subfunctions[-1])

                if use_exact:
                    _update_exact_field(t)

                for k in range(1, sweeps + 1):
                    _set_scale(k)
                    self.u_k_prev.assign(self.u_k_act)

                    if analysis:
                        self._sweep_loop()
                    else:
                        for s in self.sweep_solvers:
                            s.solve()

                    if analysis:
                        _update_exact_field(t) if use_exact else None
                        analysis_metrics = self._compute_analysis_metrics(
                            real_u if use_exact else None,
                            analysis,
                            use_exact,
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
                # --- Guardado diádico en VTK ---
                if (step % SAVE_STRIDE) == 0:
                    _maybe_save_vtk(vtk, vtk_coll, vtk_exact, t)

                # Sincronizar estados entre subfunciones
                for sub in (
                    *self.u_k_act.subfunctions,
                    *self.u_k_prev.subfunctions,
                    *self.u_0.subfunctions,
                ):
                    sub.assign(u_out)

                t += self.deltat
                self.t_0_subinterval.assign(t)
                if self.PDEs.time_dependent_constants_bts:
                    for ct in self.PDEs.time_dependent_constants_bts:
                        ct.assign(t)
                step += 1

            if not wrote_T:
                _maybe_save_vtk(vtk, vtk_coll, vtk_exact, T)

            convergence_results_path = (
                Path(self.file).with_suffix("").as_posix() + "_convergence_results.json"
            )
            with open(str(convergence_results_path), "w") as f:
                json.dump(convergence_results, f, indent=2)
            return step - 1
