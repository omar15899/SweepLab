from typing import Iterable, Literal, List
from pathlib import Path
import json
import numpy as np
import pandas as pd
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
        N: int = 1,
        dt: int | float = 1e-3,
        is_parallel: bool = True,
        solver_parameters: dict | None = None,
        prectype: int | str = 0,
        tau: np.ndarray | None = None,
        file_name: str = "solution",
        folder_name: str | None = None,
        path_name: str | None = None,
        mode: Literal["checkpoint", "vtk", "pdf"] = "checkpoint",
        full_collocation: bool = False,
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
        self.full_collocation = full_collocation

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
        self.bcs = self._define_node_time_boundary_setup()

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

        self._setup_full_collocation_solver() if self.full_collocation else None

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

        +. No matter if in _setup_paralell_sweep_solver we are working with
            self.V test,


        """

        if not self.PDEs.bcs:
            return []

        # if self.is_parallel:
        #     return list(self.PDEs.boundary_conditions)

        bcs = []
        for bc in self.PDEs.bcs:
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
                    bcs.append(bc.reconstruct(V=(subspace)))
            elif isinstance(bc, EquationBC):
                pass
            else:
                raise Exception("your bc is not accepted.")

        return tuple(bcs)

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
        R_coll = 0
        for m in range(self.M):
            Rm = inner(u_c_split[m] - u0_c_split[m], w[m])
            for j in range(self.M):
                Rm -= deltat * Q[m, j] * f(t0 + tau[j] * deltat, u_c_split[j], w[m])
            R_coll += Rm * dx

            self.R_coll = R_coll

            problem = NonlinearVariationalProblem(R_coll, u_c, bcs=self.bcs)
            self.collocation_solver = NonlinearVariationalSolver(
                problem,
                solver_parameters=self.solver_parameters
                or {
                    "snes_type": "newtonls",
                    "snes_rtol": 1e-8,
                    "ksp_type": "cg",
                },
            )

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
        R_coll = 0

        for m in range(self.M):
            # As in my notes, each test function is independemt from the rest
            u_m = self.u_k_act.subfunctions[m]
            v_m = TestFunction(u_m.function_space())

            #  assemble the part with u^{k+1}. We have to be very carefull as
            # v_m will be included in the function f.
            left = (
                inner(u_m, v_m)
                - deltat * self.scale * Q_D[m, m] * f(t0 + tau[m] * deltat, u_m, v_m)
            ) * dx  # f need to be composed with the change of variables

            # assemble part with u^{k}
            right = inner(self.u_0.subfunctions[m], v_m)
            for j in range(self.M):
                coeff = Q[m, j] - self.scale * Q_D[m, j]
                f_value = f(
                    t0 + tau[j] * deltat,
                    self.u_k_prev.subfunctions[j],
                    v_m,
                )

                right += deltat * coeff * f_value

            right = right * dx

            # Define the functional for that specific node
            R_sweep = left - right

            self.R_sweep.append(R_sweep)

            # Colin asked me to use Nonlinear instead of Solve, is there any specific reason?
            problem_m = NonlinearVariationalProblem(R_sweep, u_m, bcs=self.bcs)
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
        u_k_act_tup = split(u_k_act)
        R_sweep = 0

        Q = self.Q
        Q_D = self.Q_D
        R_coll = 0

        self.sweep_solvers = []
        self.R_sweep = []

        for m in range(self.M):
            # As in my notes, each test function is independemt from the rest
            v_m = v[m]
            # retrieve m-coordinate of the vector function
            u_m = u_k_act_tup[m]
            #  assemble the part with u^{k+1}. We have to be very carefull as
            # v_m will be included in the function f.
            left = (
                inner(u_m, v_m)
                - deltat * self.scale * Q_D[m, m] * f(t0 + tau[m] * deltat, u_m, v_m)
            ) * dx  # f need to be composed with the change of variables

            # assemble part with u^{k}
            right = inner(u_0.subfunctions[m], v_m)
            for j in range(self.M):
                coeff = Q[m, j] - self.scale * Q_D[m, j]
                right += (
                    deltat
                    * coeff
                    * f(
                        t0 + tau[j] * deltat,
                        u_k_prev.subfunctions[j],
                        v_m,
                    )
                )
            right = right * dx

            # Add to general residual functional
            R_sweep += left - right

        self.R_sweep = R_sweep

        # Colin asked me to use Nonlinear instead of Solve, is there any specific reason?
        problem_m = NonlinearVariationalProblem(R_sweep, u_k_act, bcs=self.bcs)
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

    def solve(self, T, sweeps, real_solution_exp: Function = None):
        """
        real_solution_exp = ufl expression to be projected over the W space dependent
        on space and time if needed. X will be inputed considering that they
        are going to be the coordinates OBJECT over the mesh.
        """

        def _update_exact_field(t_now):
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

        def compound_norm(u: Function, v: Function):
            return sum(
                errornorm(u_k, v_k, norm_type="L2") ** 2
                for u_k, v_k in zip(u.subfunctions, v.subfunctions)
            )

        t, step = 0.0, 0

        convergence_results = {}

        if real_solution_exp is not None:
            real_u = Function(self.W, name="u_exact")
            for u in real_u.subfunctions:
                u.interpolate(real_solution_exp(SpatialCoordinate(self.W.mesh()), t))

        if self.mode != "vtk":
            with CheckpointFile(self.file, "w") as afile:
                # Save the mesh
                afile.save_mesh(self.mesh)
                while t < T:
                    err_intra = []
                    _update_exact_field(t)
                    # Solve the full collocation solver
                    self.collocation_solver.solve()
                    for u in self.u_0_collocation.subfunctions:
                        u.assign(self.u_collocation.subfunctions[-1])

                    # Apply the sweep
                    for k in range(1, sweeps + 1):
                        if self.prectype == "MIN-SR-FLEX":
                            self.scale.assign(1.0 / k)
                        else:
                            self.scale.assign(1.0)

                        self.u_k_prev.assign(self.u_k_act)

                        #############################################
                        ############### Measuring errors ###############
                        #############################################
                        # print(f"step: {step}, time = {t}")
                        # print("-------------------------------------------------")
                        # RESIDUAL ERRORS
                        residual_sweep = (
                            (assemble(Rm).riesz_representation() for Rm in self.R_sweep)
                            if self.is_parallel
                            else (assemble(self.R_sweep).riesz_representation(),)
                        )
                        total_residual_sweep = sum(norm(r) for r in residual_sweep)
                        # print(f"Sweep residual norm: {total_residual_sweep}")
                        residual_collocation = assemble(
                            self.R_coll
                        ).riesz_representation()
                        total_residual_collocation = norm(residual_collocation)
                        # print(
                        # f"Initial collocation residual = {total_residual_collocation}"
                        # )

                        for s in self.sweep_solvers:
                            s.solve()

                        # ERROR SWEEP SOLUTION VS COLLOCATION ERROR VS SWEEP ERROR
                        sweep_vs_collocation_errornorm = errornorm(
                            self.u_collocation.subfunctions[-1],
                            self.u_k_act.subfunctions[-1],
                            norm_type="L2",
                        )

                        sweep_vs_real_errornorm = (
                            (
                                errornorm(
                                    real_u.subfunctions[-1],
                                    self.u_k_act.subfunctions[-1],
                                    norm_type="L2",
                                )
                            )
                            if real_solution_exp is not None
                            else None
                        )

                        collocation_vs_real_errornorm = errornorm(
                            self.u_collocation.subfunctions[-1],
                            real_u.subfunctions[-1],
                        )

                        # err_intra.append(float(sweep_vs_collocation_errornorm))

                        err_intra.append(
                            float(compound_norm(self.u_collocation, self.u_k_act))
                        )

                        print(
                            f"Sweep vs collocation error norm: {sweep_vs_collocation_errornorm}"
                        )
                        # print(f"Sweep vs real error norm: {sweep_vs_real_errornorm}")
                        # print(
                        #     f"Collocation vs real error norm: {collocation_vs_real_errornorm}"
                        # )
                        # print("\n")

                        convergence_results[f"{step},{t},{k}"] = [
                            total_residual_collocation,
                            total_residual_sweep,
                            sweep_vs_collocation_errornorm,
                            sweep_vs_real_errornorm,
                            collocation_vs_real_errornorm,
                        ]

                        #############################################
                        #############################################
                        #############################################

                    print(f"step {step}  t={t:.4e}  err_intra={err_intra}")
                    # print("\n\n\n")

                    u_last_node = self.u_k_act.subfunctions[-1]
                    u_last_node.rename("u")
                    afile.save_function(
                        u_last_node, idx=step, timestepping_info={"time": float(t)}
                    )
                    for sub in (
                        *self.u_k_act.subfunctions,
                        *self.u_k_prev.subfunctions,
                        *self.u_0.subfunctions,
                    ):
                        sub.assign(u_last_node)

                    t += self.deltat
                    self.t_0_subinterval.assign(t)
                    if self.PDEs.time_dependent_constants_bts:
                        for ct in self.PDEs.time_dependent_constants_bts:
                            ct.assign(t)
                    step += 1

                convergence_results_path = (
                    Path(self.file).with_suffix("").as_posix()
                    + "_convergence_results.json"
                )
                with open(str(convergence_results_path), "w") as f:
                    json.dump(convergence_results, f, indent=2)
                return step - 1

        else:
            vtk = VTKFile(self.file)
            while t < T:
                for k in range(1, sweeps + 1):
                    if self.prectype == "MIN-SR-FLEX":
                        self.scale.assign(1.0 / k)
                    else:
                        self.scale.assign(1.0)
                    self.u_k_prev.assign(self.u_k_act)
                    for s in self.sweep_solvers:
                        s.solve()
                vtk.write(self.u_k_act.subfunctions[-1], time=t)
                u_last_node = self.u_k_act.subfunctions[-1]
                for sub in (
                    *self.u_k_act.subfunctions,
                    *self.u_k_prev.subfunctions,
                    *self.u_0.subfunctions,
                ):
                    sub.assign(u_last_node)
                t += self.deltat
                self.t_0_subinterval.assign(t)
                if self.PDEs.time_dependent_constants_bts:
                    for ct in self.PDEs.time_dependent_constants_bts:
                        ct.assign(t)
                # print(f"step: {step}, time = {t}")
                step += 1

        return step - 1 if self.mode != "vtk" else None
