from typing import Iterable, Literal
from pathlib import Path
import numpy as np
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
        is_local: bool = True,
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
        self.is_local = is_local
        self.solver_parameters = solver_parameters
        self.N = N
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
            self._setup_paralell_solver_local()
            if is_local
            else self._setup_paralell_solver_global()
        )

        self._setup_full_collocation() if self.full_collocation else None

    def _define_node_time_boundary_setup(self):
        """
        Need to be very awayre of how Firedrake flattens
        MixedFunctionSpace, we have a lot of tests in
        boundary_conditions.py. Also have a look to my notes
        in Notability.
        """

        if not self.PDEs.boundary_conditions:
            return []

        if self.is_local:
            return tuple(self.PDEs.boundary_conditions)

        bcs = []
        for bc in self.PDEs.boundary_conditions:
            if isinstance(bc, DirichletBC):
                bc_function_space = (
                    bc.function_space()
                )  # it gives us the subsubspace in which vs is defined.
                idx = (
                    bc_function_space.index
                    if bc_function_space.index is not None
                    else 0
                )
                component = getattr(bc_function_space, "component", None)
                # is_function_space = isinstance(self.V.sub(idx), FunctionSpace)

                for n in range(self.N):
                    subspace = self.W.sub(idx + n * self.lenV)
                    # As Firedrake flattens the MixedFunctionSpace,
                    # we cannot have another MixedFunctionSpace nested!
                    bcs.append(
                        bc.reconstruct(
                            V=(
                                subspace
                                if component is None
                                else subspace.sub(component)
                            )
                        )
                    )
            elif isinstance(bc, EquationBC):
                pass
            else:
                raise Exception("your bc is not accepted.")

        return tuple(bcs)

    def _setup_full_collocation(self):
        deltat = self.deltat
        tau = self.tau
        t0 = self.t_0_subinterval
        f = self.PDEs.f
        Q = self.Q
        # We could use the mixed space but it's nonsense, as we don't have coupling
        # among the different finite element subspaces.
        # We store the solvers
        self.sweep_solvers = []

        # We now instantiate also de residual of the original collocation problem
        # in order to study the convergence of the sweeps.
        R_coll = 0
        w = TestFunction(self.W)

        for m in range(self.M):

            # We deal with collocation problem:
            R_node_expr = inner(
                self.u_collocation.subfunctions[m]
                - self.u_0_collocation.subfunctions[m],
                w[m],
            )
            for j in range(self.M):
                R_node_expr -= (
                    deltat
                    * Q[m, j]
                    * f(t0 + tau[j] * deltat, self.u_collocation.subfunctions[j], w[m])
                )
            R_coll += R_node_expr * dx

        collocation_problem = NonlinearVariationalProblem(
            R_coll, self.u_collocation, bcs=self.bcs
        )
        self.R_coll = R_coll
        self.collocation_solver = NonlinearVariationalSolver(
            collocation_problem,
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

    def _setup_paralell_solver_local(self):
        """
        Compute the solvers:

        Needs to work for:

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

        # We now instantiate also de residual of the original collocation problem
        # in order to study the convergence of the sweeps.
        R_coll = 0
        w = TestFunction(self.W)

        for m in range(self.M):
            # As in my notes, each test function is independemt from the rest
            v_m = TestFunction(
                self.V
            )  # WHY WE HAVE TO CREATE THE FUNCTION OVER THE SUBSPACE V?
            # IS IT BECAUSE W IS FORMED BY DIFFERENT INDEPENDENT FINITE ELEMENT CELLS (ON A SAME CELL, M DIFFERENT AND INDEPENDENT FINITE ELEMENTS DEFINED)?
            # retrieve m-coordinate of the vector function
            u_m = self.u_k_act.subfunctions[m]

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

    def _setup_paralell_solver_global(self):
        """
        Here we use the accumulated residual over W
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

        t, step = 0.0, 0

        if real_solution_exp is not None:
            real_u = Function(self.W)
            for u in real_u.subfunctions:
                u.interpolate(real_solution_exp(SpatialCoordinate(self.W.mesh()), t))

        if self.mode != "vtk":
            with CheckpointFile(self.file, "w") as afile:
                # Save the mesh
                afile.save_mesh(self.mesh)
                while t < T:
                    self.collocation_solver.solve()
                    for u in self.u_0_collocation.subfunctions:
                        u.assign(self.u_collocation.subfunctions[-1])
                    for k in range(1, sweeps + 1):
                        if self.prectype == "MIN-SR-FLEX":
                            self.scale.assign(1.0 / k)
                        else:
                            self.scale.assign(1.0)

                        self.u_k_prev.assign(self.u_k_act)

                        #############################################
                        ############### Measuring errors ###############
                        #############################################

                        # RESIDUAL ERRORS
                        r_as = (
                            (assemble(Rm).riesz_representation() for Rm in self.R_sweep)
                            if self.is_local
                            else (assemble(self.R_sweep).riesz_representation(),)
                        )
                        print(f"Sweep residual norm: {sum(norm(r) for r in r_as)}")
                        res0 = assemble(self.R_coll).riesz_representation()
                        print(f"Initial collocation residual = {norm(res0)}")

                        for s in self.sweep_solvers:
                            s.solve()

                        # ERROR SWEEP SOLUTION VS COLLOCATION ERROR VS SWEEP ERROR
                        sweep_vs_collocation_errornorm = errornorm(
                            self.collocation.subfunctions[-1],
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
                            self.collocation.subfunctions[-1], real_u.subfunctions[-1]
                        )

                        print(
                            f"Sweep vs collocation error norm: {sweep_vs_collocation_errornorm}"
                        )
                        print(f"Sweep vs real error norm: {sweep_vs_real_errornorm}")
                        print(
                            f"Collocation vs real error norm: {collocation_vs_real_errornorm}"
                        )
                        #############################################
                        #############################################
                        #############################################

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
                    print(f"step: {step}, time = {t}")
                    step += 1

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
