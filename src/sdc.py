from typing import Iterable, Literal
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

        # Define the actual functions, if we want to retrieve
        # the list of functions for each coordinate use split.
        self.u_0 = Function(self.W, name="u_0")
        self.u_k_prev = Function(self.W, name="u_k")
        self.u_k_act = Function(self.W, name="u_{k+1}")

        # As all the functions are vectorial in the codomain due
        # to the nodal discretisation of the temporal axis
        for i, (subfunction_0, subfunction_k_prev, subfunction_k_act) in enumerate(
            zip(
                self.u_0.subfunctions,
                self.u_k_prev.subfunctions,
                self.u_k_act.subfunctions,
            ),
            0,
        ):
            u0 = self.PDEs.u0.subfunctions[i % self.lenV]
            subfunction_0.interpolate(u0)
            subfunction_k_prev.interpolate(u0)
            subfunction_k_act.interpolate(u0)

        # Initial time and instantiate the solvers
        self.t_0_subinterval = Constant(0.0)
        self.scale = Constant(1.0)
        (
            self._setup_paralell_solver_local()
            if is_local
            else self._setup_paralell_solver_global()
        )

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
        self.solvers = []

        for m in range(self.M):
            # As in my notes, each test function is independemt from the rest
            # v_m = v[m]
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
                right += (
                    deltat
                    * coeff
                    * f(
                        t0 + tau[j] * deltat,
                        self.u_k_prev.subfunctions[j],
                        v_m,
                    )
                )
            right = right * dx

            # Define the functional for that specific node
            Rm = left - right

            # Colin asked me to use Nonlinear instead of Solve, is there any specific reason?
            problem_m = NonlinearVariationalProblem(Rm, u_m, bcs=self.bcs)
            self.solvers.append(
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
        f = self.f
        # We store the solvers
        self.solvers = []
        # Instantiate general residual functional
        v = self.v
        u_k_act_tup = split(u_k_act)
        Rm = 0

        Q = self.Q
        Q_D = self.Q_D

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
            Rm += left - right

        # Colin asked me to use Nonlinear instead of Solve, is there any specific reason?
        problem_m = NonlinearVariationalProblem(
            Rm, u_k_act, bcs=self.boundary_conditions
        )
        self.solvers.append(
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

    def solve(self, T, sweeps):
        t, step = 0.0, 0

        if self.mode != "vtk":
            with CheckpointFile(self.file, "w") as afile:
                # Save the mesh
                afile.save_mesh(self.mesh)
                while t < T:
                    for k in range(1, sweeps + 1):
                        if self.prectype == "MIN-SR-FLEX":
                            self.scale.assign(1.0 / k)
                        else:
                            self.scale.assign(1.0)
                        self.u_k_prev.assign(self.u_k_act)
                        for s in self.solvers:
                            s.solve()
                    last = self.u_k_act.subfunctions[-1]
                    last.rename("u")
                    afile.save_function(
                        last, idx=step, timestepping_info={"time": float(t)}
                    )
                    for sub in (
                        *self.u_k_act.subfunctions,
                        *self.u_k_prev.subfunctions,
                        *self.u_0.subfunctions,
                    ):
                        sub.assign(last)
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
                    for s in self.solvers:
                        s.solve()
                vtk.write(self.u_k_act.subfunctions[-1], time=t)
                last = self.u_k_act.subfunctions[-1]
                for sub in (
                    *self.u_k_act.subfunctions,
                    *self.u_k_prev.subfunctions,
                    *self.u_0.subfunctions,
                ):
                    sub.assign(last)
                t += self.deltat
                self.t_0_subinterval.assign(t)
                if self.PDEs.time_dependent_constants_bts:
                    for ct in self.PDEs.time_dependent_constants_bts:
                        ct.assign(t)
                print(f"step: {step}, time = {t}")
                step += 1
