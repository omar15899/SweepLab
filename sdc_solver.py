from dataclasses import dataclass
import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule
from FIAT.reference_element import DefaultLine


@dataclass
class SDCPreconditioners:
    M: float
    prectype: int | str = 0

    def __post_init__(self):
        # Calculate collocation nodes in [-1,1] (main parameter in collocation problem)
        gll_rule = GaussLobattoLegendreQuadratureLineRule(DefaultLine(), self.M)

        self.tau = 0.5 * (
            np.asarray(gll_rule.get_points()).flatten() + 1.0
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
        if self.prectype == 0:
            return np.diag(self.Q)
        elif self.prectype == "MIN-SR-NS":
            return np.diag(np.diag([tau / self.M for tau in self.tau]))
        else:
            raise Exception("there's no other preconditioners defined")


class SDCSolver(SDCPreconditioners):
    """
    Specific solver for SDC
    """

    def __init__(
        self,
        mesh,
        V,
        f,
        u0,
        bcs,
        M=4,
        dt=1e-3,
        is_linear=False,
        is_paralell=True,
        solver_parameters=None,
        prectype: int | str = 0,
    ):
        """
        Mesh : Predermined mesh
        f: python function class object where the expression of the
            f(t, x, u(t, x)) part of the heat equation is written in
            UFL. It needs to be already written in weak form and
            after integration by parts.
        V: Initial basis space
        bcs: python function calls object where its
        """
        # Initialise preconditioner infrastructure
        super().__init__(M=M, prectype=prectype)

        self.mesh = mesh
        self.V = V
        self.deltat = dt
        self.bcs = bcs
        self.f = f
        self.linear = is_linear
        self.solver_parameters = solver_parameters

        # Parametrise the mesh, this is crutial for defining f.
        self.x = SpatialCoordinate(self.mesh)  # x[0] = x, x[1] = y, x[n]= ...

        # In order to match spatial and temporal discretisation,
        # we create a MixedFunctionSpace in order to have a bag
        # of individual function space objects, so when we create
        # a function in this space we are creating M functions, one
        # for each node M defined
        self.W = MixedFunctionSpace([self.V] * self.M)

        # Define the actual functions, if we want to retrieve
        # the list of functions for each coordinate use split.
        self.u_0 = Function(self.W, name="u_0")
        self.u_k_prev = Function(self.W, name="u_k")
        self.u_k_act = Function(self.W, name="u_{k+1}")

        # Instantiate the test functions, we cannot create a different TestFunction
        # like I used to do before (v_m = TestFunction(self.V) within the loop)
        if is_paralell:
            # Use internal setting:
            self.bcs = bcs(self.V, Constant(0.0), "on_boundary")
            self.v = None
        else:
            self.bcs = [
                bcs(self.W.sub(i), Constant(0.0), "on_boundary") for i in range(self.M)
            ]
            self.v = TestFunctions(self.W)

        # As all the functions are vectorial in the codomain due
        # to the nodal discretisation of the temporal axis
        for subfunction_0, subfunction_k_prev, subfunction_k_act in zip(
            self.u_0.subfunctions, self.u_k_prev.subfunctions, self.u_k_act.subfunctions
        ):
            subfunction_0.interpolate(u0)
            subfunction_k_prev.interpolate(u0)
            subfunction_k_act.interpolate(u0)

        # Initial time and instantiate the solvers
        self.t_0_subinterval = Constant(0.0)
        self._setup_paralell_solver() if is_paralell else self._setup_general_solver()

    def _solver_ensambler(self):
        pass

    def _setup_paralell_solver(self):
        """
        Compute the solvers
        """
        deltat = self.deltat
        tau = self.tau
        Q = self.Q
        Q_D = self.Q_D
        t0 = self.t_0_subinterval
        f = self.f
        v = self.v

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
                inner(u_m, v_m) - deltat * Q_D[m] * f(t0 + tau[m] * deltat, u_m, v_m)
            ) * dx  # f need to be composed with the change of variables

            # assemble part with u^{k}
            right = inner(self.u_0.subfunctions[0], v_m)
            for j in range(self.M):
                coeff = Q[m, j] - (Q_D[m] if j == m else 0.0)
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

            if self.linear:
                problem_m = LinearVariationalProblem(Rm, u_m, bcs=self.bcs)
                self.solvers.append(
                    LinearVariationalSolver(
                        problem_m,
                        solver_parameters=(
                            {
                                "ksp_type": "",
                                "pc_type": "",
                                "pc_hypre_type": "",
                                "ksp_rtol": 1e-8,
                            }
                            if not self.solver_parameters
                            else self.solver_parameters
                        ),
                    )
                )

            else:
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

    def _setup_general_solver(self):
        """
        Here we use the accumulated residual over W
        """

        deltat = self.deltat
        tau = self.tau
        Q = self.Q
        Q_D = self.Q_D
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

        for m in range(self.M):
            # As in my notes, each test function is independemt from the rest
            v_m = v[m]
            # retrieve m-coordinate of the vector function
            u_m = u_k_act_tup[m]
            #  assemble the part with u^{k+1}. We have to be very carefull as
            # v_m will be included in the function f.
            left = (
                inner(u_m, v_m) - deltat * Q_D[m] * f(t0 + tau[m] * deltat, u_m, v_m)
            ) * dx  # f need to be composed with the change of variables

            # assemble part with u^{k}
            right = inner(u_0.subfunctions[m], v_m)
            for j in range(self.M):
                coeff = Q[m, j] - (Q_D[m] if j == m else 0.0)
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

        if self.linear:
            # problem_m = LinearVariationalProblem(Rm, u_k_act, bcs=self.bcs)
            # self.solvers.append(
            #     LinearVariationalSolver(
            #         problem_m,
            #         solver_parameters=(
            #             {
            #                 "ksp_type": "",
            #                 "pc_type": "",
            #                 "pc_hypre_type": "",
            #                 "ksp_rtol": 1e-8,
            #             }
            #             if not self.solver_parameters
            #             else self.solver_parameters
            #         ),
            #     )
            # )
            pass

        else:
            print(type(u_k_act))
            # Colin asked me to use Nonlinear instead of Solve, is there any specific reason?
            problem_m = NonlinearVariationalProblem(Rm, u_k_act, bcs=self.bcs)
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
        t = 0.0
        step = 0
        output = VTKFile(
            "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/solver/heatSDC/sol.pvd"
        )
        while t < T:
            for _ in range(sweeps):
                self.u_k_prev.assign(self.u_k_act)
                for s in self.solvers:
                    s.solve()

            output.write(self.u_k_act.subfunctions[-1], time=t)
            # once all the sweeps are done, we upload with tau = 1 for the next subinterval.
            last = self.u_k_act.subfunctions[-1]
            for subfunction in self.u_k_act.subfunctions:
                subfunction.assign(last)
            for subfunction in self.u_k_prev.subfunctions:
                subfunction.assign(last)
            for subfunction in self.u_0.subfunctions:
                subfunction.assign(last)

            t += self.deltat
            self.t_0_subinterval.assign(t)
            step += 1
            print(f"step : {step},  time = {t}")


if __name__ == "__main__":

    Tfinal = 10.0
    dt = 1e-2
    M = 4
    nsweeps = 3

    mesh = UnitSquareMesh(100, 100)
    x = SpatialCoordinate(mesh)
    xx, yy = x[0], x[1]
    V = FunctionSpace(mesh, "CG", 1)

    u0_expr = exp(-150 * ((xx - 0.25) ** 2 + (yy - 0.75) ** 2)) + 0.6 * sin(
        pi * xx
    ) * sin(2 * pi * yy)

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
