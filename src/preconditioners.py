from dataclasses import dataclass
import numpy as np
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
