from dataclasses import dataclass
from typing import Literal
import numpy as np
from FIAT.quadrature import (
    GaussLobattoLegendreQuadratureLineRule,
    RadauQuadratureLineRule,
)
from FIAT.reference_element import DefaultLine


@dataclass
class SDCPreconditioners:
    M: float
    prectype: int | Literal["lobatto", "radau-left", "radau-right"] = "radau-right"
    tau: np.ndarray | None = None
    """
    Optional to use a personal quadrature rule, I have to add more options to 
    the Gauss Lobatto one
    """

    def __post_init__(self):

        if self.tau is None:
            # Calculate collocation nodes in [-1,1] (main parameter in collocation problem)
            if self.prectype == "lobatto":
                rule = GaussLobattoLegendreQuadratureLineRule(DefaultLine(), self.M)

            elif self.prectype == "radau-left":
                # Includes the extreme x = −1  ->  tau = 0
                rule = RadauQuadratureLineRule(DefaultLine(), self.M, right=False)

            elif self.prectype == "radau-right":
                # Includes the extreme x = 1  ->  tau = 1
                rule = RadauQuadratureLineRule(DefaultLine(), self.M, right=True)
            else:
                raise ValueError(f"Unknown quadrature: {self.prectype!r}")

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
