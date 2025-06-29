from typing import Iterable
from dataclasses import dataclass
from firedrake import *


@dataclass
class PDEobj(object):
    V: FunctionSpace | Iterable[FunctionSpace]
    coord: SpatialCoordinate
    f: Function | Iterable[Function]
    u0: Function | Iterable[Function]
    boundary_conditions: callable | Iterable[callable]
    time_dependent_constants_bts: Constant | Iterable[Constant] | None = None
    name: str | None

    def __post_init__(self):
        # Create mixed function space. LO TENEMOS QUE METER NOSOTROS DE ANTES.
        W = (
            MixedFunctionSpace((V for V in self.V))
            if isinstance(self.V, Iterable)
            else self.V
        )

        # Define boundary conditions in an homogenieus way
        self.bcs = (
            [self.boundary_conditions]
            if not isinstance(self.bundary_conditions, Iterable)
            else self.boundary_conditions
        )

        self.time_dependent_constants_bts = (
            [self.time_dependent_constants_bts]
            if not isinstance(self.time_dependent_constants_bts, Iterable)
            else self.time_dependent_constants_bts
        )
