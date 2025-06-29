from typing import Iterable
from dataclasses import dataclass
from firedrake import *


@dataclass
class PDEobj(object):
    """
    V must be the definitive space for that PDE conforming the system.
    If we have one scalar pde, just use FunctionSpace, if we have a
    vector PDE use VectorFunctionSpace, if we have a mixed system use MixedFunctionSpace.
    Could we use also this object to compact the whole potential system?
    ej, we have a system of mixed scalar and vector pde's, could we incorpore
    them inmediately here or better to do it separated

    For now:
    - If just one PDE scalar, instantiate as only one object
    - If system of PDE scalar, instantiate an object for each??
    - If vector PDE, instantiante as one object using VectorFunctionSpace or MixedFunctionSpace
    - If system of vector + scalar PDE, instantiate each as one object using VectorFunctionSpace or MixedFunctionSpace
    """

    V: FunctionSpace | VectorFunctionSpace | MixedFunctionSpace
    coord: SpatialCoordinate
    f: Function | Iterable[Function]
    u0: Function | Iterable[Function]
    boundary_conditions: callable | Iterable[callable]
    time_dependent_constants_bts: Constant | Iterable[Constant] | None = None
    name: str | None

    def __post_init__(self):
        # Create mixed function space. LO TENEMOS QUE METER NOSOTROS DE ANTES.

        # Define boundary conditions in an homogenieus way (always a list)
        self.bcs = (
            [self.boundary_conditions]
            if not isinstance(self.bundary_conditions, Iterable)
            else list(self.boundary_conditions)
        )

        self.time_dependent_constants_bts = (
            [self.time_dependent_constants_bts]
            if not isinstance(self.time_dependent_constants_bts, Iterable)
            else list(self.time_dependent_constants_bts)
        )
