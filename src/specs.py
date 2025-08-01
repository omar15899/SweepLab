from dataclasses import dataclass
from firedrake import *
from typing import Union, List


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
    f: function | List[function]
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
