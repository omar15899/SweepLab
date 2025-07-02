from dataclasses import dataclass
from firedrake import *


@dataclass
class SDCfunctions:
    @staticmethod
    def _path_from_space(space):
        """
        returns de _indices attribute of the bcs, but as is an internal parameter
        im a little bit scared of using it.
        """
        path = []
        while getattr(space, "index", None) is not None:
            path.insert(0, space.index)
            space = space.parent
        return path

    @staticmethod
    def _follow_path(root, path):
        """We retrieve the actual subspace we want to be working with"""
        for i in path:
            root = root.sub(i)
        return root

    @staticmethod
    def _define_node_time_boundary_setup(bcs_original, W, M):

        if not bcs_original:
            return []

        bcs = []

        bcs = []
        # We go over all the copies of the mixed space wrt the temporal nodes
        for m in range(M):
            # Now we characterise that specific brunch
            V_m = W.sub(m)
            # Test with functions belong to this space
            for bc in bcs_original:
                path = PDESystem._path_from_space(bc.function_space())
                tgt_space = PDESystem._follow_path(V_m, path)
                if isinstance(bc, DirichletBC):
                    bcs.append(bc.reconstruct(V=tgt_space, indices=[]))

                elif isinstance(bc, EquationBC):
                    pass
                else:
                    raise TypeError(f"Not supported boundary type {type(bc)}")

        return bcs


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
    """

    mesh: Mesh
    V: FunctionSpace | VectorFunctionSpace | MixedFunctionSpace
    coord: SpatialCoordinate
    f: Function
    u0: Function  # (if system is mixed, use Function(V) withc .subfunctions[] already specified)
    boundary_conditions: tuple[DirichletBC | EquationBC]
    time_dependent_constants_bts: Constant | tuple[Constant] | None = None
    name: str | None

    def __post_init__(self):
        self._is_Mixed = isinstance(self.V, MixedFunctionSpace)
        # Define boundary conditions in an homogenieus way (always a list)
        self.bcs = (
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
