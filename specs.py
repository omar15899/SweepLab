from typing import Iterable
from dataclasses import dataclass
from firedrake import *


@dataclass
class PDEspec(object):
    V: FunctionSpace | Iterable[FunctionSpace]
    f: Function | Iterable[Function]
    u0: Function | Iterable[Function]
    boundary_conditions: callable | Iterable[callable]
