from firedrake import *
from firedrake.checkpointing import CheckpointFile

path = "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/solver/tests/heatfiles/HE/heat_equation_sdc_2.h5"  # o el nombre que te haya devuelto FileNamer
with CheckpointFile(path, "r") as afile:
    times, indices = afile.get_timesteps()
    last_time = times[-1]
    last_index = indices[-1]
    mesh = afile.load_mesh()
    u_approx = afile.load_function(mesh, "u", idx=5000)

x = SpatialCoordinate(mesh)[0]

V = FunctionSpace(mesh, "CG", degree=4)

u_exact_expr = lambda t: sin(pi * x) * exp(x * t)
u_exact = Function(V).interpolate(u_exact_expr(0.5))
l2_err = errornorm(u_exact, u_approx, norm_type="L2")

print(l2_err)
