import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from firedrake import *
from itertools import product
from src.sdc import SDCSolver
from src.specs import PDESystem
from datetime import datetime

# Marca temporal para carpetas/archivos
now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")


def solve_heat_pde(
    dt,
    n_cells,
    nsweeps,
    M,
    Tfinal,
    is_parallel=True,
    prectype="MIN-SR-FLEX",
    degree=2,
    analysis=False,
    mode="checkpoint",
    # ===== NUEVOS (opcionales) =====
    kappa_par=1.0,  # difusividad principal (dirección local "rápida")
    kappa_perp_min=0.06,  # difusividad secundaria mínima
    kappa_perp_max=0.30,  # difusividad secundaria máxima
    omega_rot=0.6,  # velocidad angular de la anisotropía
    Q_sin=1.4,  # peso del modo sinusoidal espaciotemporal
    Q_rot1=1.3,  # peso gaussiana 1
    Q_rot2=1.1,  # peso gaussiana 2
    Q_spiral=1.1,  # peso del anillo espiral
    sigma2=0.018,  # varianza de las gaussianas móviles
):
    """
    Ecuación de calor 2D en [0,1]^2 con difusión anisótropa rotante:
      u_t - ∇·(K(x,t) ∇u) = f(x,t)

    f(x,t) = Q_sin sin(2π t) sin(2π x) sin(3π y)
             + Q_rot1 * exp(-|x - c1(t)|^2 / sigma2)
             + Q_rot2 * exp(-|x - c2(t)|^2 / sigma2)
             + Q_spiral * exp(-((r - r0)^2)/(2σ_r^2)) * cos(m φ - ω_s t)

    CC: Dirichlet homogénea u|_{∂Ω} = 0.
    """

    # Strings para nombres de salida (igual estilo)
    dt_str = f"{dt:.2e}".replace(".", "p")
    Tfinal_str = f"{Tfinal:.2e}".replace(".", "p")
    file_name = (
        f"heat2D_n{n_cells}x{n_cells}_dt{dt_str}_sw{nsweeps}_"
        f"nodes{M}_deg{degree}_kpar{str(kappa_par).replace('.', 'p')}_"
        f"prectype{prectype}_tfinal{Tfinal_str}_par{str(is_parallel)}"
    )

    # Malla y espacios
    mesh = UnitSquareMesh(n_cells, n_cells)
    x = SpatialCoordinate(mesh)
    xx, yy = x
    V = FunctionSpace(mesh, "CG", degree=degree)

    # Estado inicial: pulso + altas frecuencias suaves (para textura)
    u0_expr = (
        exp(-180.0 * ((xx - 0.22) ** 2 + (yy - 0.78) ** 2))
        + 0.5 * sin(2 * pi * xx) * sin(3 * pi * yy)
        + 0.15 * cos(5 * pi * xx) * sin(4 * pi * yy)
    )
    u0 = Function(V, name="u0").interpolate(u0_expr)

    # ---- Difusión anisótropa rotante K(x,t) = R(θ) diag(k_par, k_perp(x)) R^T
    r = sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    # k_perp(x): modulación radial para generar “anillos” de contraste
    k_perp_spatial = kappa_perp_min + (kappa_perp_max - kappa_perp_min) * 0.5 * (
        1 + cos(8 * pi * r)
    )

    def K_tensor(t):
        theta = omega_rot * 2 * pi * t
        c, s = cos(theta), sin(theta)
        R = as_matrix(((c, -s), (s, c)))
        D = as_matrix(((kappa_par, 0.0), (0.0, k_perp_spatial)))
        return R * D * R.T

    # ---- Fuentes “coreografiadas”
    # Dos gaussianas en contrarrotación + modo sinusoidal + anillo espiral
    r1, r2 = 0.28, 0.18

    def centers(t):
        c1x = 0.5 + r1 * cos(2 * pi * t)
        c1y = 0.5 + r1 * sin(2 * pi * t)
        c2x = 0.5 + r2 * cos(-2 * pi * t + pi / 7)
        c2y = 0.5 + r2 * sin(-2 * pi * t + pi / 7)
        return c1x, c1y, c2x, c2y

    r0, sig_r, m, omega_s = 0.32, 0.06, 5, 3.5 * 2 * pi
    phi = atan2(yy - 0.5, xx - 0.5)

    def f_heat(t, u, v):
        # Tensor anisótropo
        Kt = K_tensor(t)

        # Centros de las gaussianas
        c1x, c1y, c2x, c2y = centers(t)

        # Componentes de la fuente
        src_sin = Q_sin * sin(2 * pi * t) * sin(2 * pi * xx) * sin(3 * pi * yy)
        src_rot1 = Q_rot1 * exp(-((xx - c1x) ** 2 + (yy - c1y) ** 2) / sigma2)
        src_rot2 = Q_rot2 * exp(-((xx - c2x) ** 2 + (yy - c2y) ** 2) / sigma2)
        src_spiral = (
            Q_spiral
            * exp(-((r - r0) ** 2) / (2 * sig_r**2))
            * cos(m * phi - omega_s * t)
        )

        # Forma débil (misma plantilla que tenías)
        return (
            -inner(Kt * grad(u), grad(v))
            + (src_sin + src_rot1 + src_rot2 + src_spiral) * v
        )

    # Condición de contorno Dirichlet homogénea
    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    # Sistema PDE (mismo patrón)
    pde = PDESystem(
        mesh=mesh,
        V=V,
        coord=SpatialCoordinate(mesh),
        f=f_heat,
        u0=u0,
        boundary_conditions=(bc,),
        name="Heat2D_AnisoRotSpiral",
    )

    # Solver (idéntico esquema)
    solver = SDCSolver(
        mesh=mesh,
        PDEs=pde,
        M=M,
        dt=dt,
        is_parallel=is_parallel,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_rtol": 1e-12,
            "snes_atol": 1e-14,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        prectype=prectype,
        file_name=file_name,
        folder_name=f"HE2D_{time_str}",
        path_name=(
            "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/"
            "programming/solver/tests/heatfiles"
        ),
        analysis=analysis,
        mode=mode,  # salida para visualización
    )

    # Ejecutar (igual que en tu plantilla)
    solver.solve(Tfinal, nsweeps)


# ====== Barrido de parámetros (idéntico patrón) ======
N_CELLS = [120]  # más fino para detalles
# DT_LIST = [4e-4]  # paso temporal algo menor para fuentes rápidas
DT_LIST = [4e-2]  # paso temporal algo menor para fuentes rápidas
SWEEPS = [4]
DEGREE = [2]
M = 4
TFINAL = 10

for n, dt, sw, deg in product(N_CELLS, DT_LIST, SWEEPS, DEGREE):
    solve_heat_pde(
        dt=dt,
        n_cells=n,
        nsweeps=sw,
        M=M,
        Tfinal=TFINAL,
        is_parallel=False,
        prectype="MIN-SR-FLEX",
        degree=deg,
        analysis=False,
        mode="vtk",
        # Puedes ajustar estos “diales” si lo deseas:
        kappa_par=1.0,
        kappa_perp_min=0.06,
        kappa_perp_max=0.30,
        omega_rot=0.6,
        Q_sin=1.4,
        Q_rot1=1.3,
        Q_rot2=1.1,
        Q_spiral=1.1,
        sigma2=0.018,
    )
