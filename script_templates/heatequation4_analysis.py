# ---- RUN THIS in your repo venv ----
from pathlib import Path
import sys
import re

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR2 = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR2)])
import importlib, src.analysis_json as aj

importlib.reload(aj)
from src.analysis_json import JSONConvergenceAnalyser


# 0) Localiza tus JSON (búsqueda recursiva)
ROOT = Path(
    "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/cluster_results_heat_eq/used_new"
)  # <-- ajusta si hace falta


# 1) Instancia el analizador con tu patrón
pat = re.compile(
    r"""
^heat_
  n(?P<n>\d+)_
  dt(?P<dt>[0-9p.eE+\-]+)_         # p-floats tipo 1p25e-03
  sw(?P<sw>\d+)_
  nodes(?P<nodes>\d+)_
  deg(?P<deg>\d+)_
  prectype(?P<prectype>[^_]+)_     # p.ej. MIN-SR-FLEX
  (?P<solver>par|global)_          # modo solver
  tfinal(?P<tfinal>[0-9p.eE+\-]+)_ # p-floats tipo 1p00e+00
  (?P<idx>\d+)
  (?:
     _convergence_results\.json     # sidecar JSON
    | _log\.txt                     # log PETSc
    | \.h5                          # checkpoint
    | \.pvd                         # VTK
  )?$
""",
    re.VERBOSE,
)


# keys = ["n", "dt", "sw", "nodes", "deg", "prectype", "solver", "tfinal", "idx"]
# types = [int, float, int, int, int, str, str, float, int]

keys = ["n", "dt", "sw", "nodes", "deg", "prectype", "solver", "tfinal", "idx"]
types = [int, float, int, int, int, str, str, float, int]

# Instancia el analizador con el patrón correcto y la carpeta real
# A = JSONConvergenceAnalyser(ROOT, pat, keys, types)
A = JSONConvergenceAnalyser(ROOT, pat, keys, types, crit_first_n=3)

# # --- EOC (bonitas) automáticas: espacial y temporal ---
# print(
#     "[EOC] Generando EOC espaciales y temporales para todos los grupos con datos suficientes…"
# )
# _df = A.df.reset_index()

# # EOC espacial: agrupar por (prectype, solver, deg, nodes, sw, dt)
# cols_space = [
#     c for c in ["prectype", "solver", "deg", "nodes", "sw", "dt"] if c in _df.columns
# ]
# for gvals, gdf in _df.groupby(cols_space, dropna=False):
#     # exigir al menos dos N distintos para poder ajustar orden
#     if "n" in gdf.columns and gdf["n"].nunique() >= 2:
#         gdict = dict(zip(cols_space, gvals if isinstance(gvals, tuple) else (gvals,)))
#         try:
#             A.eoc_space(
#                 fixed_dt=float(gdict.get("dt")),
#                 fixed_sw=int(gdict.get("sw")),
#                 prectype=gdict.get("prectype"),
#                 solver=gdict.get("solver"),
#                 deg=int(gdict.get("deg")) if gdict.get("deg") is not None else None,
#                 nodes=(
#                     int(gdict.get("nodes")) if gdict.get("nodes") is not None else None
#                 ),
#             )
#         except Exception as e:
#             print(f"[EOC-space] Skip {gdict}: {e}")

# # EOC temporal: agrupar por (prectype, solver, deg, nodes, sw, n)
# cols_time = [
#     c for c in ["prectype", "solver", "deg", "nodes", "sw", "n"] if c in _df.columns
# ]
# for gvals, gdf in _df.groupby(cols_time, dropna=False):
#     # exigir al menos dos dt distintos
#     if "dt" in gdf.columns and gdf["dt"].nunique() >= 2:
#         gdict = dict(zip(cols_time, gvals if isinstance(gvals, tuple) else (gvals,)))
#         try:
#             A.eoc_time(
#                 fixed_n=int(gdict.get("n")),
#                 fixed_sw=int(gdict.get("sw")),
#                 prectype=gdict.get("prectype"),
#                 solver=gdict.get("solver"),
#                 deg=int(gdict.get("deg")) if gdict.get("deg") is not None else None,
#                 nodes=(
#                     int(gdict.get("nodes")) if gdict.get("nodes") is not None else None
#                 ),
#             )
#         except Exception as e:
#             print(f"[EOC-time] Skip {gdict}: {e}")

# print("[EOC] Figuras EOC guardadas en ./figures con stems únicos.")

# # --- Generación de figuras (sobre todos los runs encontrados) ---
# # Curvas de contracción δ_k vs k (guarda 'contraction_curves_*' y 'sweeps_contraction_*')
# A.sweeps_curves()

# # Convergence to the collocation problem at the LAST time step: k ↦ ||u^(k) - u_coll||.
# # Use a **sweep-indexed** metric: 'sweep_vs_collocation_error_norm'.
# # Note: 'collocation_vs_real_error_norm' is constant in k and only serves as a horizontal reference.
# A.sweep_vs_collocation_errornorm(
#     metric_candidates=["sweep_vs_collocation_error_norm"],
#     figure_per_case=True,
# )

# Boxplots de tiempos por sweep k (guarda 'timings_box_vs_k_*')
# A.timings_box_vs_k()

A.timings_box_by_node_and_by_sweep(
    exclude_first_sweep=False,
    exclude_first_node=False,
    # exclude_sweeps=[0],
    # exclude_nodes=[0],
)

# # Walltime acumulado del solver de colocación (guarda 'full_coll_walltime_*')
# A.full_coll_walltime()

# # Comparativa de acumulados: ruta crítica de SDC vs collocation (guarda 'cum_walltime_vs_step_*')
# A.cumulative_walltime_vs_step()

# # Work–precision en rejilla (filas=prectype, columnas=solver)
# A.work_precision_grid()

# # Barras de speedup paralelo (guarda 'speedup_bars.*')
# A.speedup_bars(overlay=True, use_truncated_crit=True)
# # A.speedup_bars(overlay=True)

# print("[OK] Figuras generadas en ./figures (PNG + PDF).")
