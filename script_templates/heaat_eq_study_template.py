# study_template.py
import re
from pathlib import Path
from pathlib import Path
import sys
import re

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR2 = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR2)])
import importlib, src.analysis_json as aj

importlib.reload(aj)
from src.analysis_json import JSONConvergenceAnalyser

# ============ CONFIG BÁSICA ============
# Carpeta raíz donde viven los *_convergence_results.json
ROOT = Path("/path/a/tus/resultados")  # <-- AJUSTA

# Regex de filename que ya usas (igual que en tu script)
# pat = re.compile(
#     r"""
# ^heat_
#   n(?P<n>\d+)_
#   dt(?P<dt>[0-9p.eE+\-]+)_
#   sw(?P<sw>\d+)_
#   nodes(?P<nodes>\d+)_
#   deg(?P<deg>\d+)_
#   prectype(?P<prectype>[^_]+)_
#   (?P<solver>par|global)_
#   tfinal(?P<tfinal>[0-9p.eE+\-]+)_
#   (?P<idx>\d+)
#   (?:
#      _convergence_results\.json
#     | _log\.txt
#     | \.h5
#     | \.pvd
#   )?$
# """,
#     re.VERBOSE,
# )
pat = re.compile(
    r"""
^heatHOMO_
  n(?P<n>\d+)_
  dt(?P<dt>[0-9p.eE+\-]+)_
  sw(?P<sw>\d+)_
  nodes(?P<nodes>\d+)_
  deg(?P<deg>\d+)_
  prectype(?P<prectype>[^_]+)_
  (?P<solver>par|global)_
  tfinal(?P<tfinal>[0-9p.eE+\-]+)_
  (?P<idx>\d+)
  (?:
     _convergence_results\.json
    | _log\.txt
    | \.h5
    | \.pvd
  )?$
""",
    re.VERBOSE,
)


keys = ["n", "dt", "sw", "nodes", "deg", "prectype", "solver", "tfinal", "idx"]
types = [int, float, int, int, int, str, str, float, int]

# ============ MÉTRICAS: ALIAS CÓMODOS ============
# Usa estos nombres cortos en tus llamadas; se mapean a columnas reales del DataFrame.
METRIC_ALIASES = {
    # errores L2
    "sweep_coll": "sweep_vs_collocation_errornorm",
    "sweep_real": "sweep_vs_real_errornorm",
    "coll_real": "collocation_vs_real_errornorm",
    "sweep_coll_comp": "sweep_vs_collocation_compound_norm",
    "sweep_real_comp": "sweep_vs_real_compound_norm",
    "coll_real_comp": "collocation_vs_real_compound_norm",
    # residuales
    "residual_sweep": "total_residual_sweep",
    "residual_coll": "total_residual_collocation",
    # seminorma H1
    "H1_sweep_coll": "sweep_vs_collocation_h1_seminorm",
    "H1_sweep_real": "sweep_vs_real_h1_seminorm",
    # norma L2 en tiempo (cuadratura nodal)
    "tL2_sweep_coll": "sweep_vs_collocation_time_l2",
    "tL2_sweep_real": "sweep_vs_real_time_l2",
}

YLABELS = {
    "sweep_vs_collocation_errornorm": r"$\|u^{(k)}-u^{\mathrm{coll}}\|_{L^2}$",
    "sweep_vs_real_errornorm": r"$\|u^{(k)}-u^{\mathrm{real}}\|_{L^2}$",
    "collocation_vs_real_errornorm": r"$\|u^{\mathrm{coll}}-u^{\mathrm{real}}\|_{L^2}$",
    "total_residual_sweep": r"$\|\mathcal{R}^{(k)}_{\text{sweep}}\|$",
    "total_residual_collocation": r"$\|\mathcal{R}_{\text{coll}}\|$",
    "sweep_vs_collocation_h1_seminorm": r"$|u^{(k)}-u^{\mathrm{coll}}|_{H^1}$",
    "sweep_vs_real_h1_seminorm": r"$|u^{(k)}-u^{\mathrm{real}}|_{H^1}$",
    "sweep_vs_collocation_time_l2": r"$\|u^{(k)}-u^{\mathrm{coll}}\|_{L^2_t(L^2_x)}$",
    "sweep_vs_real_time_l2": r"$\|u^{(k)}-u^{\mathrm{real}}\|_{L^2_t(L^2_x)}$",
}


# ============ *PATCH* PARA EOC CON MÉTRICAS ARBITRARIAS ============
# Permite A.eoc_time/space(metric="total_residual_sweep") sin tocar tu módulo.
def enable_custom_metric_for_eoc():
    old_pick = aj._pick_error_col

    def _pick_error_col(df, metric):
        if isinstance(metric, str) and metric in df.columns:
            return metric
        # acepta también alias cortos tipo "sweep_coll"
        if isinstance(metric, str) and metric in METRIC_ALIASES:
            name = METRIC_ALIASES[metric]
            if name in df.columns:
                return name
        return old_pick(df, metric)

    aj._pick_error_col = _pick_error_col


# ============ INICIALIZACIÓN DEL ANALIZADOR ============
def make_analyser(
    root: Path = ROOT, regex=pat, k=keys, t=types, crit_first_n: int | None = 3
):
    """
    crit_first_n: si quieres que el 'par_speedup' compute también la ruta crítica truncada a N.
    """
    A = JSONConvergenceAnalyser(root, regex, k, t, crit_first_n=crit_first_n)
    enable_custom_metric_for_eoc()
    return A


# ============ UTILIDADES ============
def metric_name(name_or_alias: str) -> str:
    """Devuelve el nombre de columna real que usaremos para graficar."""
    return METRIC_ALIASES.get(name_or_alias, name_or_alias)


def list_available_metrics(A: JSONConvergenceAnalyser):
    """Lista columnas de error/residual disponibles en A.df (útil para inspeccionar datasets)."""
    df = A.df.reset_index()
    cols = [
        c
        for c in df.columns
        if any(
            s in c
            for s in (
                "error",
                "residual",
                "h1",
                "time_l2",
                "err_L2",
                "sweep_coll",
                "coll",
            )
        )
    ]
    return sorted(cols)


# ============ ENVOLTURAS DE ALTO NIVEL ============
def plot_kcurve(
    A: JSONConvergenceAnalyser,
    metric: str = "sweep_coll",
    figure_per_case: bool = True,
    use_last_step_only: bool = True,
    **filters,
):
    """
    Curva k ↦ métrica elegida en el último paso temporal (por defecto),
    opcionalmente una figura por JSON.
    Filtros: prectype=..., solver=..., n=..., dt=..., nodes=..., sw=..., deg=...
    """
    col = metric_name(metric)
    ylabel = YLABELS.get(col, col)
    A.sweep_vs_collocation_errornorm(
        metric_candidates=[col],  # elegimos SÓLO esa métrica
        target=col,  # etiqueta interna; la rutina usará ylabel por defecto
        use_last_step_only=use_last_step_only,
        figure_per_case=figure_per_case,
        **filters,
    )
    print(f"[plot_kcurve] OK → métrica='{col}'  filtros={filters or '∅'}")


def eoc_time(
    A: JSONConvergenceAnalyser,
    fixed_n: int,
    fixed_sw: int,
    metric: str = "coll",
    **filters,
):
    """
    Orden temporal con todo fijo salvo Δt. El parámetro `metric` acepta:
      - 'sweep' | 'coll' | 'final' (comportamiento estándar)
      - Cualquier nombre de columna (p.ej. 'total_residual_sweep' o alias corto 'residual_sweep').
    """
    m = metric if metric in ("sweep", "coll", "final") else metric_name(metric)
    orders = A.eoc_time(fixed_n=fixed_n, fixed_sw=fixed_sw, metric=m, **filters)
    print(
        f"[eoc_time] p≈{orders.get('p'):.3f}  R²≈{orders.get('R2'):.3f}  metric={m}  filtros={filters or '∅'}"
    )
    return orders


def eoc_space(
    A: JSONConvergenceAnalyser,
    fixed_dt: float,
    fixed_sw: int,
    metric: str = "coll",
    **filters,
):
    """
    Orden espacial con todo fijo salvo N (h=1/N). Mismas reglas de `metric`.
    """
    m = metric if metric in ("sweep", "coll", "final") else metric_name(metric)
    orders = A.eoc_space(fixed_dt=fixed_dt, fixed_sw=fixed_sw, metric=m, **filters)
    print(
        f"[eoc_space] p≈{orders.get('p'):.3f}  R²≈{orders.get('R2'):.3f}  metric={m}  filtros={filters or '∅'}"
    )
    return orders


def speedup_overlay(
    A: JSONConvergenceAnalyser, use_truncated_crit: bool = True, **filters
):
    """
    Barras de speedup con overlay (T_coll/T_par detrás, T_seq/T_par delante).
    Requiere pares 'par' vs 'global' con mismas (prectype, n, dt, nodes, deg, sw).
    """
    A.speedup_bars(overlay=True, use_truncated_crit=use_truncated_crit, **filters)
    print(f"[speedup_overlay] OK  trunc={use_truncated_crit}  filtros={filters or '∅'}")


def onepager(A: JSONConvergenceAnalyser, **filters):
    """
    Tres paneles por run: δ_k, box per-k, acumulado SDC vs collocation.
    """
    A.onepager(**filters)
    print(f"[onepager] OK  filtros={filters or '∅'}")


# ============ RECETAS RÁPIDAS ============
def plot_all_cases_for_metric(A: JSONConvergenceAnalyser, metric: str):
    """
    Una figura por JSON para la métrica dada, sobre TODOS los ficheros.
    """
    plot_kcurve(A, metric=metric, figure_per_case=True)


def plot_subset_examples(A: JSONConvergenceAnalyser):
    """
    Ejemplos típicos que puedes adaptar.
    """
    # 1) Curvas k con residual (todos los casos FLEX par)
    plot_kcurve(A, metric="residual_sweep", prectype="MIN-SR-FLEX", solver="par")

    # 2) Curvas k con sweep vs real para un dt y N concretos
    plot_kcurve(A, metric="sweep_real", n=128, dt=5e-3, solver="par")

    # 3) EOC temporal con residual para un conjunto fijo
    eoc_time(
        A,
        fixed_n=128,
        fixed_sw=6,
        metric="residual_sweep",
        prectype="MIN-SR-FLEX",
        solver="par",
        nodes=6,
        deg=1,
    )

    # 4) EOC espacial con métrica de colocación (clásico)
    eoc_space(
        A,
        fixed_dt=5e-3,
        fixed_sw=6,
        metric="coll",
        prectype="MIN-SR-FLEX",
        solver="par",
        nodes=6,
        deg=1,
    )

    # 5) Speedup overlay global vs collocation para todo
    speedup_overlay(A, use_truncated_crit=True)


# ============ MAIN DEMO ============
if __name__ == "__main__":

    # ============================================================
    # =========== MÉTRICAS: ANALISIS COMPLETO DE TOOO ============
    # ============================================================
    # A = make_analyser(ROOT)
    # print("[init] runs:", len(A.df))
    # print("[metrics disponibles]\n  " + "\n  ".join(list_available_metrics(A)))

    # # Descomenta las recetas que quieras lanzar:
    # # plot_all_cases_for_metric(A, metric="sweep_coll")
    # # plot_subset_examples(A)

    # # Ejemplo suelto: métrica H1 si la guardaste en el JSON
    # # plot_kcurve(A, metric="H1_sweep_coll", prectype="MIN-SR-FLEX", solver="par")

    # A = make_analyser(ROOT)

    # # Una figura por JSON: k ↦ ||u^(k) - u_coll|| (todos los runs)
    # plot_all_cases_for_metric(A, metric="sweep_coll")

    # # Solo FLEX paralelo, N=128, dt=5e-3, M=6, sw=6: k ↦ residual
    # plot_kcurve(
    #     A,
    #     metric="residual_sweep",
    #     prectype="MIN-SR-FLEX",
    #     solver="par",
    #     n=128,
    #     dt=5e-3,
    #     nodes=6,
    #     sw=6,
    #     deg=1,
    # )

    # # EOC temporal con residual como observable (misma signatura que tus EOC)
    # eoc_time(
    #     A,
    #     fixed_n=128,
    #     fixed_sw=6,
    #     metric="residual_sweep",
    #     prectype="MIN-SR-FLEX",
    #     solver="par",
    #     nodes=6,
    #     deg=1,
    # )

    # # EOC espacial con H¹ (si la guardaste)
    # eoc_space(
    #     A,
    #     fixed_dt=5e-3,
    #     fixed_sw=6,
    #     metric="H1_sweep_coll",
    #     prectype="MIN-SR-FLEX",
    #     solver="par",
    #     nodes=6,
    #     deg=1,
    # )

    # # Barras de speedup overlay (ruta crítica truncada a firstN si configuraste crit_first_n)
    # speedup_overlay(A, use_truncated_crit=True)

    # ============================================================
    # =========== MÉTRICAS: Estudio exlusivo EOC TIME ============
    # ============================================================

    # === Ajusta la raíz a la carpeta que contiene los *_convergence_results.json de ese array ===
    # Por lo que has lanzado, debería ser algo así (cámbialo si tu sincronización usa otra ruta):
    ROOT = Path(
        "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/cluster_results_heat_eq/used_new/MIN_SR_FLEX_SPATIAL_CONVERGENCE_TOY/2067130"
    )

    A = make_analyser(ROOT)
    cols = A.df.reset_index().columns
    print(
        "ok" if "err_L2_coll" in cols and A.df["err_L2_coll"].notna().any() else "BAD"
    )
    print("[init] runs:", len(A.df))
    print("[metrics disponibles]\n  " + "\n  ".join(list_available_metrics(A)))

    # --- Filtros fijos de este estudio ---
    FILTERS = dict(
        prectype="MIN-SR-FLEX",
        solver="par",
        n=512,  # N espacial
        deg=3,  # grado polinomial p=3
        nodes=2,  # M=2 (Radau-right)
        sw=2,  # sweeps=2
    )

    # ===== (1) EOC temporal de la SOLUCIÓN DE COLOCACIÓN (orden del método) =====
    #   'coll' en tu API generalmente estima p de u_colloc vs "real"/ref.
    # orders_coll = eoc_time(A, fixed_n=512, fixed_sw=2, metric="coll", **FILTERS)
    # orders_coll = eoc_time(A, fixed_n=512, fixed_sw=2, metric="coll", **FILTERS)
    orders_coll = eoc_time(
        A,
        fixed_n=512,
        fixed_sw=2,
        metric="coll",
        **FILTERS,
        exclude_regex=[
            r"dt1p00e-06_.*_par_.*_convergence_results\.json$",
            r"dt1p00e-05_.*_par_.*_convergence_results\.json$",
        ],
    )
    print(orders_coll)

    # # (Opcional) si quieres ver también el “EOC temporal” de la solución tras los 2 sweeps:
    orders_sweep = eoc_time(A, fixed_n=512, fixed_sw=2, metric="sweep", **FILTERS)
    print(orders_sweep)

    # # ===== (2) Curva k para una dt intermedia (sanity check de convergencia de las iteraciones) =====
    # # Elige, por ejemplo, dt=1e-3 o 5e-4 (dentro de tu lista):
    # plot_kcurve(A, metric="sweep_coll", dt=1e-3, **FILTERS)
