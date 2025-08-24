# ---- RUN THIS in your repo venv ----
from pathlib import Path
import sys, re
import pandas as pd
import numpy as np

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR2 = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT_DIR), str(ROOT_DIR2)])
from src.analysis_json import JSONConvergenceAnalyser

# ====== CONFIG ======
# Carpeta con resultados (JSON sidecars)
ROOT = Path(
    "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/cluster_results_heat_eq/used"
)

# Filtros globales (acotan figura/IO)
PREC_OK = ["MIN-SR-FLEX", "MIN-SR-NS"]
SOLVERS_EOC = ["par"]  # EOC: solo 'par' para evitar duplicar figuras
SOLVERS_WP = ["par", "global"]  # para speedup y work-precision
M_SET = [1, 2, 3, 4, 5, 6]
N_EOC_H = [32, 64, 128, 256]
DT_EOC_DT = [1e-2, 5e-3, 2.5e-3, 1.25e-3]

# Subconjuntos “representativos” para figuras pesadas
REP = dict(
    N=128,
    DT=2.5e-3,
    MS=[3, 6],  # dos Ms para barrer figuras intensivas
)

# Patrón de ficheros (coincide con heat_* que ya usas)
pat = re.compile(
    r"""
^heat_
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

# ====== LOAD ======
A = JSONConvergenceAnalyser(ROOT, pat, keys, types)
df = A.df.reset_index()


# Filtrado “sano”: solo p=1, prectypes y solvers esperados, M en 1..6
def sane_filter(df_):
    out = df_.copy()
    if "deg" in out:
        out = out[out["deg"] == 1]
    if "prectype" in out:
        out = out[out["prectype"].isin(PREC_OK)]
    if "solver" in out:
        out = out[out["solver"].isin(PREC_OK and (SOLVERS_EOC + SOLVERS_WP))]
    if "nodes" in out:
        out = out[out["nodes"].isin(M_SET)]
    return out


df = sane_filter(df)

# ====== EOC (espacio y tiempo) ======
# Guardaremos órdenes en CSV
rows_eoc = []

# EOC–h: por (prectype, solver='par', M) con los dt presentes en datos
for p in PREC_OK:
    for s in SOLVERS_EOC:
        for M in M_SET:
            # buscamos dt que existan en datos y crucen al menos dos N
            g = df[(df["prectype"] == p) & (df["solver"] == s) & (df["nodes"] == M)]
            if g.empty:
                continue
            for dt in sorted(g["dt"].dropna().unique()):
                # exigir al menos 2 N
                g2 = g[g["dt"] == dt]
                if g2["n"].nunique() < 2:
                    continue
                try:
                    res = A.eoc_space(
                        fixed_dt=float(dt),
                        fixed_sw=int(M),  # sw==M por tus jobs
                        prectype=p,
                        solver=s,
                        nodes=M,
                        deg=1,
                    )
                    rows_eoc.append(
                        dict(
                            kind="space",
                            prectype=p,
                            solver=s,
                            nodes=M,
                            dt=float(dt),
                            p_order=res.get("p"),
                            R2=res.get("R2"),
                        )
                    )
                except Exception as e:
                    print(f"[EOC-h] Skip p={p}, s={s}, M={M}, dt={dt}: {e}")

# EOC–Δt: por (prectype, solver='par', M) a N=256 si existe (si no, al máximo N disponible)
for p in PREC_OK:
    for s in SOLVERS_EOC:
        for M in M_SET:
            g = df[(df["prectype"] == p) & (df["solver"] == s) & (df["nodes"] == M)]
            if g.empty:
                continue
            # elegir N más fino disponible (ideal: 256)
            Ns = sorted(g["n"].dropna().unique())
            if not Ns:
                continue
            Npick = 256 if 256 in Ns else Ns[-1]
            g2 = g[g["n"] == Npick]
            if g2["dt"].nunique() < 2:
                continue
            try:
                res = A.eoc_time(
                    fixed_n=int(Npick),
                    fixed_sw=int(M),
                    prectype=p,
                    solver=s,
                    nodes=M,
                    deg=1,
                )
                rows_eoc.append(
                    dict(
                        kind="time",
                        prectype=p,
                        solver=s,
                        nodes=M,
                        n=int(Npick),
                        p_order=res.get("p"),
                        R2=res.get("R2"),
                    )
                )
            except Exception as e:
                print(f"[EOC-dt] Skip p={p}, s={s}, M={M}, N={Npick}: {e}")

if rows_eoc:
    pd.DataFrame(rows_eoc).to_csv("figures/_eoc_orders_summary.csv", index=False)

# ====== CONTRACTION & ERROR vs k ======
# Solo en subconjunto representativo para no saturar
for p in PREC_OK:
    for s in SOLVERS_EOC:
        for M in REP["MS"]:
            A.sweeps_curves(
                prectype=p, solver=s, deg=1, nodes=M, n=REP["N"], dt=REP["DT"]
            )
            A.sweep_vs_collocation_errornorm(
                metric_candidates=["sweep_vs_collocation_error_norm"],
                figure_per_case=False,  # una figura por (prectype, solver)
                prectype=p,
                solver=s,
                deg=1,
                nodes=M,
                n=REP["N"],
                dt=REP["DT"],
            )

# ====== TIMINGS: boxplots por nodo y por sweep ======
# M=6 (peor caso) y M=3 (medio) en el caso representativo
for M in REP["MS"]:
    for p in PREC_OK:
        for s in SOLVERS_WP:
            A.timings_box_by_node_and_by_sweep(
                prectype=p,
                solver=s,
                deg=1,
                nodes=M,
                n=REP["N"],
                dt=REP["DT"],
                exclude_first_sweep=False,
                exclude_first_node=False,
            )

# ====== COLL walltime acumulado y comparativas por step ======
# (estos métodos guardan sus propias figuras)
try:
    A.full_coll_walltime(prectype=None)  # si quieres: filtra con prectype=p
except Exception as e:
    print("[full_coll_walltime] Skip:", e)
try:
    A.cumulative_walltime_vs_step()
except Exception as e:
    print("[cumulative_walltime_vs_step] Skip:", e)

# ====== SPEEDUPS y WORK–PRECISION ======
# Barras de speedup (ambos solvers)
try:
    A.speedup_bars(overlay=True)
except Exception as e:
    print("[speedup_bars] Skip:", e)

# Work–precision en rejilla, acotado a subconjunto razonable
A.work_precision_grid(
    legend_by="nodes",
    size_by="sw",  # burbujas por sw
    prectype=None,
    solver=None,
    deg=1,  # puedes poner prectype="MIN-SR-FLEX"
)

# ====== TABLAS RESUMEN (CSV) ======
try:
    sp = A.par_speedup()
    if not sp.empty:
        sp.to_csv("figures/_speedup_summary.csv", index=False)
except Exception as e:
    print("[par_speedup] Skip:", e)

print("[OK] heatequation5_analysis: figures in ./figures + CSV summaries.")
