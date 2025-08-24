import re
import fnmatch
import json
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Callable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.patches import Patch


@contextmanager
def pub_style(width_in=3.25, height_in=None, fontsize=8):
    """
    Context manager to apply a publication-grade Matplotlib style
    (single-figure layout, LaTeX-like fonts, tight spacing).
    width_in ~ 3.25in fits 1-column; use ~6.5in for 2-column.
    """
    if height_in is None:
        # Golden ratio-ish height for aesthetics
        height_in = width_in * 0.62
    old = plt.rcParams.copy()
    try:
        plt.rcParams.update(
            {
                "figure.figsize": (width_in, height_in),
                "figure.dpi": 300,
                "font.size": fontsize,
                "axes.labelsize": fontsize,
                "axes.titlesize": fontsize + 1,
                "xtick.labelsize": fontsize - 1,
                "ytick.labelsize": fontsize - 1,
                "legend.fontsize": fontsize - 1,
                "axes.linewidth": 1.0,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.minor.visible": True,
                "ytick.minor.visible": True,
                "grid.alpha": 0.25,
                "grid.linestyle": ":",
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.06,
                "figure.constrained_layout.use": True,
                "text.usetex": False,
                "font.family": "serif",
                "mathtext.fontset": "stix",
                "axes.titleweight": "semibold",
                # Publication-grade additions:
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "xtick.major.size": 5.0,
                "xtick.minor.size": 3.0,
                "ytick.major.size": 5.0,
                "ytick.minor.size": 3.0,
                "xtick.major.width": 0.8,
                "xtick.minor.width": 0.6,
                "ytick.major.width": 0.8,
                "ytick.minor.width": 0.6,
                "axes.edgecolor": (0, 0, 0, 0.85),
                "grid.color": (0, 0, 0, 0.35),
            }
        )
        yield
    finally:
        plt.rcParams.update(old)


# --- Small helper to stamp the method (prectype/solver) on every figure ---
from matplotlib.ticker import LogLocator


def _polish_axes(ax):
    """Final aesthetic touches: inward ticks, minor locators on log axes,
    and subtle top/right spines."""
    if ax is None:
        return
    # Ticks inward already set via rcParams; enforce widths and direction
    ax.tick_params(which="both", direction="in")
    # Minor locators for log axes
    try:
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_locator(LogLocator(subs=tuple(range(2, 10))))
        if ax.get_yscale() == "log":
            ax.yaxis.set_minor_locator(LogLocator(subs=tuple(range(2, 10))))
    except Exception:
        pass
    # Spines: keep left/bottom prominent; soften top/right
    try:
        ax.spines.get("top", None) and ax.spines["top"].set_alpha(0.25)
        ax.spines.get("right", None) and ax.spines["right"].set_alpha(0.25)
    except Exception:
        pass


def save_pub_figure(
    fig, stem: str, folder: str = "figures", dpi: int = 300, also_pdf: bool = True
):
    """
    Save a figure using the exact stem provided, without renaming/truncation.
    Writes PNG (and optionally PDF) into `folder`.
    """
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / stem
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    if also_pdf:
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)


def _fmt_float(x: float, sig: int = 3) -> str:
    """2.5e-03 -> '2.5e-3'; 0.001 -> '1e-3'; mantisa recortada."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if x == 0.0:
        return "0"
    s = f"{x:.{sig}e}"
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp = str(int(exp))  # sin ceros delante
    return f"{mant}e{exp}"


def _slugify(params: dict, order: list[str] | None = None) -> str:
    """Construye un stem estable con el orden canónico de parámetros."""
    if order is None:
        order = ["n", "dt", "sw", "nodes", "deg", "prectype", "solver", "tfinal", "idx"]
    parts = []
    for k in order:
        if k in params and params[k] is not None:
            v = params[k]
            if isinstance(v, float):
                parts.append(f"{k}{_fmt_float(v)}")
            elif isinstance(v, (int, np.integer)):
                parts.append(f"{k}{int(v)}")
            else:
                parts.append(f"{k}{str(v)}")
    for k, v in params.items():  # extras no canónicos
        if k not in order and v is not None:
            if isinstance(v, float):
                parts.append(f"{k}{_fmt_float(v)}")
            elif isinstance(v, (int, np.integer)):
                parts.append(f"{k}{int(v)}")
            else:
                parts.append(f"{k}{str(v)}")
    return "_".join(parts) if parts else "all"


def _slug_from_df(df: pd.DataFrame, keys: list[str] | None = None) -> str:
    """Construye un slug con todas las columnas de `keys` que sean unívocas en `df`."""
    if keys is None:
        keys = ["prectype", "solver", "deg", "nodes", "sw", "n", "dt"]
    params: dict[str, object] = {}
    for k in keys:
        if k in df.columns:
            u = df[k].dropna().unique()
            if len(u) == 1:
                v = u[0]
                if isinstance(v, (float, np.floating)):
                    params[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    params[k] = int(v)
                else:
                    params[k] = v
    return _slugify(params)


LOCK_KEYS_EOC_TIME = ["prectype", "solver", "deg", "nodes", "sw", "n", "tfinal"]
LOCK_KEYS_EOC_SPACE = ["prectype", "solver", "deg", "nodes", "sw", "dt", "tfinal"]
EOC_TIME_SWEEP_TOL = float("inf")
EOC_SPACE_SWEEP_TOL = float("inf")


def _inject_constant_column(df: pd.DataFrame, k: str, v) -> pd.DataFrame:
    """Ensure df has column k, with the constant value v."""
    if k not in df.columns:
        df = df.copy()
        df[k] = v
    return df


def _unique_combo_or_raise(df: pd.DataFrame, keys: list[str], ctx: str) -> dict:
    """
    Ensure df has exactly one unique combination over the given keys.
    Returns the unique constants as a dict. Raises if more than one.
    Uses exact equality (no np.isclose).
    """
    # work with columns (not index)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    cols = [k for k in keys if k in df.columns]
    if not cols:
        raise RuntimeError(f"[{ctx}] None of the locked keys are present: {keys}")

    uniq = df[cols].drop_duplicates()
    if len(uniq) != 1:
        # show a compact list of conflicting combos for quick debugging
        combos = uniq.to_dict(orient="records")
        raise RuntimeError(
            f"[{ctx}] Multiple locked-parameter combos found (must be exactly one). "
            f"Locked keys={cols}. Found {len(uniq)} combos, e.g.: {combos[:3]}"
        )
    return uniq.iloc[0].to_dict()


REQUIRED_STEM_KEYS = ("n", "dt", "sw", "nodes", "deg", "prectype", "solver")


def _full_slug_from_row(row: pd.Series, required=REQUIRED_STEM_KEYS) -> str:
    import pandas as pd, numpy as np

    missing = [k for k in required if (k not in row) or pd.isna(row[k])]
    if missing:
        jp = row.get("json_path", "?")
        raise RuntimeError(
            f"[stem] Faltan {missing} para {jp}. Revisa pattern/keys o el JSON."
        )
    params = {k: row[k] for k in required}
    return _slugify(params, order=list(required))


# --- Small helper to stamp the method (prectype/solver) on every figure ---
def _annotate_method(ax, meta: dict | None, fontsize: int = 6, alpha: float = 0.7):
    """Write a tiny method label (e.g. "MIN-SR-FLEX | par") in the
    lower-right corner of the axes. Silently no-ops if meta is None.
    """
    if ax is None or not isinstance(meta, dict):
        return
    parts = []
    for k in ["prectype", "solver"]:
        if k in meta and meta[k] is not None:
            parts.append(str(meta[k]))
    if parts:
        ax.text(
            0.99,
            0.02,
            " | ".join(parts),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=fontsize,
            alpha=alpha,
        )


def _annotate_params(ax, params: dict | None, fontsize: int = 6, alpha: float = 0.7):
    """Sello pequeño con parámetros (arriba‐izquierda)."""
    if ax is None or not isinstance(params, dict) or not params:
        return
    keys = ["n", "dt", "sw", "nodes", "deg", "prectype", "solver"]
    txts = []
    for k in keys:
        if k in params and params[k] is not None:
            v = params[k]
            if isinstance(v, float):
                txts.append(f"{k}={_fmt_float(v)}")
            elif isinstance(v, (int, np.integer)):
                txts.append(f"{k}={int(v)}")
            else:
                txts.append(f"{k}={v}")
    if txts:
        ax.text(
            0.01,
            0.98,
            ", ".join(txts),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
            alpha=alpha,
        )


# --- Canonicalize INFO keys to safe snake_case (lowercase) ---
def _canon_key(s: str) -> str:
    """Map raw INFO/metric names to a consistent snake_case key.
    Examples: 'Total residual sweep' -> 'total_residual_sweep',
    'norms.final.L2' -> 'norms_final_l2'.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()


# --- Helpers de métrica, CSV y etiquetas compactas ---


def _pick_error_col(df: pd.DataFrame, metric: str) -> str:
    """
    Selecciona la columna de error según 'metric' con degradado robusto.
    metric: "sweep" | "coll" | "final"
    """
    pref_map = {
        "sweep": [
            "err_L2_sweep",
            "sweep_vs_collocation_errornorm",
            "sweep_vs_real_errornorm",
        ],
        "coll": [
            "err_L2_coll",
            "collocation_vs_real_errornorm",
        ],
        "final": [
            "err_L2_final",
            "err_L2_sweep",
            "err_L2_coll",
        ],
    }
    prefs = pref_map.get(str(metric).lower(), ["err_L2_final"])
    for c in prefs:
        if c in df.columns:
            return c
    # fallback robusto
    return "err_L2_final" if "err_L2_final" in df.columns else prefs[0]


class JSONConvergenceAnalyser:
    """
    Light-weight analyser that reads metrics from the JSON sidecar files
    written by SDCSolver (…_convergence_results.json). It does not open
    any FireDrake checkpoint; everything is extracted from JSON.

    The filename is parsed with a regex (same philosophy as CheckpointAnalyser):
    you pass a `pattern` with named groups and the corresponding `keys` and
    `keys_type` to cast those groups. Typical keys:
        keys = ["n", "dt", "sw", "nodes", "deg", "prectype", "solver", "tfinal", "idx"]
        keys_type = [int, float, int, int, int, str, str, float, int]
    (the last group 'idx' can be omitted from `keys` if you don't want it in the index).
    """

    def __init__(
        self,
        file_path: Path,
        pattern: re.Pattern,
        keys: str | List[str],
        keys_type: callable | List[callable],
        *,
        crit_first_n: int | None = None,
        use_truncated_crit_in_total: bool = False,
    ):
        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.pattern = (
            pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
        )
        self.keys = [keys] if not isinstance(keys, list) else keys
        self.keys_type: List[Callable] = (
            keys_type if isinstance(keys_type, list) else [keys_type]
        )
        self.crit_first_n = crit_first_n
        self.use_truncated_crit_in_total = use_truncated_crit_in_total
        self.json_list: Dict[tuple, tuple] = self._list_json_results()
        if not self.json_list:
            raise Exception("No JSON sidecars ('*_convergence_results.json') found.")
        self.df = self._build_master_table()
        self._diagnose_missing_stem_fields()

    def _build_exclude_predicates(
        self, exclude=None, exclude_globs=None, exclude_regex=None
    ):
        preds = []
        if exclude:
            preds.append(lambda p: any(s in str(p) for s in exclude))
        if exclude_globs:
            preds.append(
                lambda p: any(fnmatch.fnmatch(str(p), g) for g in exclude_globs)
            )
        if exclude_regex:
            regs = [re.compile(r) for r in exclude_regex]
            preds.append(lambda p: any(r.search(str(p)) for r in regs))
        return preds

    def _exclude_df(self, df, *, exclude=None, exclude_globs=None, exclude_regex=None):
        if df.empty or "json_path" not in df.columns:
            return df
        preds = self._build_exclude_predicates(exclude, exclude_globs, exclude_regex)
        if not preds:
            return df
        mask = ~df["json_path"].apply(lambda s: any(pred(Path(s)) for pred in preds))
        return df.loc[mask].copy()

    def _diagnose_missing_stem_fields(self):
        import pandas as pd

        req = list(REQUIRED_STEM_KEYS)
        df = self.df.reset_index().copy()
        miss_cols = [k for k in req if k not in df.columns]
        if miss_cols:
            raise ValueError(
                f"[init] Faltan columnas {miss_cols} en self.df. "
                "Asegúrate de que `keys` y el `pattern` las capturan."
            )
        bad = df[df[req].isna().any(axis=1)]
        if not bad.empty:
            cols = ["json_path"] + req
            print(
                "[warn] Filas con NaN en campos de stem:\n"
                + bad[cols].to_string(index=False)
            )

    # ---------- discovery ----------
    def _list_json_results(self) -> dict:
        """
        Mirror of CheckpointAnalyser.list_json_results but self-contained.
        Maps typed key tuple -> (json_path, idx).
        """

        def _cast(typ, txt):
            if typ is float:
                txt = txt.replace("p", ".")
                return float(txt)
            if typ is int:
                return int(txt)
            return txt  # strings (prectype, solver) intactos

        result = {}
        for file in self.file_path.rglob("*_convergence_results.json"):
            base = file.name[: -len("_convergence_results.json")]
            m = self.pattern.match(base)
            if not m:
                continue
            key = tuple(
                _cast(t, m.group(name)) for t, name in zip(self.keys_type, self.keys)
            )
            idx = (
                int(m["idx"]) if "idx" in m.groupdict() and m["idx"] is not None else 0
            )
            if key not in result or idx > result[key][1]:
                result[key] = (file, idx)
        return result

    # ---------- parsing helpers ----------
    @staticmethod
    def _info_map(payload: dict, values: list[float]) -> Dict[str, float]:
        """Map payload['INFO'] to values with **canonical** snake_case keys.
        Tolerates length mismatches by truncation.
        """
        names = payload.get("INFO", [])
        names = names[: len(values)]
        values = values[: len(names)]
        return {_canon_key(names[i]): values[i] for i in range(len(names))}

    @staticmethod
    def _triples(payload: dict) -> list[tuple[int, float, int, str]]:
        """
        Collect (step:int, time:float, k:int, key_string) from keys like 'step,time,k'.
        """
        out = []
        for k in payload.keys():
            if "," in k and k.count(",") == 2 and not k.endswith("_timings"):
                s, t, kk = k.split(",")
                try:
                    out.append((int(s), float(t), int(kk), k))
                except Exception:
                    pass
        return out

    @staticmethod
    def _contractions_for_step(payload: dict, step: int) -> list[dict]:
        """
        Return a list of contraction dicts for a given step: keys '... ,contraction'.
        """
        out = []
        for k, v in payload.items():
            if (
                k.startswith(f"{step},")
                and k.endswith(",contraction")
                and isinstance(v, dict)
            ):
                out.append(v)
        return out

    @staticmethod
    def _timings_blocks(payload: dict) -> list[list[dict]]:
        """
        Collect all lists of timing rows from keys ending with '_timings'.
        """
        blocks = []
        for k, v in payload.items():
            if k.endswith("_timings") and isinstance(v, list):
                blocks.append(v)
        return blocks

    @staticmethod
    def _full_coll_times(payload: dict) -> list[float]:
        """Collect wall_times from any '*full_collocation_timing' keys (per-step)."""
        times: list[float] = []
        for k, v in payload.items():
            if isinstance(v, list) and str(k).endswith("full_collocation_timing"):
                for row in v:
                    try:
                        times.append(float(row.get("wall_time", 0.0)))
                    except Exception:
                        pass
        return times

    @staticmethod
    def _geom_mean_tail(
        seq: list[float], frac: float = 0.5, min_tail: int = 3
    ) -> float:
        if not seq:
            return float("nan")
        m = max(min_tail, int(np.ceil(frac * len(seq))))
        tail = [max(float(x), 1e-300) for x in seq[-m:]]
        return float(np.exp(np.mean(np.log(tail))))

    # ---------- table builder ----------
    def _build_master_table(self) -> pd.DataFrame:
        rows = []
        for key_tuple, (json_path, idx) in self.json_list.items():
            with open(str(json_path), "r") as f:
                payload = json.load(f)

            triples = self._triples(payload)
            if not triples:
                continue
            # Pick last step, and within it the largest k entry
            last_step = max(s for (s, _, _, _) in triples)
            # k máximo en el último step
            k_last = max(k for (s, _, k, _) in triples if s == last_step)
            # entre los registros con ese k, tomar el de t máximo (fin de paso)
            cands = [
                (t, key_str)
                for (s, t, k, key_str) in triples
                if (s == last_step and k == k_last)
            ]
            t_last, key_str = max(cands, key=lambda z: z[0])

            values = payload.get(key_str, [])
            metrics = self._info_map(payload, values)

            # Tolerant extraction of final L2 error using canonical keys
            # --- errores disponibles: separa lo que mide cada cosa ---
            err_sweep_real = metrics.get(
                "sweep_vs_real_error_norm", metrics.get("sweep_vs_real_errornorm")
            )
            err_coll_real = metrics.get(
                "collocation_vs_real_error_norm",
                metrics.get("collocation_vs_real_errornorm"),
            )
            err_sweep_coll = metrics.get(
                "sweep_vs_collocation_error_norm",
                metrics.get("sweep_vs_collocation_errornorm"),
            )

            # fallback para JSON antiguos → lo más cercano a "sweep vs real"
            if err_sweep_real is None:
                for alt in ("final_l2", "norms_final_l2", "error_l2_final"):
                    if alt in metrics and metrics[alt] is not None:
                        try:
                            err_sweep_real = float(metrics[alt])
                        except Exception:
                            pass
                        break
            # Contraction info (rho_seq, delta_seq) for tail average
            contr_blocks = self._contractions_for_step(payload, last_step)
            rho_inf = float("nan")
            delta_last = float("nan")
            if contr_blocks:
                c = contr_blocks[-1]
                rho_seq = c.get("rho_seq") or []
                delta_seq = c.get("delta_seq") or []
                rho_inf = self._geom_mean_tail([float(x) for x in rho_seq])
                if delta_seq:
                    delta_last = float(delta_seq[-1])

            # --- Wall-clock aggregation: sequential sum vs. parallel critical path ---
            # --- Wall-clock aggregation: sequential sum vs. parallel critical path ---
            t_total_seq = 0.0  # suma secuencial de TODO
            t_sweep_rows = []  # estadística por subsolver (para medias)

            t_total_crit = 0.0  # ruta crítica clásica (max dentro de cada bloque)
            t_total_crit_firstn = 0.0  # NEW: ruta crítica truncada a primeros n
            crit_rows = []  # para medias/max clásicas
            crit_rows_firstn = []  # NEW: para medias/max truncadas

            crit_n = self.crit_first_n  # NEW

            for blk in self._timings_blocks(payload):
                # recolecta (solver_index, wall_time) preservando orden
                rows_si_w: list[tuple[int | None, float]] = []
                blk_max_all = 0.0

                for r in blk:
                    try:
                        w = float(r.get("wall_time", 0.0))
                    except Exception:
                        continue

                    t_total_seq += w

                    si = r.get("solver_index")
                    if isinstance(si, (int, np.integer)):
                        si = int(si)
                        t_sweep_rows.append(w)
                    else:
                        si = None
                    rows_si_w.append((si, w))

                    if w > blk_max_all:
                        blk_max_all = w

                # clásico
                t_total_crit += blk_max_all
                if blk_max_all > 0.0:
                    crit_rows.append(blk_max_all)

                # NEW: máximo sólo entre los "primeros n"
                blk_max_firstn = 0.0
                if crit_n is not None and crit_n > 0 and rows_si_w:
                    # Preferimos filtrar por solver_index si está disponible
                    if any(si is not None for si, _ in rows_si_w):
                        # candidatos: si < crit_n
                        cand = [
                            w
                            for (si, w) in rows_si_w
                            if (si is not None and si < crit_n)
                        ]
                        if not cand:
                            # fallback: toma los de menor solver_index (ordenado) hasta n
                            sorted_by_si = sorted(
                                [(si, w) for si, w in rows_si_w if si is not None],
                                key=lambda z: z[0],
                            )[:crit_n]
                            cand = [w for _, w in sorted_by_si]
                    else:
                        # si no hay solver_index, usa el orden de inserción (primeros n)
                        cand = [w for _, w in rows_si_w[:crit_n]]

                    if cand:
                        blk_max_firstn = float(np.max(cand))

                else:
                    blk_max_firstn = blk_max_all  # sin truncado → igual que clásico

                t_total_crit_firstn += blk_max_firstn
                if blk_max_firstn > 0.0:
                    crit_rows_firstn.append(blk_max_firstn)

            t_sweep_mean = (
                float(np.mean(t_sweep_rows)) if t_sweep_rows else float("nan")
            )
            t_sweep_max = float(np.max(t_sweep_rows)) if t_sweep_rows else float("nan")
            t_sweep_crit_mean = float(np.mean(crit_rows)) if crit_rows else float("nan")
            t_sweep_crit_max = float(np.max(crit_rows)) if crit_rows else float("nan")

            # NEW: estadísticas de la versión truncada
            t_sweep_crit_firstn_mean = (
                float(np.mean(crit_rows_firstn)) if crit_rows_firstn else float("nan")
            )
            t_sweep_crit_firstn_max = (
                float(np.max(crit_rows_firstn)) if crit_rows_firstn else float("nan")
            )

            # Full collocation timing (if present in JSON)
            fc_times = self._full_coll_times(payload)
            fc_total = float(np.sum(fc_times)) if fc_times else float("nan")
            fc_mean = float(np.mean(fc_times)) if fc_times else float("nan")
            fc_max = float(np.max(fc_times)) if fc_times else float("nan")

            # Decode filename parameters (typed) from key_tuple
            rec = {name: val for name, val in zip(self.keys, key_tuple)}
            solver_tag = str(rec.get("solver", "")).lower()

            rec.update(
                {
                    "idx": idx,
                    "json_path": str(json_path),
                    "t_last": float(t_last),
                    "k_last": int(k_last),
                    "err_L2_sweep": (
                        float(err_sweep_real)
                        if err_sweep_real is not None
                        else float("nan")
                    ),  # ||u^(K) - u_real||
                    "err_L2_coll": (
                        float(err_coll_real)
                        if err_coll_real is not None
                        else float("nan")
                    ),  # ||u_coll - u_real||
                    "err_sweep_coll": (
                        float(err_sweep_coll)
                        if err_sweep_coll is not None
                        else float("nan")
                    ),  # ||u^(K) - u_coll||
                    # Alias histórico: por defecto lo que ENTREGA el método tras K sweeps
                    "err_L2_final": (
                        float(err_sweep_real)
                        if err_sweep_real is not None
                        else (
                            float(err_coll_real)
                            if err_coll_real is not None
                            else float("nan")
                        )
                    ),
                    "rho_inf": rho_inf,
                    "delta_last": delta_last,
                    # NUEVO: ambas nociones de tiempo total
                    "wall_time_total_seq": float(t_total_seq),
                    "wall_time_total_crit": float(t_total_crit),
                    "wall_time_total_crit_firstN": (
                        float(t_total_crit_firstn)
                        if self.crit_first_n
                        else float("nan")
                    ),  # NEW
                    "wall_time_total": float(
                        # Si solver 'par', puedes elegir usar el truncado para el total
                        (
                            t_total_crit_firstn
                            if (
                                solver_tag == "par"
                                and self.crit_first_n
                                and self.use_truncated_crit_in_total
                            )
                            else (t_total_crit if solver_tag == "par" else t_total_seq)
                        )
                    ),
                    "t_per_sweep_mean": t_sweep_mean,
                    "t_per_sweep_max": t_sweep_max,
                    "t_per_sweep_crit_mean": t_sweep_crit_mean,
                    "t_per_sweep_crit_max": t_sweep_crit_max,
                    "t_per_sweep_crit_firstN_mean": t_sweep_crit_firstn_mean,
                    "t_per_sweep_crit_firstN_max": t_sweep_crit_firstn_max,
                    "full_coll_walltime_total": fc_total,
                    "full_coll_walltime_mean": fc_mean,
                    "full_coll_walltime_max": fc_max,
                }
            )
            # convenience derived vars
            if "n" in rec and isinstance(rec["n"], (int, float)):
                rec["h"] = 1.0 / float(rec["n"])
            # Fourier number (dimensionless) assuming kappa=1 unless specified elsewhere
            if (
                "dt" in rec
                and "n" in rec
                and isinstance(rec["dt"], (int, float))
                and isinstance(rec["n"], (int, float))
            ):
                rec["Fo"] = float(rec["dt"]) * float(rec["n"]) ** 2
            rows.append(rec)

        df = pd.DataFrame.from_records(rows)
        # Build a stable MultiIndex if possible
        index_cols = [c for c in self.keys if c in df.columns]
        if index_cols:
            df = df.set_index(index_cols).sort_index()
        return df

    def sweep_metrics_long(self) -> pd.DataFrame:
        """
        Long table with one row per (json_path, step, t, k) carrying all INFO metrics.
        Columns include the decoded filename params and the INFO-mapped metrics.
        """
        records: list[dict] = []
        for key_tuple, (json_path, idx) in self.json_list.items():
            with open(str(json_path), "r") as f:
                payload = json.load(f)
            triples = self._triples(payload)
            if not triples:
                continue
            for step, t, k, key_str in sorted(triples, key=lambda z: (z[0], z[2])):
                values = payload.get(key_str, [])
                metrics = self._info_map(payload, values)
                rec = {name: val for name, val in zip(self.keys, key_tuple)}
                rec.update(
                    {
                        "idx": idx,
                        "json_path": str(json_path),
                        "step": int(step),
                        "t": float(t),
                        "k": int(k),
                    }
                )
                # attach metrics with safe names
                for mkey, mval in metrics.items():
                    rec[_canon_key(mkey)] = float(mval) if mval is not None else np.nan
                records.append(rec)
        df = pd.DataFrame.from_records(records)
        if not df.empty:
            # Build a stable index if possible
            index_cols = [c for c in self.keys if c in df.columns] + ["step", "k"]
            df = df.set_index(index_cols).sort_index()
        return df

    def contractions_long(self, explode: bool = True) -> pd.DataFrame:
        """
        Long table with contractions.
        If explode=True, returns one row per sweep index r with columns delta_r and rho_r
        for each (json_path, step).
        If explode=False, stores the whole sequences in list-columns.
        """
        rows: list[dict] = []
        for key_tuple, (json_path, idx) in self.json_list.items():
            with open(str(json_path), "r") as f:
                payload = json.load(f)
            # collect all steps that have a contraction block
            steps = []
            for k, v in payload.items():
                if k.endswith(",contraction") and isinstance(v, dict):
                    try:
                        step = int(k.split(",")[0])
                        steps.append((step, v))
                    except Exception:
                        pass
            if not steps:
                continue
            for step, block in sorted(steps, key=lambda z: z[0]):
                rho_seq = block.get("rho_seq") or []
                delta_seq = block.get("delta_seq") or []
                delta_last = (
                    float(block.get("delta_last"))
                    if block.get("delta_last") is not None
                    else (float(delta_seq[-1]) if delta_seq else np.nan)
                )
                rec_base = {name: val for name, val in zip(self.keys, key_tuple)}
                rec_base.update(
                    {
                        "idx": idx,
                        "json_path": str(json_path),
                        "step": int(step),
                        "delta_last": float(delta_last),
                    }
                )
                if explode and delta_seq:
                    for r, (d, rho) in enumerate(
                        zip(delta_seq, [np.nan] + list(rho_seq)), start=1
                    ):
                        rec = rec_base.copy()
                        rec.update(
                            {
                                "r": int(r),
                                "delta_r": float(d),
                                "rho_r": (float(rho) if rho is not None else np.nan),
                            }
                        )
                        rows.append(rec)
                else:
                    rec = rec_base.copy()
                    rec.update(
                        {
                            "delta_seq": list(map(float, delta_seq)),
                            "rho_seq": list(map(float, rho_seq)),
                        }
                    )
                    rows.append(rec)
        df = pd.DataFrame.from_records(rows)
        if not df.empty:
            if "r" in df.columns:
                index_cols = [c for c in self.keys if c in df.columns] + ["step", "r"]
            else:
                index_cols = [c for c in self.keys if c in df.columns] + ["step"]
            df = df.set_index(index_cols).sort_index()
        return df

    def timings_long(self) -> pd.DataFrame:
        """
        Long table with timing rows. One row per timing entry with decoded (step, t, k)
        and the per-subsolver metadata carried in each timing row.
        """
        rows: list[dict] = []
        for key_tuple, (json_path, idx) in self.json_list.items():
            with open(str(json_path), "r") as f:
                payload = json.load(f)
            for key, val in payload.items():
                if not (isinstance(val, list) and key.endswith("_timings")):
                    continue
                try:
                    s_str, t_str, k_part = key.split(",")
                    step = int(s_str)
                    t = float(t_str)
                    k = int(k_part.split("_")[0])
                except Exception:
                    continue
                for row in val:
                    rec = {name: val for name, val in zip(self.keys, key_tuple)}
                    rec.update(
                        {
                            "idx": idx,
                            "json_path": str(json_path),
                            "step": step,
                            "t": t,
                            "k": k,
                        }
                    )
                    # copy through all simple fields from timing row
                    for fld in (
                        "solver_index",
                        "wall_time",
                        "comp",
                        "node",
                        "flat_idx",
                        "lenV",
                        "global",
                    ):
                        if fld in row:
                            rec[fld] = row[fld]
                    # enforce numeric type if plausible
                    if "wall_time" in rec:
                        rec["wall_time"] = float(rec["wall_time"])  # type: ignore
                    if "solver_index" in rec and isinstance(
                        rec["solver_index"], (int, np.integer)
                    ):
                        rec["solver_index"] = int(rec["solver_index"])  # type: ignore
                    rows.append(rec)
        df = pd.DataFrame.from_records(rows)
        if not df.empty:
            index_cols = [c for c in self.keys if c in df.columns] + [
                "step",
                "k",
                "solver_index",
            ]
            df = df.set_index(index_cols).sort_index()
        return df

    # ---------- simple EOC utilities ----------
    @staticmethod
    def _filter_eq(df: pd.DataFrame, key: str, value):
        """Return a filtered copy of df where column/level `key` equals `value`.
        If `key` is a level in a MultiIndex, reset_index first so it becomes a column.
        Float comparisons are done with np.isclose to avoid roundoff issues.
        """
        if isinstance(df.index, pd.MultiIndex) and key in df.index.names:
            df = df.reset_index()
        elif key not in df.columns:
            return df
        if isinstance(value, float):
            m = np.isclose(df[key].astype(float), float(value), rtol=1e-12, atol=1e-15)
            return df.loc[m]
        else:
            return df.loc[df[key] == value]

    @staticmethod
    def _fit_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        """
        Fit y = C x^p in log–log; return (p, logC, R^2).
        """
        xm = np.log(x)
        ym = np.log(y)
        m, b = np.polyfit(xm, ym, 1)
        yhat = m * xm + b
        ss_res = float(np.sum((ym - yhat) ** 2))
        ss_tot = float(np.sum((ym - np.mean(ym)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(m), float(b), float(r2)

    def eoc_space(
        self,
        fixed_dt: float,
        fixed_sw: int,
        *,
        metric: str = "coll",
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> Dict[str, float]:
        """
        EOC vs h con matching estricto: TODO fijo salvo N (h=1/N varía).
        Además, QC: exige que ||u^(K)-u_coll|| << ||u_coll-u_real|| por punto.
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )

        # Igualdad estricta de parámetros fijos
        df2 = self._filter_eq(df2, "dt", float(fixed_dt))
        df2 = self._filter_eq(df2, "sw", int(fixed_sw))
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)

        # Asegura h
        if "h" not in df2.columns and "n" in df2.columns:
            df2["h"] = 1.0 / df2["n"].astype(float)

        # Inyecta constantes por si no venían
        df_meta = df2.copy()
        df_meta = _inject_constant_column(df_meta, "dt", float(fixed_dt))
        df_meta = _inject_constant_column(df_meta, "sw", int(fixed_sw))

        # Matching ESTRICTO: todo igual salvo n (h)
        consts = _unique_combo_or_raise(
            df_meta,
            [k for k in LOCK_KEYS_EOC_SPACE if k != "n"],  # aquí varía N (h)
            ctx="eoc_space",
        )

        # Columna de error elegida
        col = _pick_error_col(df2, metric)
        df_clean = df2.replace([np.inf, -np.inf], np.nan).dropna(subset=["h", col])

        # Curva principal: mediana por h
        df_plot = (
            df_clean.groupby("h", as_index=False).agg({col: "median"}).sort_values("h")
        )
        x = df_plot["h"].astype(float).to_numpy()
        y = df_plot[col].astype(float).to_numpy()
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # QC: ratio = ||u^(K)-u_coll|| / ||u_coll-u_real||
        coll_col = "err_L2_coll"
        sweepcoll_col = "err_sweep_coll"
        have_qc = (coll_col in df_clean.columns) and (sweepcoll_col in df_clean.columns)
        if have_qc:
            df_qc = (
                df_clean.groupby("h", as_index=False)
                .agg({coll_col: "median", sweepcoll_col: "median"})
                .sort_values("h")
            )
            df_qc["ratio"] = df_qc[sweepcoll_col] / df_qc[coll_col]
            df_qc["ratio"] = df_qc["ratio"].replace([np.inf, -np.inf], np.nan)
            bad_mask = (df_qc["ratio"] > EOC_SPACE_SWEEP_TOL) & np.isfinite(
                df_qc["ratio"]
            )
        else:
            df_qc, bad_mask = None, None

        orders: Dict[str, float] = {"p": float("nan"), "R2": float("nan")}
        if x.size < 2:
            print("[eoc_space] No positive data after filtering; skipping plot.")
            return orders

        p, logC, r2 = self._fit_loglog(x, y)
        orders["p"], orders["R2"] = p, r2

        with pub_style(width_in=3.25, fontsize=9):
            fig, ax = plt.subplots()
            ax.scatter(x, y, s=10, marker="^", c="red", label="data")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("h = 1/N")
            ax.set_ylabel(r"$\|e\|_{L^2}$")
            ax.plot(
                x,
                np.exp(logC) * x**p,
                "--",
                marker="x",
                label=f"fit: p≈{p:.2f}, R²≈{r2:.3f}",
                linewidth=0.8,
                alpha=0.8,
            )

            # Overlay de puntos con ratio alto (si hay QC)
            if have_qc and not df_qc.empty and bad_mask is not None and bad_mask.any():
                bad_h = df_qc.loc[bad_mask, "h"].astype(float).to_numpy()
                tol = 1e-14
                bad_points = np.array(
                    [any(np.isclose(xx, bad_h, rtol=1e-12, atol=tol)) for xx in x]
                )
                if bad_points.any():
                    ax.scatter(
                        x[bad_points],
                        y[bad_points],
                        s=28,
                        marker="o",
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.9,
                        label=f"ratio>{EOC_SPACE_SWEEP_TOL:.2f}",
                    )
                    for h_val, ratio_val in zip(
                        df_qc.loc[bad_mask, "h"], df_qc.loc[bad_mask, "ratio"]
                    ):
                        print(
                            f"[eoc_space][warn] h={_fmt_float(float(h_val))}: "
                            f"||u^(K)-u_coll|| / ||u_coll-u_real|| = {float(ratio_val):.3e} "
                            f"> {EOC_SPACE_SWEEP_TOL:.2f}. Aumenta sweeps o usa métrica 'coll'."
                        )

            ax.legend(frameon=False)
            ax.grid(True, which="both", alpha=0.4)
            ax.xaxis.labelpad = 6
            _polish_axes(ax)

            metric_tag = {"sweep": "sweep", "coll": "coll", "final": "final"}.get(
                str(metric).lower(), str(metric)
            )

            # slug solo con constantes (N/h varía → NO incluir 'n' ni 'h')
            slug = _slugify(
                consts,
                order=["prectype", "solver", "deg", "nodes", "sw", "dt", "tfinal"],
            )

            try:
                import hashlib

                H = hashlib.sha1(
                    np.ascontiguousarray(np.vstack([x, y])).tobytes()
                ).hexdigest()[:8]
            except Exception:
                H = "nohash"

            stem = f"eoc_space_{slug}_m{metric_tag}_H{H}"
            save_pub_figure(fig, stem=stem, folder="figures", dpi=300, also_pdf=True)
            plt.close(fig)

        return orders

    def eoc_time(
        self,
        fixed_n: int,
        fixed_sw: int,
        *,
        metric: str = "coll",
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> Dict[str, float]:
        """
        EOC vs dt con matching estricto: TODO fijo salvo dt.
        Además, quality check: exige que ||u^(K)-u_coll|| << ||u_coll-u_real|| por punto.
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )

        # Filtro exacto de N y sweeps fijos
        df2 = self._filter_eq(df2, "n", int(fixed_n))
        df2 = self._filter_eq(df2, "sw", int(fixed_sw))
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)

        df2 = df2.sort_values("dt")

        err_col = _pick_error_col(df2, metric)
        if err_col not in df2.columns:
            print("[eoc_time] Required error column missing; skipping plot.")
            return {"p": float("nan"), "R2": float("nan")}

        # Inyecta columnas fijas por si no vienen del patrón
        df_meta = df2.copy()
        df_meta = _inject_constant_column(df_meta, "n", int(fixed_n))
        df_meta = _inject_constant_column(df_meta, "sw", int(fixed_sw))

        # Matching ESTRICTO: todo igual salvo dt
        consts = _unique_combo_or_raise(
            df_meta,
            [k for k in LOCK_KEYS_EOC_TIME if k != "dt"],  # dt es la que varía
            ctx="eoc_time",
        )

        # Limpieza y deduplicación por dt para la curva principal (usamos mediana si hay repetidos)
        df_clean = df2.replace([np.inf, -np.inf], np.nan).dropna(subset=["dt", err_col])

        # --- QC: necesitamos también ambas métricas para la comprobación de barrido vs collocation
        # columnas “canónicas” que construye _build_master_table:
        coll_col = "err_L2_coll"  # ≈ ||u_coll - u_real||
        sweepcoll_col = "err_sweep_coll"  # ≈ ||u^(K) - u_coll||
        have_qc = (coll_col in df_clean.columns) and (sweepcoll_col in df_clean.columns)

        # Data para la curva principal (error elegido vs dt)
        df_plot = (
            df_clean.groupby("dt", as_index=False)
            .agg({err_col: "median"})
            .sort_values("dt")
        )

        # Data para QC (ratios por dt)
        if have_qc:
            df_qc = (
                df_clean.groupby("dt", as_index=False)
                .agg({coll_col: "median", sweepcoll_col: "median"})
                .sort_values("dt")
            )
            # ratio = ||u^(K)-u_coll|| / ||u_coll-u_real||
            df_qc["ratio"] = df_qc[sweepcoll_col] / df_qc[coll_col]
            # Puede haber división por ~0 si el error de collocation es ínfimo
            df_qc["ratio"] = df_qc["ratio"].replace([np.inf, -np.inf], np.nan)
            bad_mask = (df_qc["ratio"] > EOC_TIME_SWEEP_TOL) & np.isfinite(
                df_qc["ratio"]
            )
        else:
            df_qc = None
            bad_mask = None

        x = df_plot["dt"].astype(float).to_numpy()
        y = df_plot[err_col].astype(float).to_numpy()
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        orders = {"p": float("nan"), "R2": float("nan")}
        if x.size < 2:
            print("[eoc_time] No positive data after filtering; skipping plot.")
            return orders

        p, logC, r2 = self._fit_loglog(x, y)
        orders["p"], orders["R2"] = p, r2

        with pub_style(width_in=3.25, fontsize=9):
            fig, ax = plt.subplots()
            ax.scatter(x, y, s=10, marker="^", label="data")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$\Delta t$")
            ax.set_ylabel(r"$\|e\|_{L^2}$")
            ax.plot(
                x,
                np.exp(logC) * x**p,
                "--",
                marker="x",
                label=f"fit: p≈{p:.2f}, R²≈{r2:.3f}",
                linewidth=1.0,
                alpha=0.8,
            )

            # --- Overlay de puntos “no convergidos” en rojo (si hay QC disponible) ---
            if have_qc and not df_qc.empty and bad_mask is not None and bad_mask.any():
                # Alinea con dt de la curva principal y con el 'mask' positivo
                bad_dt = df_qc.loc[bad_mask, "dt"].astype(float).to_numpy()
                # selecciona los x que están en bad_dt (con tolerancia numérica)
                tol = 1e-14
                bad_points = np.array(
                    [any(np.isclose(xx, bad_dt, rtol=1e-12, atol=tol)) for xx in x]
                )
                if bad_points.any():
                    ax.scatter(
                        x[bad_points],
                        y[bad_points],
                        s=28,
                        marker="o",
                        facecolors="none",
                        edgecolors="red",
                        linewidths=0.9,
                        label=f"ratio>{EOC_TIME_SWEEP_TOL:.2f}",
                    )
                    # Mensajes de diagnóstico con ratios
                    for dt_val, ratio_val in zip(
                        df_qc.loc[bad_mask, "dt"], df_qc.loc[bad_mask, "ratio"]
                    ):
                        print(
                            f"[eoc_time][warn] dt={_fmt_float(float(dt_val))}: "
                            f"||u^(K)-u_coll|| / ||u_coll-u_real|| = {float(ratio_val):.3e} "
                            f"> {EOC_TIME_SWEEP_TOL:.2f}. Aumenta sweeps o usa métrica 'coll'."
                        )

            ax.legend(frameon=False)
            ax.grid(True, which="both", alpha=0.4)
            ax.xaxis.labelpad = 6
            _polish_axes(ax)

            metric_tag = {"sweep": "sweep", "coll": "coll", "final": "final"}.get(
                str(metric).lower(), str(metric)
            )

            # slug solo con constantes (dt varía)
            slug = _slugify(
                consts,
                order=["prectype", "solver", "deg", "nodes", "sw", "n", "tfinal"],
            )

            try:
                import hashlib

                H = hashlib.sha1(
                    np.ascontiguousarray(np.vstack([x, y])).tobytes()
                ).hexdigest()[:8]
            except Exception:
                H = "nohash"

            stem = f"eoc_time_{slug}_m{metric_tag}_H{H}"
            save_pub_figure(fig, stem=stem, folder="figures", dpi=300, also_pdf=True)
            plt.close(fig)

        return orders

    def par_speedup(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        use_truncated_crit: bool = False,
    ) -> pd.DataFrame:
        """
        Empareja (par vs global) con idénticos (prectype, n, dt, nodes, deg, sw) y calcula:
        - speedup_vs_global = T_glob / T_par
        - speedup_vs_coll   = T_coll / T_par
        con:
        T_par  = suma de máximos por (step, k) en 'par' (ruta crítica SDC)
        T_glob = suma total de wall_time en 'global' (modelo secuencial)
        T_coll = suma de 'full_collocation_timing' por paso (si existe)
        """
        df = self.df.reset_index()
        df = self._exclude_df(
            df,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )

        # Asegura columnas esperadas
        needed = {
            "prectype",
            "n",
            "dt",
            "nodes",
            "deg",
            "sw",
            "solver",
            "wall_time_total_seq",
            "wall_time_total_crit",
            "err_L2_final",
            "full_coll_walltime_total",
        }
        for c in needed:
            if c not in df.columns:
                if isinstance(df.index, pd.MultiIndex) and c in df.index.names:
                    df[c] = df.index.get_level_values(c)
                else:
                    df[c] = np.nan

        grp_keys = ["prectype", "n", "dt", "nodes", "deg", "sw"]
        rows = []

        for gvals, gdf in df.groupby(grp_keys, dropna=False):
            # Emparejamos EXACTAMENTE 'par' con 'global'
            g_par = gdf[gdf["solver"].astype(str).str.lower() == "par"]
            g_glob = gdf[gdf["solver"].astype(str).str.lower() == "global"]
            if g_par.empty or g_glob.empty:
                continue

            T_par = float(
                g_par["wall_time_total_crit_firstN"].iloc[0]
                if (
                    use_truncated_crit
                    and "wall_time_total_crit_firstN" in g_par.columns
                    and np.isfinite(g_par["wall_time_total_crit_firstN"].iloc[0])
                    and g_par["wall_time_total_crit_firstN"].iloc[0] > 0
                )
                else g_par["wall_time_total_crit"].iloc[0]
            )
            T_glob = float(g_glob["wall_time_total_seq"].iloc[0])

            # Collocation total (coge valor válido de par o global)
            def _get_coll(dfsub):
                try:
                    val = float(dfsub["full_coll_walltime_total"].iloc[0])
                    return val if np.isfinite(val) and val > 0 else np.nan
                except Exception:
                    return np.nan

            T_coll = _get_coll(g_par)
            if not np.isfinite(T_coll):
                T_coll = _get_coll(g_glob)

            E_par = (
                float(g_par["err_L2_final"].iloc[0])
                if "err_L2_final" in g_par
                else np.nan
            )
            E_glob = (
                float(g_glob["err_L2_final"].iloc[0])
                if "err_L2_final" in g_glob
                else np.nan
            )

            rows.append(
                dict(
                    zip(grp_keys, gvals),
                    # --- Nuevos nombres claros ---
                    speedup_vs_global=(T_glob / T_par if T_par > 0 else np.nan),
                    speedup_vs_coll=(
                        T_coll / T_par
                        if (T_par > 0 and np.isfinite(T_coll))
                        else np.nan
                    ),
                    # --- Alias para compatibilidad (antes 'speedup' apuntaba a vs "seq");
                    # ahora lo dejamos como vs_global para no romper código existente.
                    speedup=(T_glob / T_par if T_par > 0 else np.nan),
                    T_par=T_par,
                    T_glob=T_glob,
                    T_coll=T_coll,
                    E_par=E_par,
                    E_glob=E_glob,
                )
            )

        return pd.DataFrame(rows).sort_values(grp_keys)

    def work_precision(
        self,
        facet_by: tuple[str, str] = ("prectype", "solver"),
        legend_by: str = "nodes",
        size_by: str | None = "sw",
        alpha: float = 0.9,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> None:
        df0 = self.df.reset_index()
        for k, v in filters.items():
            df0 = self._filter_eq(df0, k, v)

        time_col = "wall_time_total"  # por compatibilidad

        err_col = (
            "err_L2_final"  # coherente con la idea de “lo entregado tras K sweeps”
        )
        if time_col not in df0.columns or err_col not in df0.columns:
            print("[work_precision] Required columns missing; skipping.")
            return

        df0 = df0.replace([np.inf, -np.inf], np.nan).dropna(subset=[time_col, err_col])
        df0 = df0[(df0[time_col] > 0) & (df0[err_col] > 0)]
        if df0.empty:
            print("[work_precision] No positive data after filtering; skipping plot.")
            return

        def _pretty_scatter(gdf, title: str, stem_suffix: str):
            if gdf.empty:
                return False
            x = gdf[time_col].astype(float).to_numpy()
            y = gdf[err_col].astype(float).to_numpy()
            if x.size == 0:
                return False

            if size_by is not None and size_by in gdf.columns:
                raw = gdf[size_by].astype(float)
                sizes = np.clip(
                    30 + 15 * (raw - raw.min()) / max(raw.max() - raw.min(), 1e-12),
                    35,
                    120,
                )
                sizes = sizes.to_numpy()
            else:
                sizes = np.full_like(x, 60, dtype=float)

            with pub_style(width_in=3.6, fontsize=9):
                fig, ax_ = plt.subplots()
                ax_.set_xscale("log")
                ax_.set_yscale("log")
                ax_.grid(True, which="both")

                if legend_by in gdf.columns:
                    for uv, sdf in gdf.groupby(legend_by):
                        idx = sdf.index
                        ax_.scatter(
                            sdf[time_col],
                            sdf[err_col],
                            s=(
                                sizes
                                if np.isscalar(sizes)
                                else sizes[[gdf.index.get_loc(i) for i in idx]]
                            ),
                            alpha=alpha,
                            linewidths=0.4,
                            edgecolors="black",
                            label=f"{legend_by}={uv}",
                        )
                    ax_.legend(frameon=False, loc="best", title=legend_by)
                else:
                    ax_.scatter(
                        x, y, s=sizes, alpha=alpha, linewidths=0.4, edgecolors="black"
                    )

                ax_.set_xlabel("wall_time_total [s]")
                ax_.set_ylabel(r"$\|e\|_{L^2}$")

                try:
                    if x.size >= 2:
                        xm, ym = np.log(x), np.log(y)
                        m, b = np.polyfit(xm, ym, 1)
                        xs = np.linspace(x.min(), x.max(), 100)
                        ax_.plot(
                            xs,
                            np.exp(b) * xs**m,
                            linestyle="--",
                            marker="x",
                            linewidth=1.0,
                            alpha=0.7,
                        )
                except Exception:
                    pass

                _polish_axes(ax_)
                save_pub_figure(
                    fig,
                    stem=f"work_precision_{stem_suffix}",
                    folder="figures",
                    dpi=300,
                    also_pdf=True,
                )
                plt.close(fig)
            return True

        if facet_by is None or not any(k in df0.columns for k in facet_by):
            _pretty_scatter(df0, "Work–Precision", "all")
            return

        for gvals, gdf in df0.groupby(
            [k for k in facet_by if k in df0.columns], dropna=False
        ):
            if not isinstance(gvals, tuple):
                gvals = (gvals,)
            meta = dict(zip([k for k in facet_by if k in df0.columns], gvals))
            slug = "_".join(f"{k}{meta[k]}" for k in meta)
            _pretty_scatter(gdf, title="", stem_suffix=slug)

    def work_precision_grid(
        self,
        legend_by: str = "nodes",
        size_by: str | None = "sw",
        *,
        metric: str = "sweep",  # "sweep" | "coll" | "final"
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> None:
        """
        Grid of work–precision panels: rows = prectype, columns = solver.
        Each panel is log–log scatter of ||e||_{L^2} vs wall_time_total.
        """
        df0 = self.df.reset_index()
        for k, v in filters.items():
            df0 = self._filter_eq(df0, k, v)

        # --- elige columna de error ---
        colmap = {
            "sweep": "err_L2_sweep",
            "coll": "err_L2_coll",
            "final": "err_L2_final",
        }
        err_col = colmap.get(str(metric).lower(), "err_L2_sweep")
        required = {"wall_time_total", "prectype", "solver"}
        if err_col not in df0.columns:
            if "err_L2_final" in df0.columns:
                err_col = "err_L2_final"
            else:
                print("[work_precision_grid] Missing error column; skipping.")
                return
        if not required.issubset(df0.columns):
            print("[work_precision_grid] Missing required columns; skipping.")
            return

        df0 = df0.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["wall_time_total", err_col]
        )
        df0 = df0[(df0["wall_time_total"] > 0) & (df0[err_col] > 0)]

        prectypes = list(df0["prectype"].dropna().unique())
        solvers = list(df0["solver"].dropna().unique())
        if not prectypes or not solvers:
            print("[work_precision_grid] Nothing to facet; skipping.")
            return

        with pub_style(width_in=6.5, fontsize=9):
            # Prepare grid
            nrows, ncols = len(prectypes), len(solvers)
            fig, axes = plt.subplots(nrows, ncols, squeeze=False)

            # ---- Build a single, global color mapping for legend_by (default: 'nodes') ----
            have_legend = legend_by in df0.columns
            all_vals = []
            if have_legend:
                all_vals = df0[legend_by].dropna().unique().tolist()
                try:
                    all_vals = sorted(all_vals, key=lambda v: float(v))
                except Exception:
                    all_vals = sorted(all_vals, key=lambda v: str(v))
                # Get a base palette from the current Matplotlib prop cycle
                prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
                base_colors = (
                    prop_cycle.by_key().get("color", [])
                    if prop_cycle is not None
                    else []
                )
                if not base_colors:
                    # Fallback palette
                    base_colors = list(plt.cm.tab10.colors)
                val2color = {
                    v: base_colors[i % len(base_colors)] for i, v in enumerate(all_vals)
                }
            else:
                val2color = {}

            # ---- Draw panels ----
            for i, p in enumerate(prectypes):
                for j, s in enumerate(solvers):
                    ax = axes[i, j]
                    # dentro del bucle de paneles en work_precision_grid
                    sub = df0[(df0["prectype"] == p) & (df0["solver"] == s)]
                    if sub.empty:
                        ax.axis("off")
                        continue

                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.grid(True, which="both")

                    # tamaño: mantener indexación robusta
                    if size_by is not None and size_by in sub.columns:
                        raw = sub[size_by].astype(float)
                        sizes = 30 + 15 * (raw - raw.min()) / max(
                            raw.max() - raw.min(), 1e-12
                        )
                        sizes = sizes.clip(
                            lower=35, upper=120
                        )  # <- Series, conserva índice
                    else:
                        sizes = pd.Series(60, index=sub.index)

                    if have_legend:
                        for uv, sdf in sub.groupby(legend_by):
                            color_ = val2color.get(uv)
                            ax.scatter(
                                sdf["wall_time_total"],
                                sdf[err_col],  # <- usar err_col
                                s=sizes.loc[sdf.index].to_numpy(),  # <- index seguro
                                alpha=0.9,
                                linewidths=0.4,
                                edgecolors="black",
                                color=color_,
                            )
                    else:
                        ax.scatter(
                            sub["wall_time_total"],
                            sub[err_col],  # <- usar err_col
                            s=sizes.to_numpy() if not np.isscalar(sizes) else sizes,
                            alpha=0.9,
                            linewidths=0.4,
                            edgecolors="black",
                        )

                    if i == nrows - 1:
                        ax.set_xlabel("wall_time_total [s]")
                    if j == 0:
                        ax.set_ylabel(r"$\|e\|_{L^2}$")

                    # ax.set_title(f"{p}, {s}")
                    # _annotate_method(ax, {"prectype": p, "solver": s})

                    _polish_axes(ax)

            # ---- Single shared legend (bottom) for legend_by (nodes) ----
            if have_legend and all_vals:
                handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markersize=5,
                        markerfacecolor=val2color[v],
                        markeredgecolor="black",
                        label=f"{legend_by}={v}",
                    )
                    for v in all_vals
                ]
                # Leave space at the bottom for the shared legend

                fig.legend(
                    handles,
                    [h.get_label() for h in handles],
                    loc="lower center",
                    ncol=min(len(handles), 6),
                    frameon=False,
                    title=legend_by,
                )

            save_pub_figure(
                fig,
                stem="work_precision_grid",
                folder="figures",
                dpi=300,
                also_pdf=True,
            )
            plt.close(fig)

    def sweeps_curves(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> None:
        """
        δ_k vs k para todos los runs que cumplan `filters`.
        Genera stems nuevos con sufijo y también los legacy.
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)

        curves, labels, label_keys = [], [], []
        for cand in ["prectype", "solver", "nodes", "deg", "sw"]:
            if cand in df2.columns and (cand not in filters):
                label_keys.append(cand)

        for _, row in df2.iterrows():
            path = Path(row["json_path"])
            with open(str(path), "r") as f:
                payload = json.load(f)
            triples = self._triples(payload)
            if not triples:
                continue
            last_step = max(s for (s, _, _, _) in triples)
            contr_blocks = self._contractions_for_step(payload, last_step)
            if not contr_blocks:
                continue
            delta_seq = contr_blocks[-1].get("delta_seq") or []
            if not delta_seq:
                continue
            curves.append(np.asarray(delta_seq, dtype=float))
            lbl = ", ".join(f"{k}={row[k]}" for k in label_keys)
            labels.append(lbl if lbl else path.stem)

        if not curves:
            print("[sweeps_curves] No contraction data; skipping.")
            return

        # --- DIBUJO ---
        ncurves = len(curves)
        legend_outside = ncurves > 6  # umbral: si hay muchas curvas, saco la leyenda

        # versión corta de etiquetas (M=, p=, sw=, solver/prec sin prefijo)
        def _short(lbl: str) -> str:
            s = (
                lbl.replace("nodes=", "M=")
                .replace("deg=", "p=")
                .replace("sw=", "sw=")
                .replace("solver=", "")
                .replace("prectype=", "")
            )
            return s

        with pub_style(width_in=(5.2 if legend_outside else 3.25), fontsize=9):
            fig, ax = plt.subplots()
            for y, lbl in zip(curves, labels):
                x = np.arange(1, len(y) + 1)
                ax.plot(x, y, linewidth=1.0, alpha=0.95, label=_short(lbl))
            ax.set_yscale("log")
            ax.set_xlabel("k (sweep)")
            ax.set_ylabel(r"$\delta_k$")
            # ax.set_title("Contraction curves")
            ax.grid(True, which="both", alpha=0.4)

            _polish_axes(ax)

            # leyenda: fuera si hay muchas curvas; dentro en varias columnas si son pocas
            if legend_outside:
                handles, lab = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    lab,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                    fontsize=7,
                    ncol=1,
                    borderaxespad=0.0,
                    handlelength=2.2,
                    columnspacing=0.8,
                )
            else:
                ax.legend(
                    frameon=False,
                    fontsize=7,
                    ncol=min(3, max(1, int(np.ceil(ncurves / 6)))),
                    handlelength=2.0,
                    columnspacing=0.8,
                )

            # sello método + parámetros
            fixed = {}
            for key in ("prectype", "solver"):
                if key in df2.columns:
                    vals = df2[key].dropna().unique()
                    if len(vals) == 1:
                        fixed[key] = vals[0]
            # _annotate_method(ax, fixed)
            # _annotate_params(ax, filters if isinstance(filters, dict) else {})

            suffix = _slugify(filters if isinstance(filters, dict) else {})
            save_pub_figure(
                fig,
                stem=f"contraction_curves_{suffix}",
                folder="figures",
                dpi=300,
                also_pdf=True,
            )
            save_pub_figure(
                fig,
                stem=f"sweeps_contraction_{suffix}",
                folder="figures",
                dpi=300,
                also_pdf=True,
            )
            # legacy
            save_pub_figure(
                fig, stem="contraction_curves", folder="figures", dpi=300, also_pdf=True
            )
            save_pub_figure(
                fig, stem="sweeps_contraction", folder="figures", dpi=300, also_pdf=True
            )
            plt.close(fig)

    def sweep_vs_collocation_errornorm(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        *,
        facet_by: tuple[str, str] | None = ("prectype", "solver"),
        metric_candidates: list[str] | None = None,
        use_last_step_only: bool = True,
        figure_per_case: bool = False,
        target: str = "err_sweep_vs_coll",  # NUEVO: elige métrica
        **filters: Any,
    ) -> None:
        """
        Curvas de error vs índice de sweep k, normalmente en el ÚLTIMO step.
        Por defecto, una figura por (prectype, solver); si `figure_per_case=True`,
        genera **una figura por run (JSON)**.

        Correcciones clave:
        - Preferimos métricas por sweep: "sweep_vs_real_errornorm" primero.
        "collocation_vs_real_errornorm" queda al final porque no varía con k.
        - Para un (step,k) con varios registros a tiempos t crecientes,
        nos quedamos con el **de mayor t** (registro más reciente).
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)
        if df2.empty:
            print("[sweep_vs_collocation_errornorm] No runs after filtering; skipping.")
            return

        # Preferencia de métricas: queremos **convergencia a collocation** primero
        # dentro de sweep_vs_collocation_errornorm(...)
        if metric_candidates is None:
            metric_candidates = [
                # canónicas (lo que produce _canon_key / _info_map):
                "sweep_vs_collocation_error_norm",
                "sweep_vs_real_error_norm",
                "collocation_vs_real_error_norm",
                # compat con JSON antiguos:
                "sweep_vs_collocation_errornorm",
                "sweep_vs_real_errornorm",
                "collocation_vs_real_errornorm",
                # últimos recursos:
                "final_l2",
                "norms_final_l2",
                "error_l2_final",
            ]

        def _plot_group(gdf: pd.DataFrame) -> None:
            curves: list[tuple[np.ndarray, np.ndarray, str]] = []

            for _, row in gdf.iterrows():
                path = Path(row["json_path"])
                try:
                    with open(str(path), "r") as f:
                        payload = json.load(f)
                except Exception:
                    continue
                triples = self._triples(payload)
                if not triples:
                    continue

                if use_last_step_only:
                    last_step = max(s for (s, _, _, _) in triples)
                    triples_iter = [t for t in triples if t[0] == last_step]
                else:
                    triples_iter = triples

                # Para cada k guardamos (t, valor) y nos quedamos con el mayor t
                err_by_k_t: dict[int, tuple[float, float]] = {}
                for step, t, kidx, key_str in sorted(
                    triples_iter, key=lambda z: (z[0], z[2], z[1])
                ):
                    values = payload.get(key_str, [])
                    metrics = self._info_map(payload, values)
                    val = np.nan
                    for mkey in metric_candidates:
                        if mkey in metrics and metrics[mkey] is not None:
                            try:
                                val = float(metrics[mkey])
                                break
                            except Exception:
                                pass
                    if not np.isfinite(val):
                        continue
                    prev = err_by_k_t.get(kidx)
                    if (prev is None) or (t > prev[0]):
                        err_by_k_t[kidx] = (float(t), float(val))

                if not err_by_k_t:
                    continue

                ks = np.array(sorted(err_by_k_t.keys()), dtype=int)
                errs = np.array([err_by_k_t[k][1] for k in ks], dtype=float)

                if errs.size >= 2 and np.any(np.diff(errs) > 0):
                    print(
                        f"[warn] Non-monotone error vs sweep in {path.name}; check metrics/data."
                    )

                parts = []
                if "n" in row and pd.notna(row["n"]):
                    parts.append(f"N={int(row['n'])}")
                if "dt" in row and pd.notna(row["dt"]):
                    parts.append(f"dt={_fmt_float(float(row['dt']))}")
                if "nodes" in row and pd.notna(row["nodes"]):
                    parts.append(f"M={int(row['nodes'])}")
                if "deg" in row and pd.notna(row["deg"]):
                    parts.append(f"p={int(row['deg'])}")
                if "sw" in row and pd.notna(row["sw"]):
                    parts.append(f"sw={int(row['sw'])}")
                lbl = ", ".join(parts) if parts else path.stem
                curves.append((ks, errs, lbl))

            if not curves:
                return

            legend_outside = len(curves) > 6
            with pub_style(width_in=(5.2 if legend_outside else 3.25), fontsize=9):
                fig, ax = plt.subplots()
                for ks, errs, lbl in curves:
                    ax.plot(
                        ks,
                        errs,
                        "--",
                        marker="^",
                        c="red",
                        linewidth=1.0,
                        alpha=0.95,
                        label=lbl,
                    )
                ax.set_yscale("log")
                ax.set_xlabel("sweep index $k$")
                ylabel_map = {
                    "err_sweep_vs_coll": r"$\|u^{(k)} - u^{\mathrm{coll}}\|_{L^2}$",
                    "err_sweep_vs_real": r"$\|u^{(k)} - u^{\mathrm{real}}\|_{L^2}$",
                    "err_coll_vs_real": r"$\|u^{\mathrm{coll}} - u^{\mathrm{real}}\|_{L^2}$",
                }
                ax.set_ylabel(
                    ylabel_map.get(str(target).lower(), ylabel_map["err_sweep_vs_coll"])
                    + (" (last step)" if use_last_step_only else "")
                )
                ax.grid(True, which="both", alpha=0.4)

                _polish_axes(ax)

                meta_ = {}
                for kk in ("prectype", "solver"):
                    if kk in gdf.columns:
                        u = gdf[kk].dropna().unique()
                        if len(u) == 1:
                            meta_[kk] = u[0]
                # _annotate_method(ax, meta_)

                if legend_outside:
                    handles, lab = ax.get_legend_handles_labels()
                    ax.legend(
                        handles,
                        lab,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=False,
                        fontsize=7,
                        ncol=1,
                        borderaxespad=0.0,
                        handlelength=2.2,
                        columnspacing=0.8,
                    )
                else:
                    ax.legend(
                        frameon=False,
                        fontsize=7,
                        ncol=min(3, max(1, int(np.ceil(len(curves) / 6)))),
                        handlelength=2.0,
                        columnspacing=0.8,
                    )

                suffix = _slug_from_df(gdf)
                stem = "sweep_vs_collocation_errornorm_" + (suffix if suffix else "all")
                save_pub_figure(
                    fig, stem=stem, folder="figures", dpi=300, also_pdf=True
                )
                plt.close(fig)

        if figure_per_case:
            # Una figura por JSON/run
            for _, gdf in df2.groupby(["json_path"], dropna=False):
                _plot_group(gdf)
            return

        if facet_by is None:
            _plot_group(df2)
            return

        facet_keys = [k for k in facet_by if k in df2.columns]
        if not facet_keys:
            _plot_group(df2)
            return
        for _, gdf in df2.groupby(facet_keys, dropna=False):
            _plot_group(gdf)

    def timings_box_vs_k(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> None:
        """
        Boxplots del tiempo por solver agrupado por sweep k.
        Añade línea horizontal punteada con la media del full collocation por paso.
        Guarda con stem 'timings_box_vs_k_<slug(params)>'.
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)
        if df2.empty:
            print("[timings_box_vs_k] No runs after filtering; skipping.")
            return

        for _, row in df2.iterrows():
            path = Path(row["json_path"])
            with open(str(path), "r") as f:
                payload = json.load(f)

            # per k
            per_k: dict[int, list[float]] = {}
            for key, val in payload.items():
                if key.endswith("_timings") and isinstance(val, list):
                    try:
                        _, _, k_s = key.split(",")
                        k_idx = int(k_s.split("_")[0])
                    except Exception:
                        continue
                    times = [
                        float(r.get("wall_time", 0.0))
                        for r in val
                        if isinstance(r, dict)
                    ]
                    if times:
                        per_k.setdefault(k_idx, []).extend(times)

            # media del full collocation por paso
            coll_step: dict[int, float] = {}
            for key2, val2 in payload.items():
                if (
                    key2.endswith("full_collocation_timing")
                    and isinstance(val2, list)
                    and val2
                ):
                    try:
                        s_idx2 = int(key2.split(",")[0])
                    except Exception:
                        continue
                    total = float(
                        sum(
                            float(r.get("wall_time", 0.0))
                            for r in val2
                            if isinstance(r, dict)
                        )
                    )
                    coll_step[s_idx2] = coll_step.get(s_idx2, 0.0) + total
            coll_mean = (
                float(np.mean(list(coll_step.values()))) if coll_step else float("nan")
            )

            if not per_k:
                print(
                    f"[timings_box_vs_k] No timing blocks found in {path.name}; skipping."
                )
                continue

            ks = sorted(per_k.keys())
            data = [per_k[kidx] for kidx in ks]

            with pub_style(width_in=3.25, fontsize=9):
                fig, ax = plt.subplots()
                bp = ax.boxplot(
                    data,
                    showmeans=True,
                    meanline=True,
                    patch_artist=True,
                    boxprops=dict(linewidth=0.8),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    medianprops=dict(linewidth=0.8),
                    meanprops=dict(linewidth=0.8),
                    flierprops=dict(
                        marker="o", markersize=2.5, alpha=0.6, linewidth=0.0
                    ),
                )
                for b in bp["boxes"]:
                    b.set_facecolor("none")
                    b.set_alpha(0.3)

                ax.set_yscale("log")
                ax.set_xlabel("Sweep index $k$")
                ax.set_ylabel("Per-solver time [s]")
                ax.set_axisbelow(True)
                ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
                ax.grid(False, which="minor")
                ax.set_xticks(ks)
                ax.set_xticklabels([str(k) for k in ks])

                _polish_axes(ax)

                title_prectype = (
                    str(row.get("prectype"))
                    if ("prectype" in row and pd.notna(row["prectype"]))
                    else ""
                )
                title_solver = (
                    str(row.get("solver"))
                    if ("solver" in row and pd.notna(row["solver"]))
                    else ""
                )
                # ax.set_title(f"{title_prectype} — {title_solver}\nTimings per sweep")

                # _annotate_method(
                # ax, {"prectype": title_prectype, "solver": title_solver}
                # )
                params_for_slug = {
                    kk: row[kk]
                    for kk in (
                        "n",
                        "dt",
                        "sw",
                        "nodes",
                        "deg",
                        "prectype",
                        "solver",
                    )
                    if kk in row and pd.notna(row[kk])
                }
                # _annotate_params(ax, params_for_slug)

                if np.isfinite(coll_mean):
                    ax.axhline(
                        coll_mean,
                        linestyle="-",
                        linewidth=1.2,
                        color="0.2",
                        zorder=20,  # por encima de la rejilla
                    )

                stem_suffix = _full_slug_from_row(row)
                out_dir = Path("figures")
                out_dir.mkdir(parents=True, exist_ok=True)
                base = out_dir / f"timings_box_vs_k_{stem_suffix}"
                fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
                fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
                plt.close(fig)

    def timings_box_by_node_and_by_sweep(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        *,
        exclude_first_sweep: bool = False,
        exclude_first_node: bool = False,
        exclude_sweeps: list[int] | None = None,
        exclude_nodes: list[int] | None = None,
        **filters: Any,
    ) -> None:
        """
        Dos boxplots complementarios para un run:
        (A) por nodo: en cada (step, node) calcula el promedio a lo largo de todos los sweeps,
            y usa esos promedios (uno por step) como muestras ⇒ una caja por nodo.
        (B) por sweep k: en cada (step, k) calcula el promedio a lo largo de todos los nodos,
            y usa esos promedios (uno por step) como muestras ⇒ una caja por k.

        Guarda con stems:
        - 'timings_box_by_node_<slug(params)>'
        - 'timings_box_by_sweep_<slug(params)>'
        """

        # ---------- helper de dibujo 100% consistente entre ambos paneles ----------
        def _draw_boxplot(
            ax, data: list[list[float]], xvals: list[int], xlabel: str, coll_mean: float
        ):
            bp = ax.boxplot(
                data,
                showmeans=True,
                meanline=True,
                patch_artist=True,
                boxprops=dict(linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                medianprops=dict(linewidth=0.8),
                meanprops=dict(linewidth=0.8),
                flierprops=dict(marker="o", markersize=2.5, alpha=0.6, linewidth=0.0),
            )
            for b in bp["boxes"]:
                b.set_facecolor("none")
                b.set_alpha(0.3)

            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("mean per step [s]")
            ax.set_axisbelow(True)
            ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
            ax.grid(False, which="minor")
            ax.set_xticks(xvals)
            ax.set_xticklabels([str(int(x)) for x in xvals])

            _polish_axes(ax)

            # MISMA línea de referencia en AMBAS figuras: azul sólido, grosor 1.2, sin leyenda.
            if np.isfinite(coll_mean):
                ax.axhline(
                    coll_mean, linestyle="-", linewidth=1.2, color="C0", zorder=20
                )

        # --------------------------- filtrado/lectura ---------------------------
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)
        if df2.empty:
            print(
                "[timings_box_by_node_and_by_sweep] No runs after filtering; skipping."
            )
            return

        excl_sweeps = set(int(k) for k in (exclude_sweeps or []))
        excl_nodes = set(int(j) for j in (exclude_nodes or []))

        for _, row in df2.iterrows():
            path = Path(row["json_path"])
            try:
                with open(str(path), "r") as f:
                    payload = json.load(f)
            except Exception:
                continue

            # mean full collocation per step (línea horizontal de referencia)
            coll_step: dict[int, float] = {}
            for key2, val2 in payload.items():
                if (
                    key2.endswith("full_collocation_timing")
                    and isinstance(val2, list)
                    and val2
                ):
                    try:
                        s_idx2 = int(key2.split(",")[0])
                    except Exception:
                        continue
                    total = float(
                        sum(
                            float(r.get("wall_time", 0.0))
                            for r in val2
                            if isinstance(r, dict)
                        )
                    )
                    coll_step[s_idx2] = coll_step.get(s_idx2, 0.0) + total
            coll_mean = (
                float(np.mean(list(coll_step.values()))) if coll_step else float("nan")
            )

            # Acumuladores finos
            step_node_times: dict[tuple[int, int], list[float]] = {}
            step_k_times: dict[tuple[int, int], list[float]] = {}

            for key, val in payload.items():
                if not (key.endswith("_timings") and isinstance(val, list)):
                    continue
                try:
                    s_str, _, k_s = key.split(",")
                    s_idx = int(s_str)
                    k_idx = int(k_s.split("_")[0])
                except Exception:
                    continue

                # exclusiones de sweeps
                if exclude_first_sweep and k_idx == 1:
                    continue
                if k_idx in excl_sweeps:
                    continue

                for r in val:
                    try:
                        t = float(r.get("wall_time", 0.0))
                    except Exception:
                        continue
                    node = r.get("node", r.get("solver_index"))
                    if isinstance(node, (int, np.integer)):
                        node = int(node)
                    else:
                        node = -1  # si no hay identificador, lo etiquetamos como -1

                    # exclusiones de nodos
                    if exclude_first_node and node == 0:
                        continue
                    if node in excl_nodes:
                        continue

                    step_node_times.setdefault((s_idx, node), []).append(t)
                    step_k_times.setdefault((s_idx, k_idx), []).append(t)

            # (A) Por nodo: para cada (step, node) → promedio a lo largo de k
            node2vals: dict[int, list[float]] = {}
            for (s_idx, node), times in step_node_times.items():
                if not times:
                    continue
                node2vals.setdefault(node, []).append(float(np.mean(times)))

            # (B) Por sweep k: para cada (step, k) → promedio a lo largo de nodos
            k2vals: dict[int, list[float]] = {}
            for (s_idx, k_idx), times in step_k_times.items():
                if not times:
                    continue
                k2vals.setdefault(k_idx, []).append(float(np.mean(times)))

            if not node2vals and not k2vals:
                print(
                    f"[timings_box_by_node_and_by_sweep] No timing data in {path.name}; skipping."
                )
                continue

            # Parámetros para sello y stem
            params = {
                kk: row[kk]
                for kk in ("n", "dt", "sw", "nodes", "deg", "prectype", "solver")
                if kk in row and pd.notna(row[kk])
            }
            title_prectype = (
                str(row.get("prectype")) if pd.notna(row.get("prectype")) else ""
            )
            title_solver = str(row.get("solver")) if pd.notna(row.get("solver")) else ""
            stem_suffix = _full_slug_from_row(row)

            # ---------- (A) BOX PLOT POR NODO ----------
            if node2vals:
                # eliminar el nodo fantasma -1 si aparece
                if -1 in node2vals:
                    del node2vals[-1]

                nodes_sorted = sorted(node2vals.keys())
                data_nodes = [node2vals[n] for n in nodes_sorted]

                with pub_style(width_in=3.25, fontsize=9):
                    fig, ax = plt.subplots()
                    # aquí, en lugar de usar directamente nodes_sorted como posiciones X
                    # usamos 1..M para las cajas, pero en las etiquetas mostramos el índice real
                    positions = list(range(1, len(nodes_sorted) + 1))
                    _draw_boxplot(
                        ax,
                        data_nodes,
                        positions,
                        xlabel="node index $j$",
                        coll_mean=coll_mean,
                    )
                    ax.set_xticklabels(
                        [f"{j}" for j in nodes_sorted]
                    )  # j real (0,1,2,...)
                    # _annotate_method(
                    # ax, {"prectype": title_prectype, "solver": title_solver}
                    # )
                    out_dir = Path("figures")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    base = out_dir / f"timings_box_by_node_{stem_suffix}"
                    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
                    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
                    plt.close(fig)

            # ---------- (B) BOX PLOT POR SWEEP ----------
            if k2vals:
                ks_sorted = sorted(k2vals.keys())
                data_k = [k2vals[k] for k in ks_sorted]
                with pub_style(width_in=3.25, fontsize=9):
                    fig, ax = plt.subplots()
                    _draw_boxplot(
                        ax,
                        data_k,
                        ks_sorted,
                        xlabel="sweep index $k$",
                        coll_mean=coll_mean,
                    )
                    # _annotate_method(
                    # ax, {"prectype": title_prectype, "solver": title_solver}
                    # )
                    out_dir = Path("figures")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    base = out_dir / f"timings_box_by_sweep_{stem_suffix}"
                    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
                    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
                    plt.close(fig)

    def full_coll_walltime(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> None:
        """
        Single aggregate figure: cumulative collocation wall-time per time step,
        one curve per run matching `filters`.
        Saves a single figure with stem 'full_coll_walltime_all_<slug>' and also
        a legacy 'full_coll_walltime' for compatibility.
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)
        if df2.empty:
            print("[full_coll_walltime] No runs after filtering; skipping.")
            return

        curves: list[tuple[np.ndarray, np.ndarray]] = []
        labels: list[str] = []

        for _, row in df2.iterrows():
            path = Path(row["json_path"])
            try:
                with open(str(path), "r") as f:
                    payload = json.load(f)
            except Exception:
                continue

            # collocation por paso (sumando si hay varias entradas por step)
            step2t: dict[int, float] = {}
            for key, val in payload.items():
                if (
                    key.endswith("full_collocation_timing")
                    and isinstance(val, list)
                    and val
                ):
                    try:
                        s_idx = int(key.split(",")[0])
                    except Exception:
                        continue
                    total = float(
                        sum(
                            float(r.get("wall_time", 0.0))
                            for r in val
                            if isinstance(r, dict)
                        )
                    )
                    step2t[s_idx] = step2t.get(s_idx, 0.0) + total

            if not step2t:
                continue

            steps = sorted(step2t.keys())
            coll_cum = np.cumsum([step2t[s] for s in steps])
            curves.append(
                (np.asarray(steps, dtype=int), np.asarray(coll_cum, dtype=float))
            )

            # etiqueta compacta para la leyenda
            parts = []
            if "n" in row and pd.notna(row["n"]):
                parts.append(f"N={int(row['n'])}")
            if "dt" in row and pd.notna(row["dt"]):
                parts.append(f"dt={_fmt_float(float(row['dt']))}")
            if "nodes" in row and pd.notna(row["nodes"]):
                parts.append(f"M={int(row['nodes'])}")
            if "deg" in row and pd.notna(row["deg"]):
                parts.append(f"p={int(row['deg'])}")
            if "sw" in row and pd.notna(row["sw"]):
                parts.append(f"sw={int(row['sw'])}")
            if "prectype" in row and pd.notna(row["prectype"]):
                parts.append(str(row["prectype"]))
            if "solver" in row and pd.notna(row["solver"]):
                parts.append(str(row["solver"]))
            labels.append(", ".join(parts))

        if not curves:
            print("[full_coll_walltime] No full collocation timings found; skipping.")
            return

        legend_outside = len(curves) > 6
        with pub_style(width_in=(5.2 if legend_outside else 3.25), fontsize=9):
            fig, ax = plt.subplots()
            linestyles = ["-", "--", "-.", ":"]
            for i, ((steps, coll_cum), lbl) in enumerate(zip(curves, labels)):
                ls = linestyles[i % len(linestyles)]
                ax.plot(
                    steps, coll_cum, linestyle=ls, linewidth=1.0, alpha=0.95, label=lbl
                )
            ax.set_xlabel("time step")
            ax.set_ylabel("cumulative collocation wall-time [s]")
            # ax.set_title("Full collocation cumulative wall-time (all runs)")
            ax.grid(True, which="both", alpha=0.4)

            # leyenda: fuera si hay muchas curvas
            if legend_outside:
                handles, lab = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    lab,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                    fontsize=7,
                    ncol=1,
                    borderaxespad=0.0,
                    handlelength=2.2,
                    columnspacing=0.8,
                )
            else:
                ax.legend(
                    frameon=False,
                    fontsize=7,
                    ncol=min(3, max(1, int(np.ceil(len(curves) / 6)))),
                    handlelength=2.0,
                    columnspacing=0.8,
                )

            # sello solo con los filtros fijos del panel
            # # _annotate_params(ax, filters if isinstance(filters, dict) else {})

            suffix = _slug_from_df(df2)
            stem = "full_coll_walltime_all_" + (suffix if suffix else "all")
            save_pub_figure(fig, stem=stem, folder="figures", dpi=300, also_pdf=True)
            # nombre legacy por compatibilidad
            save_pub_figure(
                fig, stem="full_coll_walltime", folder="figures", dpi=300, also_pdf=True
            )
            plt.close(fig)

    def cumulative_walltime_vs_step(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> None:
        """
        Compara acumulados por paso entre:
        (i) suma sobre k del máximo tiempo por solver (ruta crítica SDC), y
        (ii) collocation total.
        Stem: 'cum_walltime_vs_step_<slug>'.
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)
        if df2.empty:
            print("[cum_walltime_vs_step] No runs after filtering; skipping.")
            return

        for _, row in df2.iterrows():
            path = Path(row["json_path"])
            with open(str(path), "r") as f:
                payload = json.load(f)

            # (ii) collocation por paso: suma de TODAS las filas del buffer del step
            coll_per_step: dict[int, float] = {}
            for key, val in payload.items():
                if (
                    key.endswith("full_collocation_timing")
                    and isinstance(val, list)
                    and val
                ):
                    try:
                        s_idx = int(key.split(",")[0])
                    except Exception:
                        continue
                    total = float(
                        sum(
                            float(r.get("wall_time", 0.0))
                            for r in val
                            if isinstance(r, dict)
                        )
                    )
                    coll_per_step[s_idx] = coll_per_step.get(s_idx, 0.0) + total

            # (i) ruta crítica SDC: para cada (step,k) tomar max entre subsolvers y sumar sobre k
            sweeps_crit_per_step: dict[int, float] = {}
            for key, val in payload.items():
                if key.endswith("_timings") and isinstance(val, list):
                    try:
                        s_str, _, _ = key.split(",")
                        s_idx = int(s_str)
                    except Exception:
                        continue
                    maxima = [
                        float(r.get("wall_time", 0.0))
                        for r in val
                        if isinstance(r, dict)
                    ]
                    if maxima:
                        sweeps_crit_per_step[s_idx] = sweeps_crit_per_step.get(
                            s_idx, 0.0
                        ) + float(np.max(maxima))

            if not sweeps_crit_per_step and not coll_per_step:
                print(
                    f"[cum_walltime_vs_step] No timing info in {path.name}; skipping."
                )
                continue

            steps = sorted(set(coll_per_step.keys()) | set(sweeps_crit_per_step.keys()))
            coll_cum = np.cumsum([coll_per_step.get(s, 0.0) for s in steps])
            sdc_cum = np.cumsum([sweeps_crit_per_step.get(s, 0.0) for s in steps])

            with pub_style(width_in=3.25, fontsize=9):
                fig, ax = plt.subplots()
                ax.plot(
                    steps, sdc_cum, linewidth=1.2, label="SDC sweeps (critical path)"
                )
                ax.plot(
                    steps,
                    coll_cum,
                    linewidth=1.2,
                    linestyle="--",
                    label="Full collocation",
                )
                ax.set_xlabel("time step")
                ax.set_ylabel("cumulative wall-time [s]")
                ax.grid(True, which="both", alpha=0.4)
                ax.legend(frameon=False, loc="best")

                meta_ = {
                    kk: row[kk]
                    for kk in ("prectype", "solver")
                    if kk in row and pd.notna(row[kk])
                }
                # _annotate_method(ax, meta_)

                params = {
                    kk: row[kk]
                    for kk in ("n", "dt", "sw", "nodes", "deg", "prectype", "solver")
                    if kk in row and pd.notna(row[kk])
                }
                stem = "cum_walltime_vs_step_" + _full_slug_from_row(row)
                save_pub_figure(
                    fig, stem=stem, folder="figures", dpi=300, also_pdf=True
                )
                plt.close(fig)

    def single_convergence_curve(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> None:
        r"""
        Minimal, focused plot: a single semilogy curve of delta_k vs k for each
        run that matches `filters` (typically you'll filter down to one run).
        Adds a geometric-tail fit (slope log rho) and reports \hat{rho} and rho_inf.
        Stems: 'single_convergence_<slug>'.
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)
        if df2.empty:
            print("[single_convergence_curve] No runs after filtering; skipping.")
            return

        for _, row in df2.iterrows():
            path = Path(row["json_path"])
            try:
                with open(str(path), "r") as f:
                    payload = json.load(f)
            except Exception as e:
                print(f"[single_convergence_curve] Could not read {path.name}: {e}")
                continue

            triples = self._triples(payload)
            if not triples:
                continue
            last_step = max(s for (s, _, _, _) in triples)
            contr_blocks = self._contractions_for_step(payload, last_step)
            if not contr_blocks:
                continue
            block = contr_blocks[-1]
            delta_seq = np.asarray(block.get("delta_seq") or [], dtype=float)
            rho_seq = np.asarray(block.get("rho_seq") or [], dtype=float)
            if delta_seq.size == 0:
                continue

            k_arr = np.arange(1, delta_seq.size + 1)
            # geometric tail fit using the last half
            start = max(0, delta_seq.size // 2)
            rho_hat = np.nan
            r2 = np.nan
            try:
                xm = k_arr[start:]
                ym = np.log(np.maximum(delta_seq[start:], 1e-300))
                m, b = np.polyfit(xm, ym, 1)
                rho_hat = float(np.exp(m))
                yhat = m * xm + b
                ss_res = float(np.sum((ym - yhat) ** 2))
                ss_tot = float(np.sum((ym - np.mean(ym)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            except Exception:
                pass

            rho_inf = self._geom_mean_tail(rho_seq.tolist()) if rho_seq.size else np.nan

            with pub_style(width_in=3.25, fontsize=9):
                fig, ax = plt.subplots()
                ax.plot(k_arr, delta_seq, marker="o", linewidth=1.0, alpha=0.95)
                ax.set_yscale("log")
                ax.set_xlabel("k (sweep)")
                ax.set_ylabel(r"$\delta_k$")
                # ax.set_title("Contraction (single curve)")
                ax.grid(True, which="both", alpha=0.4)

                # tail fit line drawn over the tail range
                if not np.isnan(rho_hat):
                    xs = k_arr[start:]
                    # C from first tail point
                    C = float(delta_seq[start]) / (rho_hat ** (k_arr[start] - 1))
                    ax.plot(
                        xs,
                        C * (rho_hat ** (xs - 1)),
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.8,
                    )
                    ax.text(
                        0.02,
                        0.02,
                        f"$\\hat\\rho$≈{rho_hat:.3f}, $R^2$≈{r2:.3f}, $\\rho_\\infty$≈{rho_inf:.3f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        alpha=0.9,
                    )

                # stamps
                meta_ = {
                    kk: row[kk]
                    for kk in ("prectype", "solver")
                    if kk in row and pd.notna(row[kk])
                }
                # _annotate_method(ax, meta_)
                params = {
                    kk: row[kk]
                    for kk in ("n", "dt", "sw", "nodes", "deg", "prectype", "solver")
                    if kk in row and pd.notna(row[kk])
                }
                # # _annotate_params(ax, params)

                stem = "single_convergence_" + _full_slug_from_row(row)
                save_pub_figure(
                    fig, stem=stem, folder="figures", dpi=300, also_pdf=True
                )
                plt.close(fig)

    def onepager(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        **filters: Any,
    ) -> None:
        r"""
        A compact 3-panel "one-pager" per run (after applying `filters`) that
        maximizes information with minimal figures:
        (1) δ_k vs k (semilogy) with geometric tail fit (reports \hat{rho}, \rho_∞).
        (2) Per-sweep timing profile (box per k) across all steps in the run.
        (3) Cumulative wall-time by step: SDC critical path vs full collocation.
        Saves as 'onepager_<slug>'.
        """
        df2 = self.df.reset_index()
        df2 = self._exclude_df(
            df2,
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
        )
        for k, v in filters.items():
            df2 = self._filter_eq(df2, k, v)
        if df2.empty:
            print("[onepager] No runs after filtering; skipping.")
            return

        for _, row in df2.iterrows():
            path = Path(row["json_path"])
            try:
                with open(str(path), "r") as f:
                    payload = json.load(f)
            except Exception as e:
                print(f"[onepager] Could not read {path.name}: {e}")
                continue

            # ---- (1) contraction curve for last step ----
            triples = self._triples(payload)
            if not triples:
                print(f"[onepager] No triples in {path.name}; skipping.")
                continue
            last_step = max(s for (s, _, _, _) in triples)
            contr_blocks = self._contractions_for_step(payload, last_step)
            delta_seq = np.asarray(
                (contr_blocks[-1].get("delta_seq") if contr_blocks else []) or [],
                dtype=float,
            )
            rho_seq = np.asarray(
                (contr_blocks[-1].get("rho_seq") if contr_blocks else []) or [],
                dtype=float,
            )

            # tail fit
            k_arr = np.arange(1, delta_seq.size + 1)
            start = max(0, delta_seq.size // 2)
            rho_hat = np.nan
            r2 = np.nan
            if delta_seq.size >= 2:
                try:
                    xm = k_arr[start:]
                    ym = np.log(np.maximum(delta_seq[start:], 1e-300))
                    m, b = np.polyfit(xm, ym, 1)
                    rho_hat = float(np.exp(m))
                    yhat = m * xm + b
                    ss_res = float(np.sum((ym - yhat) ** 2))
                    ss_tot = float(np.sum((ym - np.mean(ym)) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                except Exception:
                    pass
            rho_inf = self._geom_mean_tail(rho_seq.tolist()) if rho_seq.size else np.nan

            # ---- (2) per-k timings across all steps ----
            per_k: dict[int, list[float]] = {}
            for key, val in payload.items():
                if key.endswith("_timings") and isinstance(val, list):
                    try:
                        _, _, k_s = key.split(",")
                        k_idx = int(k_s.split("_")[0])
                    except Exception:
                        continue
                    times = [
                        float(r.get("wall_time", 0.0))
                        for r in val
                        if isinstance(r, dict)
                    ]
                    if times:
                        per_k.setdefault(k_idx, []).extend(times)
            ks_sorted = sorted(per_k.keys())
            timing_data = [per_k[kidx] for kidx in ks_sorted]

            # ---- (3) cumulative collocation vs SDC critical path ----
            coll_per_step: dict[int, float] = {}
            for key, val in payload.items():
                if (
                    key.endswith("full_collocation_timing")
                    and isinstance(val, list)
                    and val
                ):
                    try:
                        s_idx = int(key.split(",")[0])
                    except Exception:
                        continue
                    total = float(
                        sum(
                            float(r.get("wall_time", 0.0))
                            for r in val
                            if isinstance(r, dict)
                        )
                    )
                    coll_per_step[s_idx] = coll_per_step.get(s_idx, 0.0) + total

            sweeps_crit_per_step: dict[int, float] = {}
            for key, val in payload.items():
                if key.endswith("_timings") and isinstance(val, list):
                    try:
                        s_str, _, _ = key.split(",")
                        s_idx = int(s_str)
                    except Exception:
                        continue
                    maxima = [
                        float(r.get("wall_time", 0.0))
                        for r in val
                        if isinstance(r, dict)
                    ]
                    if maxima:
                        sweeps_crit_per_step[s_idx] = sweeps_crit_per_step.get(
                            s_idx, 0.0
                        ) + float(np.max(maxima))

            steps = sorted(set(coll_per_step.keys()) | set(sweeps_crit_per_step.keys()))
            coll_cum = (
                np.cumsum([coll_per_step.get(s, 0.0) for s in steps])
                if steps
                else np.array([])
            )
            sdc_cum = (
                np.cumsum([sweeps_crit_per_step.get(s, 0.0) for s in steps])
                if steps
                else np.array([])
            )

            # ---- draw 3-panel figure ----
            # Use a taller figure via pub_style height override
            with pub_style(width_in=6.5, height_in=6.5 * 0.62 * 3, fontsize=9):
                fig, axes = plt.subplots(3, 1, squeeze=True)
                ax1, ax2, ax3 = axes

                # (1) contraction
                ax1.plot(k_arr, delta_seq, marker="o", linewidth=1.0, alpha=0.95)
                ax1.set_yscale("log")
                ax1.set_xlabel("k (sweep)")
                ax1.set_ylabel(r"$\delta_k$")
                ax1.set_title("Contraction per sweep (last time step)")
                ax1.grid(True, which="both", alpha=0.4)
                if delta_seq.size >= 2 and not np.isnan(rho_hat):
                    xs = k_arr[start:]
                    C = float(delta_seq[start]) / (rho_hat ** (k_arr[start] - 1))
                    ax1.plot(
                        xs,
                        C * (rho_hat ** (xs - 1)),
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.8,
                    )
                    ax1.text(
                        0.02,
                        0.02,
                        f"$\\hat\\rho$≈{rho_hat:.3f}, $R^2$≈{r2:.3f}, $\\rho_\\infty$≈{rho_inf:.3f}",
                        transform=ax1.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        alpha=0.9,
                    )

                # (2) timings per k
                if ks_sorted:
                    bp = ax2.boxplot(
                        timing_data,
                        showmeans=True,
                        meanline=True,
                        patch_artist=True,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(linewidth=0.8),
                        meanprops=dict(linewidth=0.8),
                        flierprops=dict(
                            marker="o", markersize=2.5, alpha=0.6, linewidth=0.0
                        ),
                    )
                    for b in bp["boxes"]:
                        b.set_facecolor("none")
                        b.set_alpha(0.3)
                    ax2.set_yscale("log")
                    ax2.set_xlabel("sweep index $k$")
                    ax2.set_ylabel("per-solver time [s]")
                    ax2.set_title("Timing profile by sweep")
                    ax2.grid(True, which="both", alpha=0.4)
                    ax2.set_xticklabels([str(k) for k in ks_sorted])
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "No timing data",
                        transform=ax2.transAxes,
                        ha="center",
                        va="center",
                    )
                    ax2.axis("off")

                # (3) cumulative wall-time curves
                if steps:
                    ax3.plot(
                        steps,
                        sdc_cum,
                        linewidth=1.2,
                        label="SDC sweeps (critical path)",
                    )
                    ax3.plot(
                        steps,
                        coll_cum,
                        linewidth=1.2,
                        linestyle="--",
                        label="Full collocation",
                    )
                    ax3.set_xlabel("time step")
                    ax3.set_ylabel("cumulative wall-time [s]")
                    ax3.set_title("Cumulative wall-time by step")
                    ax3.grid(True, which="both", alpha=0.4)
                    ax3.legend(frameon=False, loc="best")
                else:
                    ax3.text(
                        0.5,
                        0.5,
                        "No cumulative timing data",
                        transform=ax3.transAxes,
                        ha="center",
                        va="center",
                    )
                    ax3.axis("off")

                meta_ = {
                    kk: row[kk]
                    for kk in ("prectype", "solver")
                    if kk in row and pd.notna(row[kk])
                }
                # _annotate_method(ax1, meta_)
                # _annotate_method(ax2, meta_)
                # _annotate_method(ax3, meta_)

                params = {
                    kk: row[kk]
                    for kk in ("n", "dt", "sw", "nodes", "deg", "prectype", "solver")
                    if kk in row and pd.notna(row[kk])
                }
                # # _annotate_params(ax1, params)

                stem = "onepager_" + _full_slug_from_row(row)
                save_pub_figure(
                    fig, stem=stem, folder="figures", dpi=300, also_pdf=True
                )
                plt.close(fig)

    def speedup_bars(
        self,
        exclude: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        exclude_regex: list[str] | None = None,
        *,
        baseline: str = "seq",  # "seq" (por defecto) o "coll" (si overlay=False)
        overlay: bool = False,  # NUEVO: dos barras por caso (coll detrás, seq delante)
        annotate: bool = True,
        bg_alpha: float = 0.35,  # opacidad barra de fondo (coll)
        fg_alpha: float = 0.90,  # opacidad barra frontal (seq)
        use_truncated_crit: bool = False,
        **filters: Any,
    ) -> None:
        """
        Barras de speedup horizontales.

        Modo clásico (overlay=False):
        - baseline="seq":   S = T_seq  / T_par
        - baseline="coll":  S = T_coll / T_par

        Modo overlay (overlay=True):
        - Dibuja **dos** barras por fila:
            fondo  → S_coll = T_coll / T_par  (color C2, alpha=bg_alpha)
            frente → S_seq  = T_seq  / T_par  (color C0, alpha=fg_alpha)
        - Línea vertical 1× en color C1.
        - Guarda como 'figures/speedup_bars_overlay.(png|pdf)'.
        """
        df = self.par_speedup(
            exclude=exclude,
            exclude_globs=exclude_globs,
            exclude_regex=exclude_regex,
            use_truncated_crit=use_truncated_crit,
        )
        if df is None or df.empty:
            print("[speedup_bars] No paired (par vs global) runs; skipping.")
            return
        for k, v in filters.items():
            df = self._filter_eq(df, k, v)
        if df.empty:
            print("[speedup_bars] No data after filtering; skipping.")
            return

        n_first = getattr(self, "crit_first_n", None)
        trunc_tag = f" [firstN={n_first}]" if (use_truncated_crit and n_first) else ""

        # limpieza mínima
        df = df.replace([np.inf, -np.inf], np.nan)
        for c in ("speedup", "speedup_vs_coll"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if overlay:
            # Necesitamos ambas columnas
            needed = []
            if "speedup_vs_global" not in df.columns:
                needed.append("speedup_vs_global")
            if "speedup_vs_coll" not in df.columns:
                needed.append("speedup_vs_coll")
            if needed:
                print(
                    f"[speedup_bars] Missing columns for overlay: {needed}; skipping."
                )
                return

            # Filtrado por valores válidos (permitimos NaN en la de coll para no perder filas)
            df = df[
                df["speedup_vs_global"].notna() & (df["speedup_vs_global"] > 0)
            ].copy()
            if df.empty:
                print(
                    "[speedup_bars] No positive 'speedup_vs_global' values; skipping."
                )
                return

            # Ordenamos por la métrica frontal (global/par)
            df = df.sort_values(["speedup_vs_coll"], ascending=False).reset_index(
                drop=True
            )

            # Etiqueta compacta
            def _lbl(row: pd.Series) -> str:
                parts1, parts2 = [], []
                if "n" in row and pd.notna(row["n"]):
                    parts1.append(f"N={int(row['n'])}")
                if "dt" in row and pd.notna(row["dt"]):
                    parts1.append(f"dt={_fmt_float(float(row['dt']))}")
                if "sw" in row and pd.notna(row["sw"]):
                    parts1.append(f"sw={int(row['sw'])}")
                if "nodes" in row and pd.notna(row["nodes"]):
                    parts2.append(f"M={int(row['nodes'])}")
                if "deg" in row and pd.notna(row["deg"]):
                    parts2.append(f"p={int(row['deg'])}")
                if "prectype" in row and pd.notna(row["prectype"]):
                    parts2.append(str(row["prectype"]))
                return " | ".join(parts1 + parts2)

            labels = [_lbl(df.loc[i]) for i in range(len(df))]
            y_glob = df["speedup_vs_global"].astype(float).to_numpy()
            y_coll = df["speedup_vs_coll"].astype(float).to_numpy()  # puede llevar NaN

            n = len(df)
            height_in = min(max(0.42 * n + 1.0, 2.6), 12.0)
            width_in = 6.5

            with pub_style(width_in=width_in, height_in=height_in, fontsize=9):
                fig, ax = plt.subplots()
                y_pos = np.arange(n)[::-1]

                # --- Fondo (más ancho): barra vs GLOBAL (verde C2) ---
                y_glob_plot = y_glob[::-1]
                ax.barh(
                    y_pos,
                    y_glob_plot,
                    height=0.66,
                    color="C2",
                    alpha=bg_alpha,
                    edgecolor="none",
                    label=r"$T_{\mathrm{glob}}/T_{\mathrm{par}}$",
                )

                # --- Frente (más estrecho): barra vs COLLOCATION (azul C0) ---
                y_coll_plot = np.where(np.isfinite(y_coll), y_coll, 0.0)[::-1]
                ax.barh(
                    y_pos,
                    y_coll_plot,
                    height=0.38,
                    color="C0",
                    alpha=fg_alpha,
                    edgecolor="black",
                    linewidth=0.8,
                    label=r"$T_{\mathrm{coll}}/T_{\mathrm{par}}$",
                )

                # Línea 1×
                ax.axvline(
                    1.0,
                    linestyle="--",
                    linewidth=0.9,
                    color="C1",
                    zorder=0,
                    label="1×",
                )

                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels[::-1])
                ax.tick_params(axis="y", labelsize=7)
                ax.set_xlabel("Speedup")
                ax.set_axisbelow(True)
                ax.grid(
                    True,
                    which="major",
                    axis="x",
                    linestyle="--",
                    linewidth=0.6,
                    alpha=0.35,
                )
                ax.grid(False, which="minor")

                # --- Anotaciones: SOLO para coll (azul). Nada para glob (verde). ---
                if annotate:
                    for yc, yp in zip(y_coll[::-1], y_pos):
                        if np.isfinite(yc):
                            ax.text(
                                yc * 1.006,
                                yp,
                                f"{yc:.2f}",
                                va="center",
                                ha="left",
                                fontsize=7,
                                color="C0",
                                alpha=min(bg_alpha + 0.15, 0.9),
                            )

                _polish_axes(ax)

                # Leyenda coherente con colores/asignaciones
                handles = [
                    Patch(
                        facecolor="C2",
                        edgecolor="none",
                        alpha=bg_alpha,
                        label=r"$T_{\mathrm{glob}}/T_{\mathrm{par}}$" + trunc_tag,
                    ),  # NEW
                    Patch(
                        facecolor="C0",
                        edgecolor="black",
                        alpha=fg_alpha,
                        label=r"$T_{\mathrm{coll}}/T_{\mathrm{par}}$" + trunc_tag,
                    ),  # NEW
                    Line2D([0], [0], color="C1", linestyle="--", label="1×"),
                ]
                ax.legend(handles=handles, frameon=False, loc="best")

                stem = "speedup_bars_overlay_glob_coll"
                if use_truncated_crit and n_first:
                    stem += f"_firstN{n_first}"  # NEW
                save_pub_figure(
                    fig, stem=stem, folder="figures", dpi=300, also_pdf=True
                )
                plt.close(fig)
                return

        # ----- Modo clásico (sin overlay), tal como lo tenías -----
        base = str(baseline).lower()
        col = "speedup" if base in ("seq", "sequential") else "speedup_vs_coll"
        if col not in df.columns:
            print(f"[speedup_bars] Column '{col}' missing; skipping.")
            return

        df = df.dropna(subset=[col])
        df = df[df[col] > 0]
        if df.empty:
            print("[speedup_bars] No positive values; skipping.")
            return
        df = df.sort_values([col], ascending=False).reset_index(drop=True)

        def _lbl(row: pd.Series) -> str:
            parts1, parts2 = [], []
            if "n" in row:
                parts1.append(f"N={int(row['n'])}")
            if "dt" in row:
                parts1.append(f"dt={_fmt_float(float(row['dt']))}")
            if "sw" in row:
                parts1.append(f"sw={int(row['sw'])}")
            if "nodes" in row:
                parts2.append(f"M={int(row['nodes'])}")
            if "deg" in row:
                parts2.append(f"p={int(row['deg'])}")
            if "prectype" in row and pd.notna(row["prectype"]):
                parts2.append(str(row["prectype"]))
            return " | ".join(parts1 + parts2)

        labels = [_lbl(df.loc[i]) for i in range(len(df))]
        yvals = df[col].astype(float).to_numpy()

        n = len(df)
        height_in = min(max(0.42 * n + 1.0, 2.6), 12.0)
        width_in = 6.5

        with pub_style(width_in=width_in, height_in=height_in, fontsize=9):
            fig, ax = plt.subplots()
            y_pos = np.arange(n)[::-1]
            ax.barh(
                y_pos,
                yvals[::-1],
                height=0.6,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            )
            ax.axvline(1.0, linestyle="--", linewidth=0.8, color="0.3", zorder=0)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels[::-1])
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel(
                "Speedup (T_seq / T_par)"
                if col == "speedup"
                else "Speedup vs collocation (T_coll / T_par)"
            )
            ax.set_axisbelow(True)
            ax.grid(
                True, which="major", axis="x", linestyle="--", linewidth=0.6, alpha=0.35
            )
            ax.grid(False, which="minor")
            if annotate:
                vals = yvals[::-1]
                for yi, yp in zip(vals, y_pos):
                    ax.text(
                        yi * 1.01, yp, f"{yi:.2f}", va="center", ha="left", fontsize=7
                    )
            _polish_axes(ax)
            stem = "speedup_bars" if col == "speedup" else "speedup_bars_vs_coll"
            save_pub_figure(fig, stem=stem, folder="figures", dpi=300, also_pdf=True)
            plt.close(fig)
