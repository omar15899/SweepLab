import re
import gc
import itertools
from pathlib import Path
from typing import List, Dict, Any
from firedrake import *
from .filenamer import FileNamer, CheckpointAnalyser
import matplotlib.pyplot as plt
import pandas as pd


class ConvergenceAnalyser(CheckpointAnalyser):
    """
    In this class we will be addign one file_name, one pattern the function names
    which we have defined in our simulations and the real solution to those functions,
    and we will retrieve all the informatio concerning to convergence and so on.

    Las funciones exactas ya tienen que estar definidas en el espacio V mediante el operador
    interpolador, además tenemos que tener todo en ese sentido muy bien definido y ser conscientes
    de las propiedade sd ecad a uno de los casos en concreto.

    Aquí solo vamos a pasar el diccionario de la función list_checekpoints, y este programa de forma
    automática hará el resto de cálculos pertinentes.

    A las funciones de los checkpoint las instanciaremos una a una y calcularemos los errores para
    eliminarlas.

    f_exact: Serán expresiones UFL a interpolar
    """

    def __init__(
        self,
        file_path: Path,
        pattern: re.Pattern,
        keys: str | List[str],
        keys_type: callable | List[callable],
        f_exact_ufl: Function | List[Function],
        function_names: str | List[str] = ["u"],
        get_function_characteristics: bool = False,
    ):
        super().__init__(
            file_path,
            pattern,
            keys,
            keys_type,
            function_names,
            get_function_characteristics,
        )
        self.fs_exact_ufl = (
            f_exact_ufl if isinstance(f_exact_ufl, list) else [f_exact_ufl]
        )
        self.df = self.create_error_table()

    @staticmethod
    def L2_error(f_exact, f_num):
        return errornorm(f_exact, f_num, norm_type="L2")

    def create_error_table(self, norm_type="L2") -> pd.DataFrame:
        """
        Recorre todos los archivos que ya tenemos listados y extrae la función,
        por ahora lo vamos a hacer para el caso de una lista de 1.
        """
        result = {}
        for k, (key, file_params) in enumerate(self.checkpoint_list.items(), 1):
            # Extract the functions:
            for f_approx_name, f_exact_ufl in zip(
                self.function_names, self.fs_exact_ufl
            ):
                mesh, _, t_end, idx, f_approx = self.load_function_from_checkpoint(
                    file_params[0], f_approx_name
                )
                V = f_approx.function_space()
                x = SpatialCoordinate(mesh)
                f_exact = Function(V).interpolate(f_exact_ufl(t_end, x))
                error = errornorm(f_exact, f_approx, norm_type=norm_type)
                result.setdefault(key, []).append(error)

            if k % 5 == 0:
                # We have to delete all the meshesh in memory that are
                # not being used, if not an exception will arise with
                # enough big files.
                gc.collect()

        # We now convert to pandas dataframe in order to analyse it properly
        df = pd.DataFrame.from_dict(result, orient="index")
        # we convert it to multiindex
        df.index = pd.MultiIndex.from_tuples(df.index, names=self.keys)
        df.columns = [f_name for f_name in self.function_names]
        return df

    # def _plot_and_fit(
    #     self,
    #     df2: pd.DataFrame,
    #     x_label: str,
    #     title_fmt: str,
    #     save_dir: str = "figures",
    #     file_stem: str = "conv",
    #     dpi: int = 300,
    # ) -> Dict[str, float]:

    #     plt.style.use("ggplot")
    #     save_dir = Path(save_dir)
    #     save_dir.mkdir(parents=True, exist_ok=True)

    #     orders = {}

    #     for col in df2.columns:
    #         # Extraemos x, y (si hay MultiIndex tomamos el nivel que toca)
    #         if isinstance(df2.index, pd.MultiIndex):
    #             x = df2.index.get_level_values(x_label).astype(float).to_numpy()
    #         else:
    #             x = df2.index.astype(float).to_numpy()
    #         y = df2[col].astype(float).to_numpy()

    #         fig, ax = plt.subplots(figsize=(6, 4))
    #         ax.scatter(x, y, s=40, label="data")

    #         # Ajuste log–log sólo con puntos positivos
    #         mask = (x > 0) & (y > 0)
    #         if mask.sum() >= 2:
    #             m, b = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    #             orders[col] = -m
    #             ax.plot(x[mask], np.exp(b) * x[mask] ** m, "--", label=f"slope={m:.2f}")
    #             ax.set_xscale("log")
    #             ax.set_yscale("log")
    #         else:
    #             orders[col] = np.nan

    #         ax.set_xlabel(x_label)
    #         ax.set_ylabel(f"{col} error")
    #         ax.set_title(title_fmt)
    #         ax.legend()
    #         ax.grid(True, which="both", alpha=0.4)
    #         fig.tight_layout()

    #         namer = FileNamer(
    #             file_name=f"{file_stem}_{col}", folder_name=save_dir, mode="png"
    #         )
    #         fig.savefig(namer.file, dpi=dpi)
    #         plt.close(fig)

    #     return orders

    # def _plot_and_fit(
    #     self,
    #     df: pd.DataFrame,
    #     x_label: str,
    #     title: str,
    #     save_dir: str = "figures",
    #     file_stem: str | None = None,
    #     dpi: int = 300,
    # ) -> dict[str, float]:
    #     """Return a convergence‑order dictionary and save a single log–log figure.

    #     Parameters
    #     ----------
    #     df : pandas.DataFrame
    #         Rows are indexed by the discretisation parameter named *x_label* (either as a
    #         simple ``Index`` or as one level of a ``MultiIndex``).  Columns store error
    #         measures to be analysed.
    #     x_label : str
    #         Name of the horizontal‑axis variable (e.g. ``"dt"``, ``"n"`` or ``"k"``).
    #     title : str
    #         Figure title.
    #     save_dir : str, default ``"figures"``
    #         Output directory (created on demand).
    #     file_stem : str | None
    #         Stem of the PNG filename.  If ``None`` a name derived from *x_label* is used.
    #     dpi : int, default 300
    #         Figure resolution.

    #     Returns
    #     -------
    #     dict[str, float]
    #         Mapping ``{column_name: order}`` with the fitted convergence orders.
    #     """

    #     # ------------------------- data extraction ------------------------------
    #     if isinstance(df.index, pd.MultiIndex):
    #         x = df.index.get_level_values(x_label).astype(float).to_numpy()
    #     else:
    #         x = df.index.astype(float).to_numpy()

    #     orders: dict[str, float] = {}

    #     # ------------------------- figure set‑up --------------------------------
    #     fig, ax = plt.subplots(figsize=(7, 5))
    #     ax.set_xscale("log")
    #     ax.set_yscale("log")
    #     ax.set_xlabel(x_label, fontsize=11)
    #     ax.set_ylabel("error", fontsize=11)
    #     ax.set_title(title, fontsize=12, pad=10)

    #     # ------------------------- each error column ----------------------------
    #     for col in df.columns:
    #         y = df[col].astype(float).to_numpy()
    #         mask = (x > 0) & (y > 0)
    #         if mask.sum() < 2:
    #             ax.scatter(x, y, label=f"{col} (data)")
    #             orders[col] = float("nan")
    #             continue

    #         # least‑squares linear fit in log–log space: log(y) = m log(x) + b
    #         m, b = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    #         orders[col] = m

    #         y_fit = np.exp(b) * x**m

    #         ax.plot(x, y, marker="o", linestyle="", linewidth=0, label=f"{col} (data)")
    #         ax.plot(x, y_fit, linestyle="--", label=f"{col} fit, p≈{m:.2f}")

    #     # ------------------------- cosmetics ------------------------------------
    #     ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    #     ax.legend(frameon=False, fontsize=9)
    #     fig.tight_layout()

    #     # ------------------------- save figure ----------------------------------
    #     save_dir = Path(save_dir)
    #     save_dir.mkdir(exist_ok=True, parents=True)
    #     namer = FileNamer(
    #         file_name=file_stem or f"conv_{x_label}",
    #         folder_name=str(save_dir),
    #         mode="png",
    #     )
    #     fig.savefig(namer.file, dpi=dpi)
    #     plt.close(fig)

    #     return orders

    def _plot_and_fit(
        self,
        df: pd.DataFrame,
        *,
        x_label: str,
        title: str | None = None,
        save_dir: str | Path = "figures",
        file_stem: str | None = None,
        dpi: int = 300,
    ) -> Dict[str, float]:
        """Plot error curves on log–log axes and return fitted convergence orders.

        All error columns in *df* are plotted in a *single* figure.  The slope
        *p* of each log–log line ``error ≍ C·x^{-p}`` is reported in the legend.
        """
        # ------------------------- x‑axis extraction -------------------------
        if isinstance(df.index, pd.MultiIndex):
            x = df.index.get_level_values(x_label).to_numpy(float)
        else:
            x = df.index.to_numpy(float)

        # guarantee monotone x for nicer plots
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        df = df.iloc[sort_idx]

        # --------------------------- figure set‑up ---------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("error   (L2)")
        if title is not None:
            ax.set_title(title)

        markers = itertools.cycle("osd^vP*X")
        linestyles = itertools.cycle(["-", "--", "-.", ":"])
        orders: Dict[str, float] = {}

        # --------------------- iterate each error column ---------------------
        for col, mk, ls in zip(df.columns, markers, linestyles):
            y = df[col].to_numpy(float)
            ax.plot(x, y, marker=mk, linestyle=ls, label=col)

            mask = (x > 0) & (y > 0)
            if mask.sum() >= 2:
                m, b = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
                p = -m  # convergence order (positive)
                orders[col] = p
                y_fit = np.exp(b) * x**m
                ax.plot(x, y_fit, ls, linewidth=1, alpha=0.5)
                ax.text(
                    x[-1],
                    y_fit[-1],
                    f"  p≈{p:.2f}",
                    va="center",
                    fontsize=8,
                )
            else:
                orders[col] = float("nan")

        # --------------------------- cosmetics ------------------------------
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

        # ----------------------- save to disk -------------------------------
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        basename = file_stem or f"conv_{x_label}"
        namer = FileNamer(file_name=basename, folder_name=str(save_dir), mode="png")
        fig.savefig(namer.file, dpi=dpi)
        plt.close(fig)
        return orders

    def spatial_error_convergence(
        self,
        spatial_key: str,
        temporal_key: str,
        temporal_val: float,
        sweep_key: str,
        sweep_val: int,
        spatial_lower_bound: float | None = None,
        spatial_higher_bound: float | None = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        We need to have the reference of the spatial_key within
        the multindex in order to achieve correctly everything.

        kwargs is in case we have more indexes from the multi index
        that we have to fix.
        """
        idx = {temporal_key: temporal_val, sweep_key: sweep_val, **kwargs}
        df2 = self.df.xs(tuple(idx.values()), level=list(idx.keys()))
        values = df2.index.get_level_values(spatial_key).to_numpy(float)
        mask = np.ones(values.size, dtype=bool)
        if spatial_lower_bound is not None:
            mask &= values >= spatial_lower_bound
        if spatial_higher_bound is not None:
            mask &= values <= spatial_higher_bound
        df2 = df2[mask]
        df2 = df2.sort_index()
        return self._plot_and_fit(
            df2,
            x_label=spatial_key,
            title_fmt=f"Spatial convergence (dt={temporal_val}, sw={sweep_val}) — {{col}}: order≈{{order:.2f}}",
        )

    def temporal_error_convergence(
        self,
        temporal_key: str,
        spatial_key: str,
        spatial_val: float,
        sweep_key: str,
        sweep_val: int,
        temporal_lower_bound: int | None = None,
        temporal_higher_bound: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Fix n=spatial_val, sw=sweep_val and vary temporal_key.
        """
        idx = {spatial_key: spatial_val, sweep_key: sweep_val, **kwargs}
        df2 = self.df.xs(tuple(idx.values()), level=list(idx.keys()))
        values = df2.index.get_level_values(temporal_key).to_numpy(float)
        mask = np.ones(values.size, dtype=bool)
        if temporal_lower_bound is not None:
            mask &= values >= temporal_lower_bound
        if temporal_higher_bound is not None:
            mask &= values <= temporal_higher_bound
        df2 = df2[mask]
        df2 = df2.sort_index()
        return self._plot_and_fit(
            df2,
            x_label=temporal_key,
            title_fmt=f"Temporal convergence (n={spatial_val}, sw={sweep_val}) — {{col}}: order≈{{order:.2f}}",
        )

    def sweep_error_convergence(
        self,
        sweep_key: str,
        spatial_key: str,
        spatial_val: float,
        temporal_key: str,
        temporal_val: float,
        sweep_lower_bound: int | None = None,
        sweep_higher_bound: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Fix n=spatial_val, dt=temporal_val and vary sweep_key.
        """

        idx = {spatial_key: spatial_val, temporal_key: temporal_val, **kwargs}
        df2 = self.df.xs(tuple(idx.values()), level=list(idx.keys()))
        values = df2.index.get_level_values(sweep_key).to_numpy(float)
        mask = np.ones(values.size, dtype=bool)
        if sweep_lower_bound is not None:
            mask &= values >= sweep_lower_bound
        if sweep_higher_bound is not None:
            mask &= values <= sweep_higher_bound
        df2 = df2[mask]
        df2 = df2.sort_index()
        return self._plot_and_fit(
            df2,
            x_label=sweep_key,
            title_fmt=f"Sweep convergence (n={spatial_val}, dt={temporal_val}) — {{col}}: order≈{{order:.2f}}",
        )

    def time_efficiency(
        self,
    ):
        pass
