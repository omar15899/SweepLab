import re
import gc
from pathlib import Path
from typing import List, Dict, Any
from firedrake import *
from .filenamer import FileNamer, CheckpointAnalyser
import matplotlib.pyplot as plt
import pandas as pd


# # general pattern ‘heat_n{n}_dt{dt}_sw{sw}_{idx?}.h5’
# PAT = re.compile(
#     r"heat_n(?P<n>\d+)_dt(?P<dt>[0-9eE\+\-]+)_sw(?P<sw>\d+)" r"(?:_(?P<idx>\d+))?\.h5$"
# )

file_path = Path(
    "/Users/omarkhalil/Desktop/Universidad/ImperialCollege/Project/programming/solver/tests/heatfiles/HE"
)
file_name = "heat_n"
pattern = rf"{file_name}_n(?P<n>\d+)_dt(?P<dt>\d*(?:\d[eE]-\d+)?)_sw(?P<sw>\d+).*(?:_(?P<idx>\d+))?\.h5$"
deg = 4


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

            if k % 20 == 0:
                gc.collect()

        # We now convert to pandas dataframe in order to analyse it properly
        df = pd.DataFrame.from_dict(result, orient="index")
        # we convert it to multiindex
        df.index = pd.MultiIndex.from_tuples(df.index, names=self.keys)
        df.columns = [f_name for f_name in self.function_names]
        return df

    def _plot_and_fit(
        self,
        df2: pd.DataFrame,
        x_label: str,
        title_fmt: str,
        save_dir: str = "figures",
        file_stem: str = "convergence",
        group_size: int = 8,
        dpi: int = 300,
        marker_size: int = 55,
        fit_line: bool = True,
    ) -> Dict[str, float]:
        """
        Dibuja errores vs. paso/grado, estima el orden y guarda PDF+PNG.

        Si alguna serie contiene ceros/negativos, ese panel pasa a escala
        lineal (se anota el motivo) para evitar el ValueError de Matplotlib.
        """
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        orders: Dict[str, float] = {}
        cols: List[str] = list(df2.columns)

        # -------- estilo global --------
        with plt.rc_context(
            {
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "lines.linewidth": 1.4,
                "figure.dpi": dpi,
            }
        ):
            for page, start in enumerate(range(0, len(cols), group_size), 1):
                subset = cols[start : start + group_size]
                n_rows = int(np.ceil(len(subset) / 2))
                n_cols = 2

                # ------------- figura -------------
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(5.0 * n_cols, 3.8 * n_rows),
                    squeeze=False,
                    constrained_layout=True,  # adiós tight_layout y sus warnings
                )
                axes = axes.flatten()

                for ax, col in zip(axes, subset):
                    y_raw = df2[col].to_numpy(float)
                    x_raw = df2.index.to_numpy(float)

                    # Filtramos pares positivos; si quedan <2, no podemos log-ajustar
                    pos_mask = (x_raw > 0) & (y_raw > 0)
                    x_use, y_use = x_raw[pos_mask], y_raw[pos_mask]

                    logscale_ok = x_use.size >= 2

                    if logscale_ok:
                        # Ajuste log–log
                        slope, intercept = np.polyfit(np.log(x_use), np.log(y_use), 1)
                        orders[col] = -slope

                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        ax.scatter(
                            x_use,
                            y_use,
                            s=marker_size,
                            zorder=3,
                            label="datos",
                        )
                        if fit_line:
                            ax.plot(
                                x_use,
                                np.exp(intercept) * x_use**slope,
                                "--",
                                alpha=0.9,
                                label=f"ajuste  p≈{slope:.2f}",
                            )
                    else:
                        # Escala lineal; anotamos la razón
                        orders[col] = np.nan
                        ax.scatter(
                            x_raw,
                            y_raw,
                            s=marker_size,
                            zorder=3,
                            label="datos",
                        )
                        ax.text(
                            0.5,
                            0.5,
                            "escala lineal\n(ceros o negativos)",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="crimson",
                        )

                    ax.set_xlabel(x_label)
                    ax.set_ylabel(f"{col}-error")
                    ax.set_title(title_fmt.format(col=col, order=orders[col]))

                    # Error mínimo de la serie (en la escala actual)
                    ymin = y_raw.min()
                    ax.annotate(
                        f"min = {ymin:.2e}",
                        xy=(x_raw[-1], ymin),
                        xytext=(3, -12),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontsize=8,
                    )

                    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)

                # Quita paneles vacíos
                for ax in axes[len(subset) :]:
                    fig.delaxes(ax)

                namer = FileNamer(
                    file_name=f"{file_stem}_page{page}",
                    folder_name=save_dir.name,
                    path_name=str(save_dir.parent),
                    mode="pdf",
                )
                fig.savefig(namer.file)  # PDF vectorial
                # fig.savefig(namer.file.with_suffix(".png"))  # PNG rápido
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
        values = df2.index.get_level_values(spatial_key)
        mask_lower = (
            (values > spatial_lower_bound) if spatial_lower_bound is not None else True
        )
        mask_upper = (
            (values < spatial_higher_bound)
            if spatial_higher_bound is not None
            else True
        )
        df2 = df2[mask_lower & mask_upper]
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
        mask1 = (
            (df2.index.get_level_values(temporal_key) > temporal_lower_bound)
            if temporal_lower_bound is not None
            else True
        )
        mask2 = (
            (df2.index.get_level_values(temporal_key) < temporal_higher_bound)
            if temporal_higher_bound is not None
            else True
        )
        mask = mask1 & mask2
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
        values = df2.index.get_level_values(sweep_key)
        mask_lower = (
            (values > sweep_lower_bound) if sweep_lower_bound is not None else True
        )
        mask_upper = (
            (values < sweep_higher_bound) if sweep_higher_bound is not None else True
        )
        df2 = df2[mask_lower & mask_upper]
        df2 = df2.sort_index()
        return self._plot_and_fit(
            df2,
            x_label=sweep_key,
            title_fmt=f"Sweep convergence (n={spatial_val}, dt={temporal_val}) — {{col}}: order≈{{order:.2f}}",
        )
