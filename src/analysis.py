from pathlib import Path
from typing import List, Dict, Any, Callable
from firedrake import *
from .filenamer import FileNamer, CheckpointAnalyser
import matplotlib.pyplot as plt
from contextlib import contextmanager
import pandas as pd
import re
import gc


@contextmanager
def pub_style(width_in=3.25, height_in=None, fontsize=9):
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
                "savefig.pad_inches": 0.01,
                "text.usetex": False,
                "font.family": "serif",
                "mathtext.fontset": "stix",
                "axes.titleweight": "semibold",
            }
        )
        yield
    finally:
        plt.rcParams.update(old)


def save_pub_figure(
    fig, stem: str, folder: str = "figures", dpi: int = 300, also_pdf: bool = True
):
    """
    Save a figure with FileNamer in PNG (and optionally PDF for vector quality).
    """
    namer_png = FileNamer(file_name=stem, folder_name=folder, mode="png")
    fig.savefig(namer_png.file, dpi=dpi)
    if also_pdf:
        namer_pdf = FileNamer(file_name=stem, folder_name=folder, mode="pdf")
        fig.savefig(namer_pdf.file)


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

    def _plot_and_fit(
        self,
        df: pd.DataFrame,
        *,
        x_label: str,
        title_fmt: str,
        save_dir: str | Path = "figures",
        file_stem: str | None = None,
        dpi: int = 300,
    ) -> Dict[str, float]:

        if isinstance(df.index, pd.MultiIndex):
            x = df.index.get_level_values(x_label).astype(float).to_numpy()
        else:
            x = df.index.astype(float).to_numpy()

        orders: Dict[str, float] = {}
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for col in df.columns:
            y = df[col].astype(float).to_numpy()
            with pub_style(width_in=3.25, fontsize=9):
                fig, ax = plt.subplots()
                ax.scatter(x, y, s=40, label="data")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(x_label)
                ax.set_ylabel(f"{col} error (L2)")

                mask = (x > 0) & (y > 0)
                if mask.sum() >= 2:
                    m, b = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
                    p = m
                    orders[col] = p
                    ax.plot(
                        x[mask],
                        np.exp(b) * x[mask] ** m,
                        "--",
                        label=f"fitting: p≈{p:.2f}",
                    )
                else:
                    orders[col] = np.nan

                ax.set_title(title_fmt.format(col=col, order=orders[col]))
                ax.legend(frameon=False)
                ax.grid(True, which="both", alpha=0.4)
                fig.tight_layout()

                stem = file_stem or f"conv_{x_label}_{col}"
                save_pub_figure(fig, stem, folder=str(save_dir), dpi=dpi, also_pdf=True)
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
            title_fmt=f"Sweep convergence \n(n={spatial_val}, dt={temporal_val}) — {{col}}: order≈{{order:.2f}}",
        )

    def time_efficiency(
        self,
    ):
        pass
