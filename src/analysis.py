import re
from pathlib import Path
from typing import List, Dict, Any
from firedrake import *
from filenamer import FileNamer, CheckpointAnalyser
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
        for key, file_params in self.checkpoint_list.items():
            # Extract the functions:
            for f_approx_name, f_exact_ufl in zip(
                self.function_names, self.fs_exact_ufl
            ):
                mesh, _, idx, t_end, f_approx = self.load_function_from_checkpoint(
                    file_params[0], f_approx_name
                )
                V = f_approx.function_space()
                f_exact = Function(V).interpolate(f_exact_ufl)
                error = errornorm(f_exact, f_approx, norm_type=norm_type)
                result.setdefault(key, []).append(error)

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
    ) -> Dict[str, float]:

        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        orders: Dict[str, float] = {}
        x_vals = df2.index.astype(float)
        cols = list(df2.columns)

        page = 0
        for start in range(0, len(cols), group_size):
            subset = cols[start : start + group_size]
            n_rows = int(np.ceil(len(subset) / 2))
            fig, axes = plt.subplots(n_rows, 2, figsize=(8.27, 11.69), squeeze=False)
            axes = axes.flatten()

            for ax, col in zip(axes, subset):
                y_vals = df2[col].values
                slope, _ = np.polyfit(np.log(x_vals), np.log(y_vals), 1)
                orders[col] = -slope

                ax.loglog(x_vals, y_vals, marker="o")
                ax.set_xlabel(x_label)
                ax.set_ylabel(f"{col}-error")
                ax.set_title(title_fmt.format(col=col, order=orders[col]))
                ax.grid(True, which="both")

            for ax in axes[len(subset) :]:
                fig.delaxes(ax)
            fig.tight_layout()

            namer = FileNamer(
                file_name=f"{file_stem}_page",
                folder_name=save_dir.name,
                path_name=str(save_dir.parent),
                mode="pdf",
            )
            fig.savefig(namer.file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            page += 1

        return orders

    def spatial_error_convergence(
        self,
        spatial_key: str,
        temporal_key: str,
        temporal_val: float,
        sweep_key: str,
        sweep_val: int,
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
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Fix n=spatial_val, sw=sweep_val and vary temporal_key.
        """
        idx = {spatial_key: spatial_val, sweep_key: sweep_val, **kwargs}
        df2 = self.df.xs(tuple(idx.values()), level=list(idx.keys()))
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
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Fix n=spatial_val, dt=temporal_val and vary sweep_key.
        """

        idx = {spatial_key: spatial_val, temporal_key: temporal_val, **kwargs}
        df2 = self.df.xs(tuple(idx.values()), level=list(idx.keys()))
        df2 = df2.sort_index()
        return self._plot_and_fit(
            df2,
            x_label=sweep_key,
            title_fmt=f"Sweep convergence (n={spatial_val}, dt={temporal_val}) — {{col}}: order≈{{order:.2f}}",
        )
