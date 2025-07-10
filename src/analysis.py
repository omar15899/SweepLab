import re
from pathlib import Path
from typing import List, Dict, Any
from firedrake import *
from filenamer import CheckpointAnalyser
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
    eliminarlas
    """

    def __init__(
        self,
        file_path: Path,
        pattern: re.Pattern,
        keys: str | List[str],
        keys_type: callable | List[callable],
        f_exact: Function | List[Function],
        function_names: List[str] = ["u"],
        get_function_characteristics: bool = False,
    ):
        super().__init__(
            self,
            file_path,
            pattern,
            keys,
            keys_type,
            function_names,
            get_function_characteristics,
        )
        self.f_exact = [f_exact] if not isinstance(f_exact, list) else f_exact
        self.df = self.create_error_table()

    @staticmethod
    def L2_error(f_exact, f_num):
        # x = SpatialCoordinate(mesh)[0]
        # V = FunctionSpace(mesh, "CG", degree=deg)
        # u_exact = Function(V).interpolate(sin(pi * x) * exp(x * tfin))
        return errornorm(f_exact, f_num, norm_type="L2")

    def create_error_table(self, norm_type="L2") -> pd.DataFrame:
        """
        Recorre todos los archivos que ya tenemos listados y extrae la función,
        por ahora lo vamos a hacer para el caso de una lista de 1.
        """
        result = {}
        for key, value in self.checkpoint_list.items():
            # Extract the functions:
            for f_approx, f_exact in zip(self.function_names, self.f_exact):
                error = errornorm(f_exact, f_approx, norm_type=norm_type)
                result[key] = result.setdefault(key, []).append(error)

        # We now convert to pandas dataframe in order to analyse it properly
        df = pd.DataFrame.from_dict(result, orient="index")
        # we convert it to multiindex
        df.aa = pd.MultiIndex.from_tuples(df.index, names=self.keys)
        df.columns = [fun.name() for fun in self.function_names]
        return df

    def _plot_and_fit(
        self,
        df2: pd.DataFrame,
        x_label: str,
        title_fmt: str,
        fitting_style: str | None,
    ) -> Dict[str, float]:
        """
        This function plots the functions
        fitting_style: "spatial", "temporal", "sweep"
        """
        orders: Dict[str, float] = {}
        x_vals = df2.index.astype(float)
        for col in df2.columns:
            y_vals = df2[col].values
            # Fit log-log line: ln(y) = slope*ln(x) + C
            slope, _ = np.polyfit(np.log(x_vals), np.log(y_vals), 1)
            order = -slope
            orders[col] = order
            # Plot
            plt.figure()
            plt.loglog(x_vals, y_vals, "o-", basex=10, basey=10)
            plt.xlabel(x_label)
            plt.ylabel(f"{col}-error")
            plt.title(title_fmt.format(col=col, order=order))
            plt.grid(True, which="both")
            plt.show()
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
