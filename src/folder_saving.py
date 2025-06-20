from typing import List, Optional
import os
import inspect
import pandas as pd


class FolderSave:
    def __init__(
        self,
        df: pd.DataFrame,
        directory: str = os.getcwd(),
        name_folder: str = "Fichero_PreScrapping",
    ) -> None:
        """
        La variable directory es opcional, sirve para indicar dónde queremos que
        se generen las carpetas. En caso de que no escribamos nada las carpetas
        se generaran en el directorio desde donde se esté ejecutando el terminal.

        Se recomienta que directory se ponga de forma manual para hacer la instalación
        pertinente de los ficheros.

        La variable df puede ser la variable df en crudo que ha diseñado Iván o
        el df ya tratado (que haya pasado por _preparacion_datos y geocodificator).

        Todo se almacena en df por no enrevesar las cosas. No necesitamos mucho más.
        """
        self.df = df
        self.directory = directory
        self.name_folder = name_folder
        # Inicializamos la función principal
        self._preparacion_datos()

    # --------------------------------------------------------------------------------------- Métodos para crear archivos
    """
    Estos métodos van a ser cruciales no solo en esta parte del código, si no también en el 
    scrapping puesto que estamos sistematizando una forma de crear archivos y carpetas en la 
    ubicación del script que nos va a permitir generar todas las carpetas que nos den la gana
    de una forma fácil y sencilla.
    """

    @staticmethod
    def _buscar_directorio_script() -> str | None:
        """
        Función que devuelve el directorio del script que llama a esta clase
        se encuentra ubicado. Para ello buscamos el frame o call stack de la
        función a la que está llamando dentro del script y de ahí substraemos
        la ubicación completa del script.

        Returns:
            str | None: La ruta del directorio del script que llama a esta función,
            o None si no se puede determinar.
        """
        # Obtenemos el marco de llamada del antecesor del script actual:
        marco = inspect.currentframe()
        ruta_script = inspect.getfile(marco.f_back)

        # Obtenemos la ruta del directorio donde se encuentra el script:
        directorio_script = os.path.dirname(ruta_script)

        return directorio_script

    @staticmethod
    def _crear_carpeta_archivo_en_ubicacion_script(
        nombre_carpeta: str,
        nombre_directorio: str = os.path.dirname(__file__),
        nombre_archivo: Optional[str] = None,
        contenido_archivo: Optional[str | List | pd.DataFrame] = None,
    ) -> str | None:
        """
        Args:
            nombre_carpeta (str): El nombre de la carpeta a buscar o crear.
            nombre_directorio (str, optional): El directorio base donde buscar o crear la carpeta.
                Por defecto, es el directorio del archivo actual.
            nombre_archivo (str | None, optional): El nombre del archivo a crear dentro de la carpeta.
                Por defecto, no se crea ningún archivo.
            contenido_archivo (str | List | pd.DataFrame | None, optional): El contenido del archivo a crear.
                Puede ser una cadena de texto, una lista, un DataFrame de pandas o None.
                Por defecto, no se crea ningún archivo.

        Returns:
            str | None: La ruta completa del archivo creado, si se creó un archivo. None en caso contrario.

        Busca una carpeta en el directorio donde se está ejecutando el script
        con nombre nombre_carpeta y si no la encuentra crea una. Si nombre_archivo
        es diferente a None creará un archivo además dentro de esa carpeta. Primero
        intentará con el nombre del archivo, si no con el sufijo _1,...,_n.
        """
        # Importamos el directorio del script
        # directorio_script = PreScrapping._buscar_directorio_script()

        # Construimos la ruta completa de la carpeta
        ruta_carpeta = os.path.join(nombre_directorio, nombre_carpeta)

        # Creamos la carpeta si no existe
        if not os.path.exists(ruta_carpeta):
            os.makedirs(ruta_carpeta)

        if nombre_archivo:
            ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
            contador = 1

            # Extraemos la extensión y el nombre base del archivo
            nombre_base, extension = os.path.splitext(nombre_archivo)

            # Mientras el archivo exista, modificamos el nombre
            while os.path.exists(ruta_archivo):
                ruta_archivo = os.path.join(
                    ruta_carpeta, f"{nombre_base}_{contador}{extension}"
                )
                contador += 1

            # Si es un archivo importable desde pandas, directamente generamos el df
            if extension == ".xlsx" and isinstance(contenido_archivo, pd.DataFrame):
                contenido_archivo.to_excel(ruta_archivo)
            elif extension == ".csv" and isinstance(contenido_archivo, pd.DataFrame):
                contenido_archivo.to_csv(ruta_archivo, index=False)
            elif extension == ".json" and isinstance(contenido_archivo, pd.DataFrame):
                contenido_archivo.to_json(ruta_archivo, orient="records", lines=True)
            elif extension == ".pkl" and isinstance(contenido_archivo, pd.DataFrame):
                contenido_archivo.to_pickle(ruta_archivo)
            elif not isinstance(contenido_archivo, pd.DataFrame):
                # Si no es un df, creamos el archivo (vacío) en la ruta final para que
                # solo tengamos que esribir el archivo desde fuera. La ventaja es  que
                # estará en una carpeta y con un nombre a todas las versiones previas.
                # Además, como se ha usado el with open as, el archivo se cierra solo,
                # si no lo tendríamos que explicitar en este código como archivo.close
                # antes del pass.
                with open(ruta_archivo, "w") as archivo:
                    pass
                # En caso de que solo se cree devolvermos el nombre del archivo para
                # poder escribirlo a continuación:
                return ruta_archivo
