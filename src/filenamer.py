import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Literal
from firedrake import *
from firedrake.checkpointing import CheckpointFile


class FileNamer:
    extensions = {
        "checkpoint": ".h5",
        "vtk": ".pvd",
        "pdf": ".pdf",
        "png": ".png",
        "txt": ".txt",
        "json": ".json",
    }

    def __init__(
        self,
        file_name: str = "solution",
        folder_name: str | None = None,
        path_name: str | None = None,
        mode: Literal["checkpoint", "vtk", "pdf"] = "checkpoint",
    ):

        # File saving attributes
        self.file_name = os.path.splitext(file_name)
        self.folder_name = folder_name if folder_name else "solution"
        self.path_name = path_name if path_name else os.getcwd()
        self.mode = mode
        self.file = self._create_unique_path()

        # if self.mode not in self.extensions.values():
        #     raise Exception("Invalid mode.")

    def _create_unique_path(self):
        """
        Create correct folder organisation.
        if vtk, we store the solution in different folders
        if chekcpoint, we store the solution in only one folder
        """
        base_name, _ = self.file_name
        base_dir = os.path.join(self.path_name, self.folder_name)
        os.makedirs(base_dir, exist_ok=True)
        if self.mode != "vtk":
            # files with no extension
            all_files = {
                os.path.splitext(name)[0]
                for name in os.listdir(os.path.join(self.path_name, self.folder_name))
            }

            # If the file is a checkpoint, we enumerate the files
            i = 0
            while True:
                file_name = f"{base_name}_{i}"
                if file_name not in all_files:
                    break
                i += 1

            return os.path.join(base_dir, file_name + self.extensions[self.mode])

        else:
            all_folders = {
                name
                for name in os.listdir(self.path_name)
                if os.path.isdir(self.path_name)
            }

            i = 0
            while True:
                folder_name = f"{self.folder_name}_{i}"
                if folder_name not in all_folders:
                    break
                i += 1

            return os.path.join(self.path_name, folder_name, base_name + ".pvd")


class CheckpointAnalyser:
    """
    holds a path, a regex, anda list of field names,
    so that each method can just refer to self.xxx.

    When defining the regex pattern is crutial to set
    some callable groups (not those with (?:) at the start)
    the main reason is that if we want to retrieve the real
    parameters for analysis purposes this is the best way
    of gettin it done.

    keys: they are the keys of the groups we have created
        in re expresion when we insert the pattern.

    keys_type: they are the types of the keys we have created
        in re expresion when we insert the pattern.

    function_names: the names of the fields we want to load
        from the checkpoint files, by default it is set to ["u"]
        ### ALL THE FILES NEED TO HAVE EXACTLY THE SAME AMMOUNT
        OF FUNCTIONS. IT HAS TO BE A SIMULATION IN WHICH THE ONLY
        THINGS WE HAVE VARIED ARE SOME PARAMETERS.

    Things to work on:
    - For now we assume we only have one mesh.
    - The getfuntioncaracteristics is only ment to be done
    if we are sure that we can store in ram all the results, if
    not we have to create a generator in order to call all the
    errors in the ConvergenceAnalyser

    """

    def __init__(
        self,
        file_path: Path,
        pattern: re.Pattern,
        keys: str | List[str],
        keys_type: callable | List[callable],
        function_names: str | List[str] = ["u"],
        get_function_characteristics: bool = False,
    ):
        self.file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self.pattern = (
            pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
        )
        self.keys = [keys] if not isinstance(keys, list) else keys
        self.keys_type: List[Callable] = (
            keys_type if isinstance(keys_type, list) else [keys_type]
        )
        self.function_names = (
            function_names if isinstance(function_names, list) else [function_names]
        )
        self.checkpoint_list = self.list_checkpoints()
        if not self.checkpoint_list:
            raise Exception("No files to explore.")
        self._mesh = None

        if get_function_characteristics and self.checkpoint_list:
            self.mesh, self.V, self.t_end, self.idx, self.f_approx = (
                [],
                [],
                [],
                [],
                [],
            )
            first_file = next(iter(self.checkpoint_list.values()))[0]
            for f_name in self.function_names:
                mesh, _, t_end, idx, func = (
                    CheckpointAnalyser.load_function_from_checkpoint(
                        self.checkpoint_list[0], f_name
                    )
                )

                self.mesh.append(mesh)
                self.V = func.function_space()
                self.f_approx = func
                self.t_end.append(t_end)
                self.idx.append(idx)

    def list_checkpoints(self) -> dict:
        """
        no args needed here, uses self.file_path & self.pattern
        """
        result = {}
        for file in self.file_path.glob("*.h5"):
            m = self.pattern.match(file.name)
            if not m:
                continue
            # key = (int(m["n"]), float(m["dt"]), int(m["sw"]))
            key = tuple(
                typename(m.group(name))
                for typename, name in zip(self.keys_type, self.keys)
            )
            idx = int(m["idx"] or 0)
            if key not in result or idx > result[key][1]:
                result[key] = (file, idx)
        return result

    def _mesh_from_file(self, file: CheckpointFile):
        if self._mesh is None:
            self._mesh = file.load_mesh()
        return self._mesh

    def load_function_from_checkpoint(self, filepath: Path, function_name: str):
        """
        load a single field (defaulting to the first in self.field_names)
        """
        with CheckpointFile(str(filepath), "r") as file:
            mesh = file.load_mesh()
            hist = file.get_timestepping_history(mesh, function_name)
            idx = hist["index"][-1]
            t_end = hist["time"][-1]
            func = file.load_function(mesh, function_name, idx=idx)
        return mesh, hist, t_end, idx, func

    @staticmethod
    def load_multiple_functions_from_checkpoint(
        filepath: Path,
        function_names: List[str],
    ):
        """
        loads every name in self.field_names
        """
        data = {"mesh": None, "fields": {}}
        with CheckpointFile(str(filepath), "r") as f:
            mesh = f.load_mesh()
            data["mesh"] = mesh
            for name in function_names:
                hist = f.get_timestepping_history(mesh, name)
                idx = hist["index"][-1]
                t_end = hist["time"][-1]
                func = f.load_function(mesh, name, idx=idx)
                data["fields"][name] = {
                    "hist": hist,
                    "idx": idx,
                    "t_end": t_end,
                    "func": func,
                }
        return data
