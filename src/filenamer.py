import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from firedrake import *
from firedrake.checkpointing import CheckpointFile


class FileNamer:
    def __init__(
        self,
        file_name: str = "solution",
        folder_name: str | None = None,
        path_name: str | None = None,
        is_vtk: bool = False,
    ):

        self.is_vtk = is_vtk
        self.is_checkpoint = True if not is_vtk else False
        # File saving attributes
        self.file_name = os.path.splitext(file_name)
        self.folder_name = folder_name if folder_name else "solution"
        self.path_name = path_name if path_name else os.getcwd()
        self.file = self._create_unique_path()

    def _create_unique_path(self):
        """
        Create correct folder organisation.
        if vtk, we store the solution in different folders
        if chekcpoint, we store the solution in only one folder
        """
        base_name, _ = self.file_name
        base_dir = os.path.join(self.path_name, self.folder_name)
        os.makedirs(base_dir, exist_ok=True)
        if self.is_checkpoint:
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

            return os.path.join(base_dir, file_name + ".h5")

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

    """

    def __init__(
        self,
        file_path: Path,
        pattern: re.Pattern,
        keys: str | List[str],
        keys_type: callable | List[callable],
        function_names: List[str] = ["u"],
        get_function_characteristics: bool = False,
    ):
        self.file_path = file_path
        self.pattern = re.compile(pattern)
        self.keys = [keys] if keys is not isinstance(keys, list) else keys
        self.keys_type = [keys_type] if not isinstance(keys_type, list) else keys_type
        self.function_names = (
            [function_names] if not isinstance(function_names, list) else function_names
        )
        self.checkpoint_list = self.list_checkpoints()
        # Print all the spaces V in which the solutions live in order to use them for the
        # exact solution
        self.mesh = [] if get_function_characteristics else None
        self.V = [] if get_function_characteristics else None
        self.t_end = [] if get_function_characteristics else None
        self.idx = [] if get_function_characteristics else None
        self.f_approx = [] if get_function_characteristics else None

        if get_function_characteristics:
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
        for f in self.file_path.glob("*.h5"):
            m = self.pattern.match(f.name)
            if not m:
                continue
            # key = (int(m["n"]), float(m["dt"]), int(m["sw"]))
            key = tuple(
                typename(m.group(name))
                for typename, name in zip(self.keys_type, self.keys)
            )
            idx = int(m["idx"] or 0)
            if key not in result or idx > result[key][1]:
                result[key] = (f, idx)
        return result

    def _mesh_from_file(self, chk: CheckpointFile):
        if self._mesh is None:
            self._mesh = chk.load_mesh()
        return self._mesh

    def load_function_from_checkpoint(self, filepath: Path, function_name: str):
        """
        load a single field (defaulting to the first in self.field_names)
        """
        with CheckpointFile(str(filepath), "r") as file:
            mesh = self._mesh_from_file(file)
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
