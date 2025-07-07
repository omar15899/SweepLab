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
    holds a path, a regex, and/or a list of field names,
    so that each method can just refer to self.xxx
    """

    def __init__(self, file_path: Path, pattern: str, field_names: List[str] = ["u"]):
        self.file_path = file_path
        self.pattern = re.compile(pattern)
        self.field_names = field_names

    def list_checkpoints(self) -> dict:
        """
        no args needed here, uses self.file_path & self.pattern
        """
        out = {}
        for f in self.file_path.glob("*.h5"):
            m = self.pattern.match(f.name)
            if not m:
                continue
            key = (int(m["n"]), float(m["dt"]), int(m["sw"]))
            idx = int(m["idx"] or 0)
            if key not in out or idx > out[key][1]:
                out[key] = (f, idx)
        return out

    def load_checkpoint(self, filepath: Path, field: str = None):
        """
        load a single field (defaulting to the first in self.field_names)
        """
        field = field or self.field_names[0]
        with CheckpointFile(str(filepath), "r") as f:
            mesh = f.load_mesh()
            hist = f.get_timestepping_history(mesh, field)
            idx = hist["index"][-1]
            t_end = hist["time"][-1]
            func = f.load_function(mesh, field, idx=idx)
        return mesh, hist, t_end, idx, func

    def load_checkpoint_multiple_f(self, filepath: Path):
        """
        loads every name in self.field_names
        """
        data = {"mesh": None, "fields": {}}
        with CheckpointFile(str(filepath), "r") as f:
            mesh = f.load_mesh()
            data["mesh"] = mesh
            for name in self.field_names:
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
