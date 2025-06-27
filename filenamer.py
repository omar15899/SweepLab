import os


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
