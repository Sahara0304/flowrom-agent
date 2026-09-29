from pathlib import Path

import pyvista as pv


class VTUTool:

    def inspect(self, path: str) -> dict:

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"找不到文件: {file_path}"
            )

        mesh = pv.read(file_path)

        point_fields = {}

        for name in mesh.point_data.keys():
            data = mesh.point_data[name]

            point_fields[name] = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
            }

        cell_fields = {}

        for name in mesh.cell_data.keys():
            data = mesh.cell_data[name]

            cell_fields[name] = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
            }

        return {
            "file": str(file_path),
            "n_points": int(mesh.n_points),
            "n_cells": int(mesh.n_cells),
            "bounds": list(mesh.bounds),
            "point_fields": point_fields,
            "cell_fields": cell_fields,
        }