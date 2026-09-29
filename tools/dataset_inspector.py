from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv


class DatasetInspector:

    def __init__(
        self,
        raw_dir="data/raw",
        manifest_path="data/manifest.json"
    ):
        self.raw_dir = Path(raw_dir)
        self.manifest_path = Path(manifest_path)

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    # =========================================================
    # 1. 找到所有 VTU 文件
    # =========================================================

    def find_files(self) -> list[Path]:

        files = sorted(
            self.raw_dir.glob("*.vtu")
        )

        if not files:
            raise FileNotFoundError(
                f"{self.raw_dir} 中没有找到 .vtu 文件"
            )

        return files

    # =========================================================
    # 2. 从文件名中提取数字
    # =========================================================

    def extract_index(
        self,
        path: Path
    ) -> int:

        matches = re.findall(
            r"\d+",
            path.stem
        )

        if not matches:
            return 0

        return int(matches[-1])

    # =========================================================
    # 3. 按 snapshot 编号排序
    # =========================================================

    def sort_files(
        self,
        files: list[Path]
    ) -> list[Path]:

        return sorted(
            files,
            key=self.extract_index
        )

    # =========================================================
    # 4. 获取字段信息
    # =========================================================

    @staticmethod
    def field_info(
        data
    ) -> list[int]:

        shape = list(data.shape)

        if len(shape) == 1:
            return [1]

        return [shape[-1]]

    # =========================================================
    # 5. 检查一个 VTU
    # =========================================================

    def inspect_single(
        self,
        path: Path
    ) -> dict:

        mesh = pv.read(path)

        result = {

            "file":
                str(path),

            "n_points":
                int(mesh.n_points),

            "n_cells":
                int(mesh.n_cells),

            "bounds":
                [float(x) for x in mesh.bounds],

            "point_fields": {},

            "cell_fields": {},

            "has_nan": False,

            "has_inf": False

        }

        # -------------------------
        # Point Data
        # -------------------------

        for name in mesh.point_data.keys():

            data = np.asarray(
                mesh.point_data[name]
            )

            result[
                "point_fields"
            ][name] = self.field_info(
                data
            )

            if np.issubdtype(
                data.dtype,
                np.number
            ):

                result[
                    "has_nan"
                ] |= bool(
                    np.isnan(data).any()
                )

                result[
                    "has_inf"
                ] |= bool(
                    np.isinf(data).any()
                )

        # -------------------------
        # Cell Data
        # -------------------------

        for name in mesh.cell_data.keys():

            data = np.asarray(
                mesh.cell_data[name]
            )

            result[
                "cell_fields"
            ][name] = self.field_info(
                data
            )

            if np.issubdtype(
                data.dtype,
                np.number
            ):

                result[
                    "has_nan"
                ] |= bool(
                    np.isnan(data).any()
                )

                result[
                    "has_inf"
                ] |= bool(
                    np.isinf(data).any()
                )

        return result

    # =========================================================
    # 6. 比较两个网格是否一致
    # =========================================================

    def mesh_consistent(
        self,
        reference_path: Path,
        current_path: Path
    ) -> bool:

        ref = pv.read(
            reference_path
        )

        cur = pv.read(
            current_path
        )

        # 点数和单元数必须一致
        if ref.n_points != cur.n_points:
            return False

        if ref.n_cells != cur.n_cells:
            return False

        # 坐标允许极小浮点误差
        if not np.allclose(
            ref.points,
            cur.points,
            rtol=1e-10,
            atol=1e-10
        ):
            return False

        return True

    # =========================================================
    # 7. 比较字段结构
    # =========================================================

    @staticmethod
    def fields_consistent(
        reference: dict,
        current: dict
    ) -> bool:

        return (
            reference["point_fields"]
            ==
            current["point_fields"]
            and
            reference["cell_fields"]
            ==
            current["cell_fields"]
        )

    # =========================================================
    # 8. 尝试计算时间
    # =========================================================

    def infer_times(
        self,
        files: list[Path]
    ) -> dict:

        indices = [
            self.extract_index(path)
            for path in files
        ]

        if len(indices) < 2:
            return {
                "available": False,
                "reason": (
                    "快照数量不足，无法估计 dt"
                )
            }

        differences = np.diff(
            indices
        )

        # 如果编号是连续的
        if np.all(
            differences == differences[0]
        ):

            return {
                "available": True,
                "source": "filename_index",
                "index_step": int(
                    differences[0]
                ),
                "dt": None,
                "warning": (
                    "文件名只提供离散编号，"
                    "未发现物理时间尺度"
                )
            }

        return {
            "available": False,
            "reason": (
                "文件名编号不规则"
            )
        }

    # =========================================================
    # 9. 划分训练 / 测试
    # =========================================================

    @staticmethod
    def build_split(
        num_snapshots: int,
        train_ratio: float = 0.5
    ) -> dict:

        if num_snapshots < 2:
            raise ValueError(
                "至少需要两个 snapshot"
            )

        split_index = int(
            num_snapshots
            * train_ratio
        )

        if split_index <= 0:
            split_index = 1

        if split_index >= num_snapshots:
            split_index = (
                num_snapshots - 1
            )

        return {

            "train": {
                "start": 0,
                "end": split_index - 1,
                "count": split_index
            },

            "test": {
                "start": split_index,
                "end": num_snapshots - 1,
                "count": (
                    num_snapshots
                    - split_index
                )
            }
        }

    # =========================================================
    # 10. 主流程
    # =========================================================

    def inspect_dataset(self) -> dict:

        files = self.find_files()

        files = self.sort_files(files)

        print(
            f"发现 {len(files)} 个 VTU 文件"
        )

        # -------------------------
        # 检查第一个 snapshot
        # -------------------------

        reference_path = files[0]

        reference = self.inspect_single(
            reference_path
        )

        print(
            "参考网格：",
            reference_path.name
        )

        # -------------------------
        # 全数据统计
        # -------------------------

        snapshots = []

        mesh_consistent = True

        fields_consistent = True

        any_nan = False

        any_inf = False

        for i, path in enumerate(files):

            print(
                f"[{i + 1}/{len(files)}]"
                f" 检查 {path.name}"
            )

            info = self.inspect_single(
                path
            )

            snapshots.append(info)

            # 字段一致性
            if not self.fields_consistent(
                reference,
                info
            ):
                fields_consistent = False

            # 网格一致性
            if not self.mesh_consistent(
                reference_path,
                path
            ):
                mesh_consistent = False

            any_nan |= info["has_nan"]

            any_inf |= info["has_inf"]

        # -------------------------
        # 时间
        # -------------------------

        time_info = self.infer_times(
            files
        )

        # -------------------------
        # Train / Test
        # -------------------------

        split = self.build_split(
            len(files),
            train_ratio=0.5
        )

        # -------------------------
        # Manifest
        # -------------------------

        manifest = {

            "num_snapshots":
                len(files),

            "files": [
                str(path)
                for path in files
            ],

            "mesh": {

                "consistent":
                    mesh_consistent,

                "n_points":
                    reference["n_points"],

                "n_cells":
                    reference["n_cells"],

                "bounds":
                    reference["bounds"]
            },

            "fields": {

                "consistent":
                    fields_consistent,

                "point_data":
                    reference[
                        "point_fields"
                    ],

                "cell_data":
                    reference[
                        "cell_fields"
                    ]
            },

            "quality": {

                "has_nan":
                    any_nan,

                "has_inf":
                    any_inf
            },

            "time":
                time_info,

            "split":
                split
        }

        # -------------------------
        # 保存
        # -------------------------

        self.manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        return manifest