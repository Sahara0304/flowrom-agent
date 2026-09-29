from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyvista as pv


# ============================================================
# 文件编号
# ============================================================

def extract_snapshot_index(path: Path) -> int:
    """
    从文件名中提取最后一个数字作为 snapshot index。

    例如：
        flow_1.vtu        -> 1
        snapshot_025.vtu  -> 25
        case_A_800.vtu    -> 800
    """
    matches = re.findall(r"\d+(?=\D*$)", path.stem)

    if not matches:
        raise ValueError(
            f"无法从文件名中识别 snapshot 编号: {path.name}"
        )

    return int(matches[0])


def discover_vtu_files(raw_dir: Path) -> list[tuple[int, Path]]:
    """
    自动发现 raw_dir 中所有 VTU 文件，并按照编号排序。
    """
    files = []

    for path in raw_dir.glob("*.vtu"):
        try:
            idx = extract_snapshot_index(path)
            files.append((idx, path))
        except ValueError:
            # 没有编号的文件忽略
            continue

    if not files:
        raise FileNotFoundError(
            f"{raw_dir} 中没有找到带编号的 .vtu 文件"
        )

    files.sort(key=lambda x: x[0])

    indices = [idx for idx, _ in files]

    if len(indices) != len(set(indices)):
        raise ValueError("发现重复的 snapshot 编号。")

    return files


# ============================================================
# VTU 字段
# ============================================================

def list_available_fields(
    path: Path,
) -> dict[str, list[str]]:
    """
    查看一个 VTU 中有哪些 point_data / cell_data。
    """
    mesh = pv.read(path)

    return {
        "point_data": list(mesh.point_data.keys()),
        "cell_data": list(mesh.cell_data.keys()),
    }


def read_field(
    path: Path,
    field_name: str,
    location: str = "point",
    mode: str = "all",
    components: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """
    读取一个 VTU 字段，并展平成一维向量。

    参数
    ----
    location:
        "point" 或 "cell"

    mode:
        "all"       : 保留全部分量
        "magnitude" : 如果为向量场，转换成模长
        "components": 使用指定 components

    返回
    ----
    shape = (n_dof,)
    """

    mesh = pv.read(path)

    if location == "point":
        data = mesh.point_data
    elif location == "cell":
        data = mesh.cell_data
    else:
        raise ValueError(
            "location 必须是 'point' 或 'cell'"
        )

    if field_name not in data:
        available = list(data.keys())

        raise KeyError(
            f"{path.name} 中不存在字段 '{field_name}'。\n"
            f"可用字段: {available}"
        )

    arr = np.asarray(data[field_name])

    # 标量场
    if arr.ndim == 1:
        return arr.astype(np.float64).reshape(-1)

    # 向量/多分量场
    if arr.ndim != 2:
        raise ValueError(
            f"暂不支持字段 {field_name} 的 shape={arr.shape}"
        )

    if mode == "all":
        selected = arr

    elif mode == "magnitude":
        selected = np.linalg.norm(arr, axis=1)

    elif mode == "components":
        if components is None:
            raise ValueError(
                "mode='components' 时必须提供 components"
            )

        components = list(components)

        selected = arr[:, components]

    else:
        raise ValueError(
            "mode 必须是 'all'、'magnitude' 或 'components'"
        )

    return np.asarray(selected, dtype=np.float64).reshape(-1)


# ============================================================
# Snapshot Matrix
# ============================================================

def build_snapshot_matrix(
    raw_dir: Path,
    output_dir: Path,
    field_name: str,
    location: str = "point",
    mode: str = "all",
    components: Optional[Iterable[int]] = None,
    development_ratio: float = 0.5,
) -> dict:
    """
    构造：

        X = [x_1, ..., x_N]

    这里只使用 development 数据。

    返回 metadata。
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = discover_vtu_files(raw_dir)

    n_total = len(files)

    if n_total < 2:
        raise ValueError(
            "snapshot 数量太少，无法进行 POD 分析。"
        )

    if not (0 < development_ratio <= 1):
        raise ValueError(
            "development_ratio 必须在 (0, 1] 内。"
        )

    n_dev = int(n_total * development_ratio)

    development_files = files[:n_dev]

    first_idx = development_files[0][0]
    last_idx = development_files[-1][0]

    # --------------------------------------------------------
    # 第一份 snapshot 确定自由度
    # --------------------------------------------------------

    first_vector = read_field(
        development_files[0][1],
        field_name=field_name,
        location=location,
        mode=mode,
        components=components,
    )

    n_dof = first_vector.size

    # --------------------------------------------------------
    # 使用 npy memmap
    # 避免把整个巨型矩阵强行塞入 RAM
    # --------------------------------------------------------

    matrix_path = (
        output_dir /
        "snapshot_matrix_development.npy"
    )

    X = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float64,
        shape=(n_dof, n_dev),
    )

    X[:, 0] = first_vector

    for j, (idx, path) in enumerate(
        development_files[1:],
        start=1,
    ):

        vector = read_field(
            path,
            field_name=field_name,
            location=location,
            mode=mode,
            components=components,
        )

        if vector.size != n_dof:
            raise ValueError(
                f"snapshot {idx} 自由度数量不一致："
                f"{vector.size} != {n_dof}"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"snapshot {idx} 存在 NaN / Inf。"
            )

        X[:, j] = vector

    X.flush()

    # --------------------------------------------------------
    # 计算均值
    # --------------------------------------------------------

    mean_field = np.asarray(
        X.mean(axis=1),
        dtype=np.float64,
    )

    mean_path = (
        output_dir /
        "snapshot_mean.npy"
    )

    np.save(mean_path, mean_field)

    metadata = {
        "field_name": field_name,
        "location": location,
        "mode": mode,
        "components": (
            list(components)
            if components is not None
            else None
        ),
        "num_total_snapshots": n_total,
        "num_development_snapshots": n_dev,
        "development_ratio": development_ratio,
        "snapshot_index_start": first_idx,
        "snapshot_index_end": last_idx,
        "matrix_shape": [n_dof, n_dev],
        "matrix_definition": "X[:, j] = x_j",
        "mean_centered": True,
        "matrix_path": str(matrix_path),
        "mean_path": str(mean_path),
    }

    metadata_path = (
        output_dir /
        "snapshot_matrix_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metadata


# ============================================================
# Method of Snapshots POD
# ============================================================

def compute_gram_matrix(
    X: np.ndarray,
    mean: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    """
    计算：

        G = X_c^T X_c

    使用 block 方式，避免一次性构造完整 X_c。
    """

    n_dof, n_snapshots = X.shape

    G = np.zeros(
        (n_snapshots, n_snapshots),
        dtype=np.float64,
    )

    for i in range(
        0,
        n_snapshots,
        block_size,
    ):
        i_end = min(
            i + block_size,
            n_snapshots,
        )

        A = np.asarray(
            X[:, i:i_end],
            dtype=np.float64,
            copy=True,
        )

        A -= mean[:, None]

        for j in range(
            i,
            n_snapshots,
            block_size,
        ):
            j_end = min(
                j + block_size,
                n_snapshots,
            )

            B = np.asarray(
                X[:, j:j_end],
                dtype=np.float64,
                copy=True,
            )

            B -= mean[:, None]

            block = A.T @ B

            G[i:i_end, j:j_end] = block

            if i != j:
                G[j:j_end, i:i_end] = block.T

    return G


def compute_pod_spectrum(
    matrix_path: Path,
    mean_path: Path,
    output_dir: Path,
) -> dict:
    """
    基于 method of snapshots 计算 POD singular spectrum。
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = np.load(
        matrix_path,
        mmap_mode="r",
    )

    mean = np.load(mean_path)

    n_dof, n_snapshots = X.shape

    # --------------------------------------------------------
    # Gram matrix
    # --------------------------------------------------------

    G = compute_gram_matrix(
        X,
        mean,
    )

    # 数值对称化
    G = 0.5 * (G + G.T)

    # --------------------------------------------------------
    # 特征值分解
    # --------------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(G)

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 消除非常小的负数数值误差
    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    singular_values = np.sqrt(
        eigenvalues
    )

    energy = eigenvalues.copy()

    total_energy = energy.sum()

    if total_energy <= 0:
        raise ValueError(
            "总 POD 能量为 0，无法进行谱分析。"
        )

    energy_ratio = (
        energy / total_energy
    )

    cumulative_energy = np.cumsum(
        energy_ratio
    )

    # --------------------------------------------------------
    # 有效秩
    # --------------------------------------------------------

    p = energy_ratio[
        energy_ratio > 1e-15
    ]

    effective_rank = float(
        np.exp(
            -np.sum(
                p * np.log(p)
            )
        )
    )

    # --------------------------------------------------------
    # 不同精度需要多少 mode
    # --------------------------------------------------------

    thresholds = [
        0.90,
        0.95,
        0.99,
        0.999,
    ]

    modes_required = {}

    for threshold in thresholds:

        rank = int(
            np.searchsorted(
                cumulative_energy,
                threshold,
            )
            + 1
        )

        modes_required[
            str(threshold)
        ] = rank
    # --------------------------------------------------------
    # 保存 temporal POD basis
    #
    # V[:, i] 是第 i 个时间方向
    # 后续 latent coordinate:
    #
    # Z = Sigma_r @ V_r.T
    #
    # --------------------------------------------------------

    temporal_basis_path = (
        output_dir /
        "pod_temporal_basis.npy"
    )

    np.save(
        temporal_basis_path,
        eigenvectors,
    )
    result = {
        "matrix_shape": [
            int(n_dof),
            int(n_snapshots),
        ],
        "singular_values": (
            singular_values.tolist()
        ),
        "energy_ratio": (
            energy_ratio.tolist()
        ),
        "cumulative_energy": (
            cumulative_energy.tolist()
        ),
        "effective_rank": effective_rank,
        "modes_required": modes_required,
        "temporal_basis_path": str(
            temporal_basis_path
        ),
    }

    result_path = (
        output_dir /
        "pod_spectrum.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return result