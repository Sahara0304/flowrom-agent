from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# 将项目根目录加入 Python 模块搜索路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 现在再导入项目内部模块
# ============================================================

import argparse
import json

from tools.pod_analysis import (
    discover_vtu_files,
    list_available_fields,
    build_snapshot_matrix,
    compute_pod_spectrum,
)


def main():

    parser = argparse.ArgumentParser(
        description="VTU Snapshot -> POD spectrum"
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
    )

    parser.add_argument(
        "--field",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--location",
        type=str,
        choices=["point", "cell"],
        default="point",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "all",
            "magnitude",
            "components",
        ],
        default="all",
    )

    parser.add_argument(
        "--components",
        type=int,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--development-ratio",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--list-fields",
        action="store_true",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    files = discover_vtu_files(raw_dir)

    print()
    print("=" * 60)
    print("VTU DATASET")
    print("=" * 60)

    print(f"Snapshot 数量: {len(files)}")

    print(
        f"编号范围: "
        f"{files[0][0]} -> {files[-1][0]}"
    )

    # --------------------------------------------------------
    # 只查看字段
    # --------------------------------------------------------

    if args.list_fields:

        fields = list_available_fields(
            files[0][1]
        )

        print()
        print("Point Data:")

        for field in fields["point_data"]:
            print("  -", field)

        print()
        print("Cell Data:")

        for field in fields["cell_data"]:
            print("  -", field)

        return

    if args.field is None:

        print()
        print("尚未指定 --field。")

        print(
            "请先运行："
        )

        print(
            "python scripts/run_pod.py "
            "--raw-dir data/raw "
            "--list-fields"
        )

        return

    # --------------------------------------------------------
    # 建立 Snapshot Matrix
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BUILD SNAPSHOT MATRIX")
    print("=" * 60)

    metadata = build_snapshot_matrix(
        raw_dir=raw_dir,
        output_dir=output_dir,
        field_name=args.field,
        location=args.location,
        mode=args.mode,
        components=args.components,
        development_ratio=args.development_ratio,
    )

    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # POD
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("POD ANALYSIS")
    print("=" * 60)

    result = compute_pod_spectrum(
        matrix_path=Path(
            metadata["matrix_path"]
        ),
        mean_path=Path(
            metadata["mean_path"]
        ),
        output_dir=output_dir,
    )

    print(
        f"矩阵尺寸: "
        f"{result['matrix_shape']}"
    )

    print(
        f"Effective Rank: "
        f"{result['effective_rank']:.3f}"
    )

    print()
    print(
        "不同能量阈值所需 POD 模态数:"
    )

    for threshold, rank in (
        result["modes_required"].items()
    ):
        print(
            f"  {threshold}: {rank}"
        )

    print()

    print(
        "POD 结果已保存到: "
        f"{output_dir / 'pod_spectrum.json'}"
    )


if __name__ == "__main__":
    main()