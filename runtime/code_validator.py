from __future__ import annotations

import ast
from pathlib import Path


class CodeValidationError(ValueError):
    pass


# ============================================================
# Allowed imports for V1
# ============================================================

ALLOWED_IMPORTS = {
    "numpy",
    "math",
    "typing",
    "dataclasses",
}


FORBIDDEN_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "shutil",
    "glob",
    "pickle",
    "joblib",
    "torch",
    "tensorflow",
    "jax",
}


FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
}


REQUIRED_METHODS = {
    "__init__",
    "fit",
    "predict_next",
    "rollout",
}


def validate_imports(
    tree: ast.AST,
) -> list[str]:

    problems = []

    for node in ast.walk(tree):

        if isinstance(
            node,
            ast.Import,
        ):

            for alias in node.names:

                module = (
                    alias.name.split(".")[0]
                )

                if module in FORBIDDEN_IMPORTS:

                    problems.append(
                        f"禁止导入模块: {module}"
                    )

                elif module not in ALLOWED_IMPORTS:

                    problems.append(
                        f"未允许的第三方模块: "
                        f"{module}"
                    )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            module = (
                node.module
                or ""
            )

            root = (
                module.split(".")[0]
            )

            if root in FORBIDDEN_IMPORTS:

                problems.append(
                    f"禁止导入模块: {module}"
                )

            elif root not in ALLOWED_IMPORTS:

                problems.append(
                    f"未允许的模块: {module}"
                )

    return problems


def validate_calls(
    tree: ast.AST,
) -> list[str]:

    problems = []

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        # func(...)
        if isinstance(
            node.func,
            ast.Name,
        ):

            name = node.func.id

            if name in FORBIDDEN_CALLS:

                problems.append(
                    f"禁止调用: {name}"
                )

        # os.system(...)
        if isinstance(
            node.func,
            ast.Attribute,
        ):

            if (
                node.func.attr
                in FORBIDDEN_CALLS
            ):

                problems.append(
                    f"禁止调用: "
                    f"{node.func.attr}"
                )

    return problems


def validate_file_access(
    tree: ast.AST,
) -> list[str]:

    problems = []

    forbidden_names = {
        "final_test",
        "snapshot_401",
        "snapshot401",
        "data_raw",
    }

    for node in ast.walk(tree):

        if isinstance(
            node,
            ast.Constant,
        ):

            if isinstance(
                node.value,
                str,
            ):

                value = (
                    node.value.lower()
                )

                for forbidden in (
                    forbidden_names
                ):

                    if forbidden in value:

                        problems.append(
                            "代码中出现禁止的数据引用: "
                            f"{forbidden}"
                        )

    return problems


def find_candidate_class(
    tree: ast.AST,
) -> ast.ClassDef:

    classes = [
        node
        for node in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
        and node.name == "CandidateModel"
    ]

    if not classes:

        raise CodeValidationError(
            "代码必须定义 CandidateModel 类。"
        )

    if len(classes) > 1:

        raise CodeValidationError(
            "只能定义一个 CandidateModel 类。"
        )

    return classes[0]


def validate_candidate_interface(
    candidate_class: ast.ClassDef,
) -> list[str]:

    problems = []

    methods = {
        node.name
        for node in candidate_class.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    }

    missing = (
        REQUIRED_METHODS
        - methods
    )

    for method in sorted(
        missing
    ):

        problems.append(
            "CandidateModel 缺少方法: "
            f"{method}"
        )

    return problems


def validate_code(
    code: str,
) -> dict:

    if not code.strip():

        raise CodeValidationError(
            "代码为空。"
        )

    try:

        tree = ast.parse(
            code
        )

    except SyntaxError as exc:

        raise CodeValidationError(
            "Python syntax error:\n"
            f"{exc}"
        ) from exc

    problems = []

    problems.extend(
        validate_imports(
            tree
        )
    )

    problems.extend(
        validate_calls(
            tree
        )
    )

    problems.extend(
        validate_file_access(
            tree
        )
    )

    candidate_class = (
        find_candidate_class(
            tree
        )
    )

    problems.extend(
        validate_candidate_interface(
            candidate_class
        )
    )

    if problems:

        raise CodeValidationError(
            "\n".join(
                f"- {problem}"
                for problem in problems
            )
        )

    return {
        "status": "valid",
        "required_methods": sorted(
            REQUIRED_METHODS
        ),
        "candidate_class": (
            "CandidateModel"
        ),
    }


def save_validated_code(
    code: str,
    path: Path,
) -> dict:

    result = validate_code(
        code
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        code,
        encoding="utf-8",
    )

    return result