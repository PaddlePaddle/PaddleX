# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static check: flag PEP 585 (``list[...]``/``tuple[...]``/...) and PEP 604
(``X | Y``) usage in runtime-evaluated annotations when the file lacks
``from __future__ import annotations``.

Without the ``__future__`` import, these forms are evaluated at function
definition time and raise ``TypeError`` on Python 3.8 at import — which is
still a supported target for PaddleX.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from typing import Iterator, List, Tuple

PEP585_BUILTINS = frozenset({"list", "tuple", "dict", "set", "frozenset", "type"})


def has_future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return True
    return False


def iter_annotations(tree: ast.AST) -> Iterator[ast.AST]:
    """Yield every AST node that is a runtime-evaluated annotation."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.returns
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if arg.annotation is not None:
                    yield arg.annotation
            if args.vararg is not None and args.vararg.annotation is not None:
                yield args.vararg.annotation
            if args.kwarg is not None and args.kwarg.annotation is not None:
                yield args.kwarg.annotation


def find_issues(annotation: ast.AST) -> List[Tuple[int, int, str]]:
    issues: List[Tuple[int, int, str]] = []
    for node in ast.walk(annotation):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            if name in PEP585_BUILTINS:
                issues.append(
                    (
                        node.lineno,
                        node.col_offset,
                        f"PEP 585 builtin generic `{name}[...]`",
                    )
                )
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            issues.append((node.lineno, node.col_offset, "PEP 604 union `X | Y`"))
    return issues


def check(file_path: str) -> bool:
    src = pathlib.Path(file_path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=file_path)
    except SyntaxError as e:
        print(f"{file_path}:{e.lineno}:{e.offset}: failed to parse: {e.msg}")
        return False

    if has_future_annotations(tree):
        return True

    ok = True
    for annotation in iter_annotations(tree):
        for line, col, msg in find_issues(annotation):
            print(
                f"{file_path}:{line}:{col}: {msg} — "
                "add `from __future__ import annotations` for Py3.8 support"
            )
            ok = False
    return ok


def main() -> int:
    files = sys.argv[1:]
    ok = True
    for f in files:
        if not check(f):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
