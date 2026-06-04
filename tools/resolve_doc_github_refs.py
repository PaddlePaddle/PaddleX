#!/usr/bin/env python3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
from pathlib import Path


TEXT_SUFFIXES = {".md", ".yml", ".yaml"}


def resolve_placeholders(root, placeholder, source_ref):
    if (
        not source_ref
        or source_ref.strip() != source_ref
        or any(c.isspace() for c in source_ref)
    ):
        raise ValueError("source_ref must be a non-empty ref without whitespace")

    root = Path(root)
    changed = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8")
        if placeholder not in content:
            continue
        path.write_text(content.replace(placeholder, source_ref), encoding="utf-8")
        changed.append(path)
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Resolve docs GitHub source-ref placeholders before building docs."
    )
    parser.add_argument("--root", default="docs", help="Directory to rewrite.")
    parser.add_argument("--placeholder", required=True)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args(argv)

    changed = resolve_placeholders(
        args.root,
        placeholder=args.placeholder,
        source_ref=args.source_ref,
    )
    print(f"Resolved {len(changed)} file(s) under {args.root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
