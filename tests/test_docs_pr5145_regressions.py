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

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"


def _read(relative_path):
    return (DOCS_ROOT / relative_path).read_text(encoding="utf-8")


def test_docs_do_not_pass_removed_use_common_ocr_argument():
    offenders = []
    for path in DOCS_ROOT.rglob("*.md"):
        if "use_common_ocr=" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(REPO_ROOT))

    assert offenders == []


def test_pp_structure_v3_example_has_valid_keyword_separator():
    text = _read("pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.en.md")

    assert 'input="./pp_structure_v3_demo.png",,' not in text


def test_pp_chatocr_model_paths_include_the_full_config_hierarchy():
    layout_path = "SubPipelines.LayoutParser.SubModules.LayoutDetection.model_dir"
    seal_path = (
        "SubPipelines.LayoutParser.SubPipelines.SealRecognition."
        "SubPipelines.SealOCR.SubModules.TextDetection.model_dir"
    )
    layout_docs = (
        "practical_tutorials/"
        "document_scene_information_extraction(layout_detection)_tutorial"
    )
    seal_docs = (
        "practical_tutorials/"
        "document_scene_information_extraction(seal_recognition)_tutorial"
    )

    layout_snippet = (
        "SubPipelines:\n"
        "  LayoutParser:\n"
        "    # ...\n"
        "    SubModules:\n"
        "      LayoutDetection:"
    )
    seal_snippet = (
        "SubPipelines:\n"
        "  LayoutParser:\n"
        "    # ...\n"
        "    SubPipelines:\n"
        "      SealRecognition:\n"
        "        # ...\n"
        "        SubPipelines:\n"
        "          SealOCR:\n"
        "            # ...\n"
        "            SubModules:\n"
        "              TextDetection:"
    )

    for suffix in (".md", ".en.md"):
        layout_text = _read(f"{layout_docs}{suffix}")
        seal_text = _read(f"{seal_docs}{suffix}")
        assert layout_path in layout_text
        assert layout_snippet in layout_text
        assert seal_path in seal_text
        assert seal_snippet in seal_text
