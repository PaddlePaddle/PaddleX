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

"""Regression tests for per-page namespacing of Markdown image paths.

``save_to_markdown`` derives image filenames from (label, box) only, so a
figure at the same coordinates on different pages of one document used to
resolve to the same path and later pages silently overwrote earlier ones.
``MarkdownConverter.convert(..., page_index=...)`` now namespaces the image
paths of pages after the first.
"""

from paddlex.inference.common.result.converter.markdown_converter import (
    MarkdownConverter,
)


def _convert(imgs_in_doc, page_index):
    return MarkdownConverter.convert(
        [],
        handle_funcs_dict={},
        imgs_in_doc=imgs_in_doc,
        page_index=page_index,
    )


def test_same_coordinates_on_different_pages_do_not_collide():
    imgs = [{"path": "imgs/img_in_image_box_0_0_100_100.jpg", "img": "DATA"}]
    page1 = _convert(imgs, page_index=1)
    page2 = _convert(imgs, page_index=2)
    keys = set(page1["markdown_images"]) | set(page2["markdown_images"])
    # Two distinct filenames instead of one overwritten file.
    assert keys == {
        "imgs/page_1/img_in_image_box_0_0_100_100.jpg",
        "imgs/page_2/img_in_image_box_0_0_100_100.jpg",
    }


def test_first_page_and_single_page_paths_are_unchanged():
    imgs = [{"path": "imgs/img_in_image_box_1_2_3_4.jpg", "img": "DATA"}]
    for page_index in (0, None):
        result = _convert(imgs, page_index=page_index)
        assert "imgs/img_in_image_box_1_2_3_4.jpg" in result["markdown_images"]


def test_text_references_are_rewritten_to_match_saved_files():
    result = {
        "markdown_texts": "before ![](imgs/img_in_image_box_1_2_3_4.jpg) after",
        "markdown_images": {"imgs/img_in_image_box_1_2_3_4.jpg": "DATA"},
    }
    out = MarkdownConverter._namespace_page_images(result, 2)
    assert "![](imgs/page_2/img_in_image_box_1_2_3_4.jpg)" in out["markdown_texts"]
    # Every saved image key is referenced by the text (no dangling references).
    for key in out["markdown_images"]:
        assert key in out["markdown_texts"]


def test_rewrite_is_substring_safe():
    # One path is a numeric prefix of the other; both must be rewritten exactly.
    result = {
        "markdown_texts": (
            "![](imgs/img_in_image_box_1_2_3_4.jpg) "
            "![](imgs/img_in_image_box_1_2_3_40.jpg)"
        ),
        "markdown_images": {
            "imgs/img_in_image_box_1_2_3_4.jpg": "A",
            "imgs/img_in_image_box_1_2_3_40.jpg": "B",
        },
    }
    out = MarkdownConverter._namespace_page_images(result, 3)
    assert set(out["markdown_images"]) == {
        "imgs/page_3/img_in_image_box_1_2_3_4.jpg",
        "imgs/page_3/img_in_image_box_1_2_3_40.jpg",
    }
    assert "![](imgs/page_3/img_in_image_box_1_2_3_4.jpg)" in out["markdown_texts"]
    assert "![](imgs/page_3/img_in_image_box_1_2_3_40.jpg)" in out["markdown_texts"]
