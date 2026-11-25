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

"""MarkdownConverter — converts document blocks to Markdown format."""

from __future__ import annotations

import copy

from .markdown_format_funcs import merge_formula_and_number


class MarkdownConverter:
    """Converts a list of document blocks into Markdown text + images.

    The caller is responsible for building ``handle_funcs_dict`` (typically via
    :func:`~.markdown_format_funcs.build_handle_funcs_dict`) and passing it in.  This
    class centralises the iteration / image-collection / special-case logic
    that was previously duplicated across multiple Result classes.
    """

    @staticmethod
    def convert(
        blocks,
        *,
        handle_funcs_dict,
        show_formula_number=False,
        use_seg_flag=False,
        get_seg_flag_func=None,
        imgs_in_doc=None,
        page_index=None,
    ) -> dict:
        """Convert *blocks* to Markdown.

        Args:
            blocks: Iterable of objects satisfying the DocumentBlock protocol
                (``label``, ``content``, ``bbox``, ``image``).
            handle_funcs_dict: ``{label: formatting_func}`` mapping built by
                :func:`build_handle_funcs_dict`.
            show_formula_number: If *True*, merge adjacent formula + number
                blocks (PaddleOCR-VL behaviour).
            use_seg_flag: If *True*, use paragraph-continuity detection so
                that consecutive ``"text"`` blocks are joined without extra
                newlines (PP-StructureV3 / LayoutParsingResultV2 behaviour).
            get_seg_flag_func: ``(block, prev_block) -> (start, end)`` called
                when *use_seg_flag* is *True*.
            imgs_in_doc: Extra images to include (list of ``{"path", "img"}``).
            page_index: Zero-based index of this page within a multi-page
                document. When set to a value greater than ``0`` the emitted
                image paths are namespaced by page so that pages saved into the
                same directory do not overwrite each other. ``None`` / ``0`` (a
                single page) leaves paths unchanged for backward compatibility.

        Returns:
            dict with keys ``markdown_texts``, ``markdown_images``, and
            optionally ``page_continuation_flags``.
        """
        blocks_list = list(blocks)  # ensure indexable for lookahead

        markdown_content = ""
        markdown_images: dict = {}
        last_label = None
        prev_block = None
        seg_start_flag = True
        seg_end_flag = True
        page_first_element_seg_start_flag = None

        for idx, block in enumerate(blocks_list):
            label = block.label

            # --- collect images ---
            if block.image is not None:
                markdown_images[block.image["path"]] = block.image["img"]

            # --- paragraph continuity (result_v2 only) ---
            if use_seg_flag and get_seg_flag_func is not None:
                seg_start_flag, seg_end_flag = get_seg_flag_func(block, prev_block)
                if page_first_element_seg_start_flag is None:
                    page_first_element_seg_start_flag = seg_start_flag

            # --- formula-number merging (paddleocr_vl only) ---
            if (
                show_formula_number
                and label in ("display_formula", "formula")
                and idx < len(blocks_list) - 1
            ):
                next_block = blocks_list[idx + 1]
                if next_block.label == "formula_number":
                    block = copy.copy(block)
                    block.content = merge_formula_and_number(
                        block.content, next_block.content
                    )

            # --- format & append ---
            handle_func = handle_funcs_dict.get(label, None)
            if handle_func:
                if use_seg_flag:
                    prev_block = block
                if (
                    use_seg_flag
                    and label == last_label == "text"
                    and seg_start_flag is False
                ):
                    markdown_content += handle_func(block)
                else:
                    markdown_content += (
                        "\n\n" + handle_func(block)
                        if markdown_content
                        else handle_func(block)
                    )
                last_label = label

        # --- build return dict ---
        result = {
            "markdown_texts": markdown_content,
            "markdown_images": markdown_images,
        }

        if use_seg_flag:
            if page_first_element_seg_start_flag is None:
                page_first_element_seg_start_flag = True
            result["page_continuation_flags"] = (
                page_first_element_seg_start_flag,
                seg_end_flag,
            )

        if imgs_in_doc:
            for img in imgs_in_doc:
                result["markdown_images"][img["path"]] = img["img"]

        # In a multi-page document every page is converted separately but saved
        # into the same directory. Image filenames are derived from (label, box)
        # only, so a figure at the same coordinates on different pages resolves
        # to the same path and later pages silently overwrite earlier ones.
        # Namespace the image paths of pages after the first by their page index
        # so they stay unique (page 0 / single-page output is left unchanged).
        if page_index:
            result = MarkdownConverter._namespace_page_images(result, page_index)

        return result

    @staticmethod
    def _namespace_page_images(result: dict, page_index: int) -> dict:
        """Prefix every ``imgs/`` image path with ``imgs/page_{page_index}/``.

        Rewrites both the ``markdown_images`` keys and the corresponding
        references inside ``markdown_texts`` so they stay consistent. Paths are
        rewritten longest-first so no path can match inside another during the
        text substitution.
        """
        images = result.get("markdown_images") or {}
        if not images:
            return result

        text = result.get("markdown_texts", "")
        remapped: dict = {}
        for path in sorted(images, key=len, reverse=True):
            img = images[path]
            if isinstance(path, str) and path.startswith("imgs/"):
                new_path = f"imgs/page_{page_index}/" + path[len("imgs/") :]
                if new_path != path:
                    text = text.replace(path, new_path)
                remapped[new_path] = img
            else:
                remapped[path] = img

        result["markdown_texts"] = text
        result["markdown_images"] = remapped
        return result
