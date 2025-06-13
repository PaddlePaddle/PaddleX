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
from typing import Any, List, Union

import numpy as np

from .setting import BLOCK_LABEL_MAP, LINE_SETTINGS
from .utils import (
    caculate_euclidean_dist,
    calculate_projection_overlap_ratio,
    is_english_letter,
    is_non_breaking_punctuation,
    is_numeric,
)

__all__ = [
    "TextSpan",
    "TextLine",
    "LayoutBlock",
    "LayoutRegion",
]


class TextSpan(object):
    """Text span class"""

    def __init__(self, box, text, label):
        """
        Initialize a TextSpan object.

        Args:
            box (list): The bounding box of the text span.
            text (str): The text content of the text span.
            label (int): The label of the text span.
        """
        self.box = box
        self.text = text
        self.label = label

    def __str__(self) -> str:
        return f"{self.text}"

    def __repr__(self) -> str:
        return f"{self.text}"


class TextLine(object):
    """Text line class"""

    def __init__(self, spans: List[TextSpan] = [], direction="horizontal"):
        """
        Initialize a TextLine object.

        Args:
            spans (List[TextSpan]): A list of TextSpan objects. Defaults to [].
            direction (str): The direction of the text line. Defaults to "horizontal".
        """
        self.spans = spans
        self.direction = direction
        self.region_box = self.get_region_box()
        self.need_new_line = False

    @property
    def labels(self):
        return [span.label for span in self.spans]

    @property
    def boxes(self):
        return [span.box for span in self.spans]

    @property
    def height(self):
        start_idx = 1 if self.direction == "horizontal" else 0
        end_idx = 3 if self.direction == "horizontal" else 2
        return abs(self.region_box[end_idx] - self.region_box[start_idx])

    @property
    def width(self):
        start_idx = 0 if self.direction == "horizontal" else 1
        end_idx = 2 if self.direction == "horizontal" else 3
        return abs(self.region_box[end_idx] - self.region_box[start_idx])

    def __str__(self) -> str:
        return f"{' '.join([str(span.text) for span in self.spans])}\n"

    def __repr__(self) -> str:
        return f"{' '.join([str(span.text) for span in self.spans])}\n"

    def add_span(self, span: Union[TextSpan, List[TextSpan]]):
        """
        Add a span to the text line.

        Args:
            span (Union[TextSpan, List[TextSpan]]): A single TextSpan object or a list of TextSpan objects.
        """
        if isinstance(span, list):
            self.spans.extend(span)
        else:
            self.spans.append(span)
        self.region_box = self.get_region_box()

    def get_region_box(self):
        """
        Get the region box of the text line.

        Returns:
            list: The region box of the text line.
        """
        if not self.spans:
            return None  # or an empty list, or however you want to handle no spans

        # Initialize min and max values with the first span's box
        x_min, y_min, x_max, y_max = self.spans[0].box

        for span in self.spans:
            x_min = min(x_min, span.box[0])
            y_min = min(y_min, span.box[1])
            x_max = max(x_max, span.box[2])
            y_max = max(y_max, span.box[3])

        return [x_min, y_min, x_max, y_max]

    def get_texts(
        self,
        block_label: str,
        block_text_width: int,
        block_start_coordinate: int,
        block_stop_coordinate: int,
        ori_image,
        text_rec_model=None,
        text_rec_score_thresh=None,
    ):
        """
        Get the text of the text line.

        Args:
            block_label (str): The label of the block.
            block_text_width (int): The width of the block.
            block_start_coordinate (int): The starting coordinate of the block.
            block_stop_coordinate (int): The stopping coordinate of the block.
            ori_image (np.ndarray): The original image.
            text_rec_model (Any): The text recognition model.
            text_rec_score_thresh (float): The text recognition score threshold.

        Returns:
            str: The text of the text line.
        """
        span_box_start_index = 0 if self.direction == "horizontal" else 1
        lines_start_index = 1 if self.direction == "horizontal" else 3
        self.spans.sort(
            key=lambda span: (
                span.box[span_box_start_index] // 2,
                (
                    span.box[lines_start_index]
                    if self.direction == "horizontal"
                    else -span.box[lines_start_index]
                ),
            )
        )
        if "formula" in self.labels:
            sort_index = 0 if self.direction == "horizontal" else 1
            splited_spans = self.split_boxes_by_projection()
            if len(self.spans) != len(splited_spans):
                splited_spans.sort(key=lambda span: span.box[sort_index])
                new_spans = []
                for span in splited_spans:
                    bbox = span.box
                    if span.label == "text":
                        crop_img = ori_image[
                            int(bbox[1]) : int(bbox[3]),
                            int(bbox[0]) : int(bbox[2]),
                        ]
                        crop_img_rec_res = next(text_rec_model([crop_img]))
                        crop_img_rec_score = crop_img_rec_res["rec_score"]
                        crop_img_rec_text = crop_img_rec_res["rec_text"]
                        span.text = crop_img_rec_text
                        if crop_img_rec_score < text_rec_score_thresh:
                            continue
                    new_spans.append(span)
                self.spans = new_spans
        line_text = self.format_line(
            block_text_width,
            block_start_coordinate,
            block_stop_coordinate,
            line_gap_limit=self.height * 1.5,
            block_label=block_label,
        )
        return line_text

    def is_projection_contained(self, box_a, box_b, start_idx, end_idx):
        """Check if box_a completely contains box_b in the x-direction."""
        return box_a[start_idx] <= box_b[start_idx] and box_a[end_idx] >= box_b[end_idx]

    def split_boxes_by_projection(self, offset=1e-5):
        """
        Check if there is any complete containment in the x-direction
        between the bounding boxes and split the containing box accordingly.

        Args:
            offset (float): A small offset value to ensure that the split boxes are not too close to the original boxes.
        Returns:
            A new list of boxes, including split boxes, with the same `rec_text` and `label` attributes.
        """

        new_spans = []
        if self.direction == "horizontal":
            projection_start_index, projection_end_index = 0, 2
        else:
            projection_start_index, projection_end_index = 1, 3

        for i in range(len(self.spans)):
            span = self.spans[i]
            is_split = False
            for j in range(i, len(self.spans)):
                box_b = self.spans[j].box
                box_a, text, label = span.box, span.text, span.label
                if self.is_projection_contained(
                    box_a, box_b, projection_start_index, projection_end_index
                ):
                    is_split = True
                    # Split box_a based on the x-coordinates of box_b
                    if box_a[projection_start_index] < box_b[projection_start_index]:
                        w = (
                            box_b[projection_start_index]
                            - offset
                            - box_a[projection_start_index]
                        )
                        if w > 1:
                            new_bbox = box_a.copy()
                            new_bbox[projection_end_index] = (
                                box_b[projection_start_index] - offset
                            )
                            new_spans.append(
                                TextSpan(
                                    box=np.array(new_bbox),
                                    text=text,
                                    label=label,
                                )
                            )
                    if box_a[projection_end_index] > box_b[projection_end_index]:
                        w = (
                            box_a[projection_end_index]
                            - box_b[projection_end_index]
                            + offset
                        )
                        if w > 1:
                            box_a[projection_start_index] = (
                                box_b[projection_end_index] + offset
                            )
                            span = TextSpan(
                                box=np.array(box_a),
                                text=text,
                                label=label,
                            )
                if j == len(self.spans) - 1 and is_split:
                    new_spans.append(span)
            if not is_split:
                new_spans.append(span)

        return new_spans

    def format_line(
        self,
        block_text_width: int,
        block_start_coordinate: int,
        block_stop_coordinate: int,
        line_gap_limit: int = 10,
        block_label: str = "text",
    ) -> str:
        """
        Format a line of text spans based on layout constraints.

        Args:
            block_text_width (int): The width of the block.
            block_start_coordinate (int): The starting coordinate of the block.
            block_stop_coordinate (int): The stopping coordinate of the block.
            line_gap_limit (int): The limit for the number of pixels after the last span that should be considered part of the last line. Default is 10.
            block_label (str): The label associated with the entire block. Default is 'text'.
        Returns:
            str: Formatted line of text.
        """
        first_span_box = self.spans[0].box
        last_span_box = self.spans[-1].box

        line_text = ""
        for span in self.spans:
            if span.label == "formula" and block_label != "formula":
                formula_rec = span.text
                if not formula_rec.startswith("$") and not formula_rec.endswith("$"):
                    if len(self.spans) > 1:
                        span.text = f"${span.text}$"
                    else:
                        span.text = f"\n${span.text}$"
            line_text += span.text
            if (
                len(span.text) > 0
                and is_english_letter(line_text[-1])
                or span.label == "formula"
            ):
                line_text += " "

        if self.direction == "horizontal":
            text_stop_index = 2
        else:
            text_stop_index = 3

        if line_text.endswith(" "):
            line_text = line_text[:-1]

        if len(line_text) == 0:
            return ""

        last_char = line_text[-1]

        if (
            not is_english_letter(last_char)
            and not is_non_breaking_punctuation(last_char)
            and not is_numeric(last_char)
        ) or (
            block_stop_coordinate - last_span_box[text_stop_index]
            > block_text_width * 0.3
        ):
            if (
                self.direction == "horizontal"
                and block_stop_coordinate - last_span_box[text_stop_index]
                > line_gap_limit
            ) or (
                self.direction == "vertical"
                and (
                    block_stop_coordinate - last_span_box[text_stop_index]
                    > line_gap_limit
                    or first_span_box[1] - block_start_coordinate > line_gap_limit
                )
            ):
                self.need_new_line = True

        if line_text.endswith("-"):
            line_text = line_text[:-1]
            return line_text

        if (len(line_text) > 0 and is_english_letter(last_char)) or line_text.endswith(
            "$"
        ):
            line_text += " "
        if (
            len(line_text) > 0
            and not is_english_letter(last_char)
            and not is_numeric(last_char)
        ) or self.direction == "vertical":
            if (
                block_stop_coordinate - last_span_box[text_stop_index]
                > block_text_width * 0.3
                and len(line_text) > 0
                and not is_non_breaking_punctuation(last_char)
            ):
                line_text += "\n"
                self.need_new_line = True
        elif (
            block_stop_coordinate - last_span_box[text_stop_index]
            > (block_stop_coordinate - block_start_coordinate) * 0.5
        ):
            line_text += "\n"
            self.need_new_line = True

        return line_text


class LayoutBlock(object):
    """Layout Block Class"""

    def __init__(self, label, bbox, content="") -> None:
        """
        Initialize a LayoutBlock object.

        Args:
            label (str): Label assigned to the block.
            bbox (list): Bounding box coordinates of the block.
            content (str, optional): Content of the block. Defaults to an empty string.
        """
        self.label = label
        self.order_label = None
        self.bbox = list(map(int, bbox))
        self.content = content
        self.seg_start_coordinate = float("inf")
        self.seg_end_coordinate = float("-inf")
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = float(self.width) * float(self.height)
        self.num_of_lines = 1
        self.image = None
        self.index = None
        self.order_index = None
        self.text_line_width = 1
        self.text_line_height = 1
        self.child_blocks = []
        self.update_direction()

    def __str__(self) -> str:
        _str = f"\n\n#################\nindex:\t{self.index}\nlabel:\t{self.label}\nregion_label:\t{self.order_label}\nbbox:\t{self.bbox}\ncontent:\t{self.content}\n#################"
        return _str

    def __repr__(self) -> str:
        _str = f"\n\n#################\nindex:\t{self.index}\nlabel:\t{self.label}\nregion_label:\t{self.order_label}\nbbox:\t{self.bbox}\ncontent:\t{self.content}\n#################"
        return _str

    def to_dict(self) -> dict:
        return self.__dict__

    def update_direction(self, direction=None) -> None:
        """
        Update the direction of the block based on its bounding box.

        Args:
            direction (str, optional): Direction of the block. If not provided, it will be determined automatically using the bounding box. Defaults to None.
        """
        if not direction:
            direction = self.get_bbox_direction()
        self.direction = direction
        self.update_direction_info()

    def update_direction_info(self) -> None:
        """Update the direction information of the block based on its direction."""
        if self.direction == "horizontal":
            self.secondary_direction = "vertical"
            self.short_side_length = self.height
            self.long_side_length = self.width
            self.start_coordinate = self.bbox[0]
            self.end_coordinate = self.bbox[2]
            self.secondary_direction_start_coordinate = self.bbox[1]
            self.secondary_direction_end_coordinate = self.bbox[3]
        else:
            self.secondary_direction = "horizontal"
            self.short_side_length = self.width
            self.long_side_length = self.height
            self.start_coordinate = self.bbox[1]
            self.end_coordinate = self.bbox[3]
            self.secondary_direction_start_coordinate = self.bbox[0]
            self.secondary_direction_end_coordinate = self.bbox[2]

    def append_child_block(self, child_block) -> None:
        """
        Append a child block to the current block.

        Args:
            child_block (LayoutBlock): Child block to be added.
        Returns:
            None
        """
        if not self.child_blocks:
            self.ori_bbox = self.bbox.copy()
        x1, y1, x2, y2 = self.bbox
        x1_child, y1_child, x2_child, y2_child = child_block.bbox
        union_bbox = (
            min(x1, x1_child),
            min(y1, y1_child),
            max(x2, x2_child),
            max(y2, y2_child),
        )
        self.bbox = union_bbox
        self.update_direction_info()
        child_blocks = [child_block]
        if child_block.child_blocks:
            child_blocks.extend(child_block.get_child_blocks())
        self.child_blocks.extend(child_blocks)

    def get_child_blocks(self) -> list:
        """Get all child blocks of the current block."""
        self.bbox = self.ori_bbox
        child_blocks = self.child_blocks.copy()
        self.child_blocks = []
        return child_blocks

    def get_centroid(self) -> tuple:
        """Get the centroid of the bounding box of the block."""
        x1, y1, x2, y2 = self.bbox
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        return centroid

    def get_bbox_direction(self, direction_ratio: float = 1.0) -> str:
        """
        Determine if a bounding box is horizontal or vertical.

        Args:
            direction_ratio (float): Ratio for determining direction. Default is 1.0.

        Returns:
            str: "horizontal" or "vertical".
        """
        return (
            "horizontal" if self.width * direction_ratio >= self.height else "vertical"
        )

    def calculate_text_line_direction(
        self, bboxes: List[List[int]], direction_ratio: float = 1.5
    ) -> bool:
        """
        Calculate the direction of the text based on the bounding boxes.

        Args:
            bboxes (list): A list of bounding boxes.
            direction_ratio (float): Ratio for determining direction. Default is 1.5.

        Returns:
            str: "horizontal" or "vertical".
        """

        horizontal_box_num = 0
        for bbox in bboxes:
            if len(bbox) != 4:
                raise ValueError(
                    "Invalid bounding box format. Expected a list of length 4."
                )
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            horizontal_box_num += 1 if width * direction_ratio >= height else 0

        return "horizontal" if horizontal_box_num >= len(bboxes) * 0.5 else "vertical"

    def group_boxes_into_lines(
        self, ocr_rec_res, line_height_iou_threshold
    ) -> List[TextLine]:
        """
        Group the bounding boxes into lines based on their direction.

        Args:
            ocr_rec_res (dict): The result of OCR recognition.
            line_height_iou_threshold (float): The minimum IOU value required for two spans to belong to the same line.

        Returns:
            list: A list of TextLines.
        """
        rec_boxes = ocr_rec_res["boxes"]
        rec_texts = ocr_rec_res["rec_texts"]
        rec_labels = ocr_rec_res["rec_labels"]

        text_boxes = [
            rec_boxes[i] for i in range(len(rec_boxes)) if rec_labels[i] == "text"
        ]
        direction = self.calculate_text_line_direction(text_boxes)
        self.update_direction(direction)

        spans = [TextSpan(*span) for span in zip(rec_boxes, rec_texts, rec_labels)]

        if not spans:
            return []

        # sort spans by direction
        if self.direction == "vertical":
            spans.sort(
                key=lambda span: span.box[0], reverse=True
            )  # sort by x coordinate
            match_direction = "horizontal"
        else:
            spans.sort(
                key=lambda span: span.box[1], reverse=False
            )  # sort by y coordinate
            match_direction = "vertical"

        lines = []
        current_line = TextLine([spans[0]], direction=self.direction)

        for span in spans[1:]:
            overlap_ratio = calculate_projection_overlap_ratio(
                current_line.region_box, span.box, match_direction, mode="small"
            )

            if overlap_ratio >= line_height_iou_threshold:
                current_line.add_span(span)
            else:
                lines.append(current_line)
                current_line = TextLine([span], direction=self.direction)

        lines.append(current_line)

        if lines and self.direction == "vertical":
            line_heights = np.array([line.height for line in lines])
            min_height = np.min(line_heights)
            max_height = np.max(line_heights)

            # if height is too large, filter out the line
            if max_height > min_height * 2:
                normal_height_threshold = min_height * 1.1
                normal_height_count = np.sum(line_heights < normal_height_threshold)

                # if the number of lines with height less than the threshold is less than 40%, then filter out the line
                if normal_height_count < len(lines) * 0.4:
                    keep_condition = line_heights <= normal_height_threshold
                    lines = [line for line, keep in zip(lines, keep_condition) if keep]

        # calculate the average height of the text line
        if lines:
            line_heights = [line.height for line in lines]
            line_widths = [line.width for line in lines]
            self.text_line_height = np.mean(line_heights)
            self.text_line_width = np.mean(line_widths)
        else:
            self.text_line_height = 0
            self.text_line_width = 0

        return lines

    def update_text_content(
        self,
        image: list,
        ocr_rec_res: dict,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> None:
        """
        Update the text content of the block based on the OCR result.

        Args:
            image (list): The input image.
            ocr_rec_res (dict): The result of OCR recognition.
            text_rec_model (Any): The model used for text recognition.
            text_rec_score_thresh (Union[float, None]): The score threshold for text recognition. If None, use the default setting.

        Returns:
            None
        """

        if len(ocr_rec_res["rec_texts"]) == 0:
            self.content = ""
            return

        lines = self.group_boxes_into_lines(
            ocr_rec_res,
            LINE_SETTINGS.get("line_height_iou_threshold", 0.8),
        )

        # words start coordinate and stop coordinate in the line
        coord_start_idx = 0 if self.direction == "horizontal" else 1
        coord_end_idx = coord_start_idx + 2

        if self.label == "reference":
            rec_boxes = ocr_rec_res["boxes"]
            block_start = min([box[coord_start_idx] for box in rec_boxes])
            block_stop = max([box[coord_end_idx] for box in rec_boxes])
        else:
            block_start = self.bbox[coord_start_idx]
            block_stop = self.bbox[coord_end_idx]

        text_lines = []
        text_width_list = []
        need_new_line_num = 0

        for line_idx, line in enumerate(lines):
            line: TextLine = line
            text_width_list.append(line.width)
            # get text from line
            line_text = line.get_texts(
                block_label=self.label,
                block_text_width=max(text_width_list),
                block_start_coordinate=block_start,
                block_stop_coordinate=block_stop,
                ori_image=image,
                text_rec_model=text_rec_model,
                text_rec_score_thresh=text_rec_score_thresh,
            )

            if line.need_new_line:
                need_new_line_num += 1

            # set segment start and end coordinate
            if line_idx == 0:
                self.seg_start_coordinate = line.spans[0].box[0]
            elif line_idx == len(lines) - 1:
                self.seg_end_coordinate = line.spans[-1].box[2]

            text_lines.append(line_text)

        delim = LINE_SETTINGS["delimiter_map"].get(self.label, "")

        if delim == "":
            content = ""
            pre_line_end = False
            last_char = ""
            for idx, line_text in enumerate(text_lines):
                if len(line_text) == 0:
                    continue

                line: TextLine = lines[idx]
                if pre_line_end:
                    start_gep_len = line.region_box[coord_start_idx] - block_start
                    if (
                        (
                            start_gep_len > line.height * 1.5
                            and not is_english_letter(last_char)
                            and not is_numeric(last_char)
                        )
                        or start_gep_len > (block_stop - block_start) * 0.4
                    ) and not content.endswith("\n"):
                        line_text = "\n" + line_text
                content += f"{line_text}"

                if len(line_text) > 2 and line_text.endswith(" "):
                    last_char = line_text[-2]
                else:
                    last_char = line_text[-1]
                if (
                    len(line_text) > 0
                    and not line_text.endswith("\n")
                    and not is_english_letter(last_char)
                    and not is_non_breaking_punctuation(last_char)
                    and not is_numeric(last_char)
                    and need_new_line_num > len(text_lines) * 0.5
                ) or need_new_line_num > len(text_lines) * 0.6:
                    content += f"\n"
                if (
                    block_stop - line.region_box[coord_end_idx]
                    > (block_stop - block_start) * 0.3
                ):
                    pre_line_end = True
        else:
            content = delim.join(text_lines)

        self.content = content
        self.num_of_lines = len(text_lines)


class LayoutRegion(LayoutBlock):
    """LayoutRegion class"""

    def __init__(
        self,
        bbox,
        blocks: List[LayoutBlock] = [],
    ) -> None:
        """
        Initialize a LayoutRegion object.

        Args:
            bbox (List[int]): The bounding box of the region.
            blocks (List[LayoutBlock]): A list of blocks that belong to this region.
        """
        super().__init__("region", bbox, content="")
        self.bbox = bbox
        self.block_map = {}
        self.direction = "horizontal"
        self.doc_title_block_idxes = []
        self.paragraph_title_block_idxes = []
        self.vision_block_idxes = []
        self.unordered_block_idxes = []
        self.vision_title_block_idxes = []
        self.normal_text_block_idxes = []
        self.euclidean_distance = float(np.inf)
        self.header_block_idxes = []
        self.footer_block_idxes = []
        self.text_line_width = 20
        self.text_line_height = 10
        self.num_of_lines = 10
        self.init_region_info_from_layout(blocks)
        self.update_euclidean_distance()

    def init_region_info_from_layout(self, blocks: List[LayoutBlock]) -> None:
        """Initialize the information about the layout region from the given blocks.

        Args:
            blocks (List[LayoutBlock]): A list of blocks that belong to this region.
        Returns:
            None
        """
        horizontal_normal_text_block_num = 0
        text_line_height_list = []
        text_line_width_list = []
        for idx, block in enumerate(blocks):
            self.block_map[idx] = block
            block.index = idx
            if block.label in BLOCK_LABEL_MAP["header_labels"]:
                self.header_block_idxes.append(idx)
            elif block.label in BLOCK_LABEL_MAP["doc_title_labels"]:
                self.doc_title_block_idxes.append(idx)
            elif block.label in BLOCK_LABEL_MAP["paragraph_title_labels"]:
                self.paragraph_title_block_idxes.append(idx)
            elif block.label in BLOCK_LABEL_MAP["vision_labels"]:
                self.vision_block_idxes.append(idx)
            elif block.label in BLOCK_LABEL_MAP["vision_title_labels"]:
                self.vision_title_block_idxes.append(idx)
            elif block.label in BLOCK_LABEL_MAP["footer_labels"]:
                self.footer_block_idxes.append(idx)
            elif block.label in BLOCK_LABEL_MAP["unordered_labels"]:
                self.unordered_block_idxes.append(idx)
            else:
                self.normal_text_block_idxes.append(idx)
                text_line_height_list.append(block.text_line_height)
                text_line_width_list.append(block.text_line_width)
                if block.direction == "horizontal":
                    horizontal_normal_text_block_num += 1
        direction = (
            "horizontal"
            if horizontal_normal_text_block_num
            >= len(self.normal_text_block_idxes) * 0.5
            else "vertical"
        )
        self.update_direction(direction)
        self.text_line_width = (
            np.mean(text_line_width_list) if text_line_width_list else 20
        )
        self.text_line_height = (
            np.mean(text_line_height_list) if text_line_height_list else 10
        )

    def update_euclidean_distance(self):
        """Update euclidean distance between each block and the reference point"""
        blocks: List[LayoutBlock] = list(self.block_map.values())
        if self.direction == "horizontal":
            ref_point = (0, 0)
            block_distance = [
                caculate_euclidean_dist((block.bbox[0], block.bbox[1]), ref_point)
                for block in blocks
            ]
        else:
            ref_point = (self.bbox[2], 0)
            block_distance = [
                caculate_euclidean_dist((block.bbox[2], block.bbox[1]), ref_point)
                for block in blocks
            ]
        self.euclidean_distance = min(block_distance) if len(block_distance) > 0 else 0

    def update_direction(self, direction=None):
        """
        Update the direction of the layout region.

        Args:
            direction (str): The new direction of the layout region.
        """
        super().update_direction(direction=direction)
        if self.direction == "horizontal":
            self.direction_start_index = 0
            self.direction_end_index = 2
            self.secondary_direction_start_index = 1
            self.secondary_direction_end_index = 3
            self.secondary_direction = "vertical"
        else:
            self.direction_start_index = 1
            self.direction_end_index = 3
            self.secondary_direction_start_index = 0
            self.secondary_direction_end_index = 2
            self.secondary_direction = "horizontal"

        self.direction_center_coordinate = (
            self.bbox[self.direction_start_index] + self.bbox[self.direction_end_index]
        ) / 2
        self.secondary_direction_center_coordinate = (
            self.bbox[self.secondary_direction_start_index]
            + self.bbox[self.secondary_direction_end_index]
        ) / 2
