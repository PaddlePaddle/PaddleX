# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import os
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import PIL

from ...common.tokenizer.bert_tokenizer import BertTokenizer
from .....utils.lazy_loader import LazyLoader

# NOTE: LazyLoader is used to avoid conflicts between ultra-infer and Paddle
paddle = LazyLoader("lazy_paddle", globals(), "paddle")
T = LazyLoader("T", globals(), "paddle.vision.transforms")
F = LazyLoader("F", globals(), "paddle.nn.functional")


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def _text_pad_batch_data(
    insts,
    pad_idx=0,
    max_seq_len=None,
    return_pos=False,
    return_input_mask=False,
    return_max_len=False,
    return_num_token=False,
    return_seq_lens=False,
    pad_2d_pos_ids=False,
    pad_segment_id=False,
    select=False,
    extract=False,
):
    """Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts) if max_seq_len is None else max_seq_len
    if pad_2d_pos_ids:
        boxes = [x + [[0, 0, 0, 0]] * (max_len - len(x)) for x in insts]
        boxes = np.array(boxes, dtype="int64")
        return boxes

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts]
    )
    return_list += [inst_data.astype("int64").reshape([-1, max_len, 1])]

    if return_pos:
        inst_pos = np.array(
            [
                list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
                for inst in insts
            ]
        )

        return_list += [inst_pos.astype("int64").reshape([-1, max_len, 1])]

    if return_input_mask:
        input_mask_data = np.array(
            [[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts]
        )
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


class GroundingDINOPostProcessor(object):
    """PostProcessors for GroundingDINO"""

    def __init__(
        self,
        tokenizer,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """Init Function for GroundingDINO PostProcessor

        Args:
            tokenzier (BertTokenizer): tokenzier used for prompt tokenize.
            box_threshold (float): threshold for low confidence bbox filtering.
            text_threshold (float): threshold for determining predicted bbox text labels.
        """
        self.tokenizer = tokenizer
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def __call__(
        self,
        pred_boxes,
        pred_logits,
        prompt,
        src_images,
        box_threshold=None,
        text_threshold=None,
        **kwargs,
    ):

        box_threshold = self.box_threshold if box_threshold is None else box_threshold
        text_threshold = (
            self.text_threshold if text_threshold is None else text_threshold
        )

        if isinstance(pred_logits, np.ndarray):
            pred_logits = paddle.to_tensor(pred_logits)
        if isinstance(pred_boxes, np.ndarray):
            pred_boxes = paddle.to_tensor(pred_boxes)

        assert pred_logits.ndim == 3 and pred_boxes.ndim == 3
        assert pred_logits.shape[0] == pred_boxes.shape[0] == len(src_images)

        rst_boxes = []
        for pred_logit, pred_box, src_image in zip(pred_logits, pred_boxes, src_images):
            rst_boxes.append(
                self.postprocess(
                    pred_logit,
                    pred_box,
                    prompt,
                    src_image,
                    box_threshold,
                    text_threshold,
                )
            )

        return rst_boxes

    def postprocess(
        self,
        pred_logits,
        pred_boxes,
        src_prompt,
        src_image,
        box_threshold,
        text_threshold,
    ):
        """Post Process for prediction result of single image."""

        logits = F.sigmoid(pred_logits)
        boxes = pred_boxes

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        H, W, *_ = src_image.shape

        pred_bboxes = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = self.decode(logit > text_threshold, src_prompt)
            pred_score = logit.max().item()
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            box *= paddle.to_tensor([W, H, W, H]).astype(paddle.float32)
            pred_bboxes.append(
                {
                    "coordinate": box.detach().cpu().tolist(),
                    "label": pred_phrase,
                    "score": pred_score,
                }
            )

        return pred_bboxes

    def decode(self, posmap, prompt):

        tokenized = self.tokenizer(prompt)
        if posmap.dim() == 1:
            non_zero_idx = posmap.nonzero(as_tuple=True)[0].squeeze(-1).tolist()
            token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
            return self.tokenizer.decode(token_ids)
        else:
            raise NotImplementedError("posmap must be 1-dim")


class GroundingDINOProcessor(object):
    """Image and Text Processors for GroundingDINO"""

    def __init__(
        self,
        model_dir,
        text_max_words: int = 256,
        image_do_resize: bool = True,
        image_target_size: Union[Tuple[int], int] = (800, 1333),
        image_do_normalize: bool = True,
        image_mean: Union[float, List[float]] = [0.485, 0.456, 0.406],
        image_std: Union[float, List[float]] = [0.229, 0.224, 0.225],
        image_do_nested: bool = True,
        **kwargs,
    ):
        self.text_processor = GroundingDinoTextProcessor(text_max_words)
        self.image_processor = GroundingDinoImageProcessor(
            do_resize=image_do_resize,
            target_size=image_target_size,
            do_normalize=image_do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_nested=image_do_nested,
        )
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        assert os.path.isdir(tokenizer_dir), f"{tokenizer_dir} not exists."
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

    def __call__(
        self,
        images: List[PIL.Image.Image],
        text: str,
        **kwargs,
    ):

        self.prompt = self.text_processor.pre_caption(text)
        input_ids = self.tokenizer([self.prompt]).input_ids
        special_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]
        )
        tokenized_out = self.text_processor(input_ids, special_tokens)

        image_tensor, mask = self.image_processor(images)

        paddle_rst = [
            tokenized_out["attention_mask"],
            tokenized_out["input_ids"],
            mask,
            tokenized_out["position_ids"],
            tokenized_out["text_self_attention_masks"],
            image_tensor,
        ]
        return [arr.numpy() for arr in paddle_rst]


class GroundingDinoTextProcessor(object):
    """Constructs a GroundingDino text processor."""

    def __init__(
        self,
        max_words: int = 256,
    ):
        self.max_words = max_words

    def __call__(
        self,
        input_ids,
        special_tokens_list,
    ):
        """Preprocess the text with tokenization."""
        tokenized_out = {}
        input_ids = _text_pad_batch_data(input_ids)
        input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64).squeeze(-1)
        tokenized_out["input_ids"] = input_ids
        tokenized_out["attention_mask"] = paddle.cast(input_ids != 0, paddle.int64)

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = self.generate_masks_with_special_tokens_and_transfer_map(
            tokenized_out, special_tokens_list
        )

        if text_self_attention_masks.shape[1] > self.max_words:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_words, : self.max_words
            ]
            position_ids = position_ids[:, : self.max_words]
            tokenized_out["input_ids"] = tokenized_out["input_ids"][:, : self.max_words]
            tokenized_out["attention_mask"] = tokenized_out["attention_mask"][
                :, : self.max_words
            ]
        tokenized_out["position_ids"] = position_ids
        tokenized_out["text_self_attention_masks"] = text_self_attention_masks

        return tokenized_out

    def pre_caption(self, caption: str) -> str:
        """Preprocess the text before tokenization."""
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        return caption

    def generate_masks_with_special_tokens_and_transfer_map(
        self, tokenized, special_tokens_list
    ):
        """Generate attention mask between each pair of special tokens
        Args:
            input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
            special_tokens_mask (list): special tokens mask.
        Returns:
            torch.Tensor: attention mask between each special tokens.
        """
        input_ids = tokenized["input_ids"]
        bs, num_token = input_ids.shape
        special_tokens_mask = paddle.zeros((bs, num_token), dtype=paddle.bool)
        for special_token in special_tokens_list:
            special_tokens_mask |= input_ids == special_token

        idxs = paddle.nonzero(special_tokens_mask)

        attention_mask = (
            paddle.eye(num_token, dtype=paddle.int32)
            .cast(paddle.bool)
            .unsqueeze(0)
            .tile([bs, 1, 1])
        )
        position_ids = paddle.zeros((bs, num_token), dtype=paddle.int64)
        cate_to_token_mask_list = [[] for _ in range(bs)]
        previous_col = 0

        for i in range(idxs.shape[0]):
            row, col = idxs[i]
            if (col == 0) or (col == num_token - 1):
                attention_mask[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention_mask[
                    row, previous_col + 1 : col + 1, previous_col + 1 : col + 1
                ] = True
                position_ids[row, previous_col + 1 : col + 1] = paddle.arange(
                    0, col - previous_col
                )
                c2t_maski = paddle.zeros(
                    [
                        num_token,
                    ]
                ).cast(paddle.bool)
                c2t_maski[previous_col + 1 : col] = True
                cate_to_token_mask_list[row].append(c2t_maski)
            previous_col = col

        return attention_mask, position_ids.cast(paddle.int64), cate_to_token_mask_list


class GroundingDinoImageProcessor(object):
    """Constructs a GroundingDino image processor."""

    def __init__(
        self,
        do_resize: bool = True,
        target_size: Union[Tuple[int], int] = (800, 1333),
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = [0.485, 0.456, 0.406],
        image_std: Union[float, List[float]] = [0.229, 0.224, 0.225],
        do_nested: bool = True,
    ) -> None:

        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        assert isinstance(target_size, (tuple, list)) and len(target_size) == 2
        self.target_size = target_size

        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_nested = do_nested

    def __call__(self, images, **kwargs):
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, **kwargs)

    def resize(self, image, size=None, max_size=1333):
        """Officially aligned Image resize."""

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return get_size_with_aspect_ratio(image_size, size, max_size)

        size = get_size(image.size, size, max_size)
        rescaled_image = T.resize(image, size)

        return rescaled_image

    def nested_tensor_from_tensor_list(self, tensor_list):
        if tensor_list[0].ndim == 3:
            max_size = _max_by_axis([list(img.shape) for img in tensor_list])
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            tensor = paddle.zeros(batch_shape, dtype=dtype)
            mask = paddle.ones((b, h, w), dtype=paddle.bool)
            for i in range(b):
                img = tensor_list[i]
                tensor[i, : img.shape[0], : img.shape[1], : img.shape[2]] = img
                mask[i, : img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError(
                f"Not supported tensor format of {tensor_list[0].shape}, only support shape like 'CHW' ."
            )
        return tensor, mask

    def preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        target_size: Optional[Dict[str, int]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_nested: bool = None,
        **kwargs,
    ):
        """Preprocess an image or batch of images."""
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_nested = do_nested if do_nested is not None else self.do_nested

        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        target_size = target_size if target_size is not None else self.target_size

        if not isinstance(images, (list, tuple)):
            images = [images]
        if isinstance(images[0], np.ndarray):
            images = [PIL.Image.fromarray(image) for image in images]

        if do_resize:
            min_size = min(self.target_size)
            max_size = max(self.target_size)
            images = [
                T.to_tensor(self.resize(image=image, size=min_size, max_size=max_size))
                for image in images
            ]

        if do_normalize:
            images = T.normalize(images, mean=image_mean, std=image_std)

        if do_nested:
            tensors, masks = self.nested_tensor_from_tensor_list(images)

        return tensors, masks
