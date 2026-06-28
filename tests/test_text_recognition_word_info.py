import numpy as np

from paddlex.inference.models.text_recognition.processors import (
    BaseRecLabelDecode,
    is_japanese_kana_char,
)
from paddlex.inference.pipelines.components.common.cal_ocr_word_box import (
    cal_ocr_word_box,
)


def test_japanese_kana_chars_are_classified_for_char_level_boxes():
    assert is_japanese_kana_char("あ")
    assert is_japanese_kana_char("ア")
    assert is_japanese_kana_char("ｰ")
    assert not is_japanese_kana_char("亜")
    assert not is_japanese_kana_char("A")


def test_get_word_info_treats_japanese_kana_as_cn_state():
    decoder = BaseRecLabelDecode()
    text = "りんごジュースストレート"
    selection = np.ones(len(text), dtype=bool)

    word_list, word_col_list, state_list = decoder.get_word_info(text, selection)

    assert ["".join(word) for word in word_list] == [text]
    assert word_col_list == [list(range(len(text)))]
    assert state_list == ["cn"]


def test_japanese_kana_state_produces_character_level_word_boxes():
    text = "りんご"
    word_info = [
        len(text),
        [list(text)],
        [list(range(len(text)))],
        ["cn"],
    ]
    line_box = np.array([[0, 0], [60, 0], [60, 20], [0, 20]], dtype=np.float32)

    words, boxes = cal_ocr_word_box(text, line_box, word_info)

    assert words == list(text)
    assert len(boxes) == len(text)
