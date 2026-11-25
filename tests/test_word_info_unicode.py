import numpy as np

from paddlex.inference.models.text_recognition.processors import BaseRecLabelDecode


def _grouped(text):
    decoder = BaseRecLabelDecode()
    selection = np.ones(len(text), dtype=bool)
    word_list, _, state_list = decoder.get_word_info(text, selection)
    return ["".join(word) for word in word_list], state_list


def test_accented_latin_letters_stay_in_one_word():
    assert _grouped("Köln") == (["Köln"], ["en&num"])
    assert _grouped("Flächennutzung") == (["Flächennutzung"], ["en&num"])
    assert _grouped("Naturverhältnisse") == (["Naturverhältnisse"], ["en&num"])


def test_accents_inside_mixed_chinese_latin_text():
    words, states = _grouped("德语Flächennutzung处")
    assert words == ["德语", "Flächennutzung", "处"]
    assert states == ["cn", "en&num", "cn"]


def test_separators_still_split_words():
    words, states = _grouped("定金(订金)100元")
    assert words == ["定金", "(", "订金", ")", "100", "元"]
    assert states == ["cn", "symbol", "cn", "symbol", "en&num", "cn"]


def test_underscore_and_dash_grouping_unchanged():
    assert _grouped("hello_world") == (["hello", "_", "world"],
                                       ["en&num", "symbol", "en&num"])
    assert _grouped("VGG-16") == (["VGG-16"], ["en&num"])
