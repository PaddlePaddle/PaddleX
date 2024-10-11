from pathlib import Path
from .base import BaseResult, CVResult


class SealOCRResult(CVResult):
    """SealOCRResult"""

    def save_to_img(self, save_path):
        if not save_path.lower().endswith((".jpg", ".png")):
            input_path = self["input_path"]
            save_path = Path(save_path) / f"{Path(input_path).stem}"
        else:
            save_path = Path(save_path).stem
        layout_save_path = f"{save_path}_layout.jpg"
        layout_result = self["layout_result"]
        layout_result.save_to_img(layout_save_path)
        for idx, seal_result in enumerate(self["ocr_result"]):
            ocr_save_path = f"{save_path}_{idx}_seal_ocr.jpg"
            seal_result.save_to_img(ocr_save_path)