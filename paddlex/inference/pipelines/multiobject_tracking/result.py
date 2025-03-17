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

import cv2
import numpy as np

from ...utils.io import VideoReader
from ...common.result import BaseVideoResult


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def draw_tracking(
    image,
    track_res,
    frame_id=0,
):
    im = np.ascontiguousarray(np.copy(image))
    text_scale = max(0.5, image.shape[1] / 3000.0)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.0))

    cv2.putText(
        im,
        "frame: %d num: %d" % (frame_id, len(track_res)),
        (0, int(15 * text_scale) + 5),
        cv2.FONT_ITALIC,
        text_scale,
        (0, 0, 255),
        thickness=text_thickness,
    )

    for i, t in enumerate(track_res):
        x1, y1, x2, y2 = t["coordinate"]
        intbox = tuple(map(round, (x1, y1, x2, y2)))
        id_text = "{}_{}".format(t["label"], t["track_id"])

        color = get_color(abs(t["track_id"]))
        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
        )
        cv2.putText(
            im,
            id_text,
            (intbox[0], intbox[1] - 25),
            cv2.FONT_ITALIC,
            text_scale,
            color,
            thickness=text_thickness,
        )

        text = "score: {:.2f}".format(t["score"])
        cv2.putText(
            im,
            text,
            (intbox[0], intbox[1] - 6),
            cv2.FONT_ITALIC,
            text_scale,
            color,
            thickness=text_thickness,
        )
    return im


class MOTVideoResult(BaseVideoResult):

    def _to_video(self):
        """Draw label on image"""
        video_reader = VideoReader(backend="decord")
        video = video_reader.read(self["input_path"])
        video = list(video)
        write_fps = video_reader.get_fps()
        video_list = []
        assert len(video) == len(self["result"])

        for i in range(len(video)):
            image = video[i].asnumpy()
            results = self["result"][i]
            image = draw_tracking(image, results["track_res"], results["frame_id"])
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            video_list.append(image)
        return {"res": (np.array(video_list), write_fps)}
