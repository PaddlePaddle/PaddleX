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

from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="video_classification")

output = pipeline.predict("./test_samples/general_video_classification_001.mp4", topk=8)
# input_list = ["./test_samples/general_video_classification_001.mp4",
#                 "./test_samples/__lt03EF4ao.mkv",
#                 "./test_samples/__u6odV5hrI.mp4",
#                 "./test_samples/_8c2EG7JDOw.mp4.webm",]
# output = pipeline.predict(input_list, topk=5)

for res in output:
    print(res)
    res.print()  ## 打印预测的结构化输出
    res.save_to_video("./output/1.mp4")  ## 保存结果可视化视频
    res.save_to_video("./output/")  ## 保存结果可视化视频
    res.save_to_json("./output/")  ## 保存预测的结构化输出
