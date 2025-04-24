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

pipeline = create_pipeline(pipeline="PP-ShiTuV2")

index_data = pipeline.build_index(
    gallery_imgs="drink_dataset_v2.0/", gallery_label="drink_dataset_v2.0/gallery.txt"
)
index_data.save("drink_index")

output = pipeline.predict("./drink_dataset_v2.0/test_images/", index=index_data)
for res in output:
    res.print()
    res.save_to_img("./output/")
