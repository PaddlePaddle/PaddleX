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


def constr_device(device_type, device_ids):
    device_ids = ",".join(map(str, device_ids))
    return f"{device_type}:{device_ids}"


def parse_device(device):
    parts = device.split(":")
    if len(parts) > 2:
        raise ValueError(f"Invalid device: {device}")
    if len(parts) == 1:
        device_type, device_ids = parts[0], None
    else:
        device_type, device_ids = parts
        device_ids = device_ids.split(",")
        for device_id in device_ids:
            if not device_id.isdigit():
                raise ValueError(f"Invalid device ID: {device_id}")
        device_ids = list(map(int, device_ids))
    device_type = device_type.lower()
    return device_type, device_ids
