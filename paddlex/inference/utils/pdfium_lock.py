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

"""
Global lock for pypdfium2 operations.

PDFium is inherently not thread-safe. It is not allowed to call pdfium
functions simultaneously across different threads, not even with different
documents. Simultaneous pdfium calls across threads will crash or corrupt
the process.

See: https://pypdfium2.readthedocs.io/en/stable/python_api.html

This module provides a global lock that must be used to serialize all
pypdfium2 operations across the application.
"""

import threading

pdfium_lock = threading.Lock()
