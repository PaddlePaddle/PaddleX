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

from typing import List

import requests

from .hub_config import config


class ServerConnectionError(Exception):
    def __init__(self, url: str):
        self.url = url

    def __str__(self):
        tips = "Can't connect to UltraInfer Model Server: {}".format(self.url)
        return tips


class ModelServer(object):
    """
    UltraInfer server source

    Args:
        url(str) : Url of the server
        timeout(int) : Request timeout
    """

    def __init__(self, url: str, timeout: int = 10):
        self._url = url
        self._timeout = timeout

    def search_model(
        self, name: str, format: str = None, version: str = None
    ) -> List[dict]:
        """
        Search model from model server.

        Args:
            name(str) : UltraInfer model name
            format(str): UltraInfer model format
            version(str) : UltraInfer model version
        Return:
            result(list): search results
        """
        params = {}
        params["name"] = name
        if format:
            params["format"] = format
        if version:
            params["version"] = version
        result = self.request(path="ultra_infer_search", params=params)
        if result["status"] == 0 and len(result["data"]) > 0:
            return result["data"]
        return None

    def stat_model(self, name: str, format: str, version: str):
        """
        Note a record when download a model for statistics.

        Args:
            name(str) : UltraInfer model name
            format(str): UltraInfer model format
            version(str) : UltraInfer model version
        Return:
            is_successful(bool): True if successful, False otherwise
        """
        params = {}
        params["name"] = name
        params["format"] = format
        params["version"] = version
        params["from"] = "ultra_infer"
        try:
            result = self.request(path="stat", params=params)
        except Exception:
            return False
        if result["status"] == 0:
            return True
        else:
            return False

    def request(self, path: str, params: dict) -> dict:
        """Request server."""
        api = "{}/{}".format(self._url, path)
        try:
            result = requests.get(api, params, timeout=self._timeout)
            return result.json()
        except requests.exceptions.ConnectionError as e:
            raise ServerConnectionError(self._url)

    def get_model_list(self):
        """
        Get all pre-trained models information in dataset.
        Return:
            result(dict): key is category name, value is a list which contains models \
                information such as name, format and version.
        """
        api = "{}/{}".format(self._url, "ultra_infer_listmodels")
        try:
            result = requests.get(api, timeout=self._timeout)
            return result.json()
        except requests.exceptions.ConnectionError as e:
            raise ServerConnectionError(self._url)

    def is_connected(self):
        return self.check(self._url)

    @classmethod
    def check(cls, url: str) -> bool:
        """
        Check if the specified url is a valid model server

        Args:
            url(str) : Url to check
        """
        try:
            r = requests.get(url + "/search")
            return r.status_code == 200
        except:
            return False


model_server = ModelServer(config.server)
