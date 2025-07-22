from paddlex_hps_server import BaseTritonPythonModel
from paddlex_hps_server.storage import create_storage

# Do we need a lock?
DEFAULT_INDEX_DIR = ".indexes"


class BaseFaceRecognitionModel(BaseTritonPythonModel):
    def initialize(self, args):
        super().initialize(args)
        self.context = {}
        if self.app_config.extra and "index_storage" in self.app_config.extra:
            self.context["index_storage"] = create_storage(
                self.app_config.extra["index_storage"]
            )
        else:
            self.context["index_storage"] = create_storage(
                {"type": "file_system", "directory": DEFAULT_INDEX_DIR}
            )
