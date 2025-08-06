import json
import time
import uuid

from paddlex import create_pipeline
from paddlex.inference import load_pipeline_config
from paddlex.utils.device import constr_device
from pydantic import ValidationError

from . import constants, env, logging, protocol
from .config import create_app_config
from .lazy_mods import pb_utils


class BaseTritonPythonModel(object):
    @property
    def id(self):
        # TODO: Add model name
        return self._id

    @property
    def pipeline_creation_kwargs(self):
        return {}

    def initialize(self, args):
        self.triton_model_config = json.loads(args["model_config"])
        logging.info("Triton model config: %s", self.triton_model_config)

        self.input_names = []
        for input_config in self.triton_model_config["input"]:
            self.input_names.append(input_config["name"])
        if self.input_names != [constants.INPUT_NAME]:
            raise pb_utils.TritonModelException(
                f"Expected one and only one input named {repr(constants.INPUT_NAME)}, but got input names: {self.input_names}"
            )
        logging.info("Input names: %s", self.input_names)

        self.output_names = []
        self.output_dtypes = []
        for output_config in self.triton_model_config["output"]:
            self.output_names.append(output_config["name"])
            self.output_dtypes.append(output_config["data_type"])
        if self.output_names != [constants.OUTPUT_NAME]:
            raise pb_utils.TritonModelException(
                f"Expected one and only one output named {repr(constants.OUTPUT_NAME)}, but got output names: {self.output_names}"
            )
        logging.info("Output names: %s", self.output_names)

        if args["model_instance_kind"] == "GPU":
            self._device_type = "gpu"
            self._device_id = int(args["model_instance_device_id"])
        elif args["model_instance_kind"] == "CPU":
            self._device_type = "cpu"
            self._device_id = None
        else:
            raise pb_utils.TritonModelException(
                f"Unsupported model instance kind: {args['model_instance_kind']}"
            )

        self.pipeline_config = load_pipeline_config(env.PIPELINE_CONFIG_PATH)
        use_hpip = env.USE_HPIP == "1"
        self.pipeline = self._create_pipeline(self.pipeline_config, use_hpip)
        self.app_config = create_app_config(self.pipeline_config)

        self._id = self._generate_model_id()
        logging.info("%s initialized successfully", self.id)

    def execute(self, requests):
        responses = []

        for request in requests:
            log_id = protocol.generate_log_id()
            tokens = logging.set_context_vars(self.id, log_id)

            start_time = time.perf_counter()
            try:
                input_ = pb_utils.get_input_tensor_by_name(
                    request, constants.INPUT_NAME
                )
                input_ = input_.as_numpy()
                input_model_type = self.get_input_model_type()
                try:
                    input_ = protocol.parse_triton_input(input_, input_model_type)
                except ValidationError as e:
                    output = protocol.create_aistudio_output_without_result(422, str(e))
                else:
                    try:
                        result_or_output = self.run(input_, log_id)
                    except Exception as e:
                        logging.error("Unhandled exception", exc_info=e)
                        output = protocol.create_aistudio_output_without_result(
                            500, "Internal server error", log_id=log_id
                        )
                    else:
                        result_model_type = self.get_result_model_type()
                        if isinstance(result_or_output, result_model_type):
                            output = protocol.create_aistudio_output_with_result(
                                result_or_output, log_id=log_id
                            )
                        else:
                            output = result_or_output
                output = protocol.create_triton_output(output)
                output = pb_utils.Tensor(constants.OUTPUT_NAME, output)
                response = pb_utils.InferenceResponse(output_tensors=[output])
            except Exception as e:
                logging.error("Unhandled exception", exc_info=e)
                response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError("An error occurred during execution"),
                )
            finally:
                end_time = time.perf_counter()
                logging.info("Time taken: %.3f ms", (end_time - start_time) * 1000)
                logging.reset_context_vars(*tokens)

            responses.append(response)

        return responses

    def get_input_model_type(self):
        raise NotImplementedError

    def get_result_model_type(self):
        raise NotImplementedError

    def run(self, input, log_id):
        raise NotImplementedError

    def _create_pipeline(self, config, use_hpip):
        if self._device_id is not None:
            device = constr_device(self._device_type, [self._device_id])
        else:
            device = self._device_type
        pipeline = create_pipeline(
            config=config,
            device=device,
            use_hpip=use_hpip,
            **self.pipeline_creation_kwargs,
        )
        return pipeline

    def _generate_model_id(self):
        return uuid.uuid4().hex
