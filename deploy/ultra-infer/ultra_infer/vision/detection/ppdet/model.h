// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "ultra_infer/vision/detection/ppdet/base.h"
#include "ultra_infer/vision/detection/ppdet/multiclass_nms.h"
#include "ultra_infer/vision/detection/ppdet/multiclass_nms_rotated.h"

namespace ultra_infer {
namespace vision {
namespace detection {

class ULTRAINFER_DECL PicoDet : public PPDetBase {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g picodet/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g picodet/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * config_file Path of configuration file for deployment, e.g
   * picodet/infer_cfg.yml \param[in] custom_option RuntimeOption for inference,
   * the default will use cpu, and choose the backend defined in
   * `valid_cpu_backends` \param[in] model_format Model format of the loaded
   * model, default is Paddle format
   */
  PicoDet(const std::string &model_file, const std::string &params_file,
          const std::string &config_file,
          const RuntimeOption &custom_option = RuntimeOption(),
          const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER,
                          Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_rknpu_backends = {Backend::RKNPU2};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
    valid_timvx_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PicoDet"; }
};

class ULTRAINFER_DECL SOLOv2 : public PPDetBase {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g picodet/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g picodet/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * config_file Path of configuration file for deployment, e.g
   * picodet/infer_cfg.yml \param[in] custom_option RuntimeOption for inference,
   * the default will use cpu, and choose the backend defined in
   * `valid_cpu_backends` \param[in] model_format Model format of the loaded
   * model, default is Paddle format
   */
  SOLOv2(const std::string &model_file, const std::string &params_file,
         const std::string &config_file,
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER, Backend::TRT};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "SOLOv2"; }
};

class ULTRAINFER_DECL PPYOLOE : public PPDetBase {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ppyoloe/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g picodet/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * config_file Path of configuration file for deployment, e.g
   * picodet/infer_cfg.yml \param[in] custom_option RuntimeOption for inference,
   * the default will use cpu, and choose the backend defined in
   * `valid_cpu_backends` \param[in] model_format Model format of the loaded
   * model, default is Paddle format
   */
  PPYOLOE(const std::string &model_file, const std::string &params_file,
          const std::string &config_file,
          const RuntimeOption &custom_option = RuntimeOption(),
          const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER,
                          Backend::LITE, Backend::TVM};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_timvx_backends = {Backend::LITE};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_rknpu_backends = {Backend::RKNPU2};
    valid_ascend_backends = {Backend::LITE};
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
    valid_horizon_backends = {Backend::HORIZONNPU};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PPYOLOE"; }
};

class ULTRAINFER_DECL PPYOLO : public PPDetBase {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ppyolo/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g ppyolo/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * config_file Path of configuration file for deployment, e.g
   * picodet/infer_cfg.yml \param[in] custom_option RuntimeOption for inference,
   * the default will use cpu, and choose the backend defined in
   * `valid_cpu_backends` \param[in] model_format Model format of the loaded
   * model, default is Paddle format
   */
  PPYOLO(const std::string &model_file, const std::string &params_file,
         const std::string &config_file,
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/PP-YOLO"; }
};

class ULTRAINFER_DECL YOLOv3 : public PPDetBase {
public:
  YOLOv3(const std::string &model_file, const std::string &params_file,
         const std::string &config_file,
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER,
                          Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv3"; }
};

class ULTRAINFER_DECL PaddleYOLOX : public PPDetBase {
public:
  PaddleYOLOX(const std::string &model_file, const std::string &params_file,
              const std::string &config_file,
              const RuntimeOption &custom_option = RuntimeOption(),
              const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER,
                          Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOX"; }
};

class ULTRAINFER_DECL FasterRCNN : public PPDetBase {
public:
  FasterRCNN(const std::string &model_file, const std::string &params_file,
             const std::string &config_file,
             const RuntimeOption &custom_option = RuntimeOption(),
             const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/FasterRCNN"; }
};

class ULTRAINFER_DECL MaskRCNN : public PPDetBase {
public:
  MaskRCNN(const std::string &model_file, const std::string &params_file,
           const std::string &config_file,
           const RuntimeOption &custom_option = RuntimeOption(),
           const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/MaskRCNN"; }
};

class ULTRAINFER_DECL SSD : public PPDetBase {
public:
  SSD(const std::string &model_file, const std::string &params_file,
      const std::string &config_file,
      const RuntimeOption &custom_option = RuntimeOption(),
      const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/SSD"; }
};

class ULTRAINFER_DECL PaddleYOLOv5 : public PPDetBase {
public:
  PaddleYOLOv5(const std::string &model_file, const std::string &params_file,
               const std::string &config_file,
               const RuntimeOption &custom_option = RuntimeOption(),
               const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv5"; }
};

class ULTRAINFER_DECL PaddleYOLOv6 : public PPDetBase {
public:
  PaddleYOLOv6(const std::string &model_file, const std::string &params_file,
               const std::string &config_file,
               const RuntimeOption &custom_option = RuntimeOption(),
               const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv6"; }
};

class ULTRAINFER_DECL PaddleYOLOv7 : public PPDetBase {
public:
  PaddleYOLOv7(const std::string &model_file, const std::string &params_file,
               const std::string &config_file,
               const RuntimeOption &custom_option = RuntimeOption(),
               const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv7"; }
};

class ULTRAINFER_DECL PaddleYOLOv8 : public PPDetBase {
public:
  PaddleYOLOv8(const std::string &model_file, const std::string &params_file,
               const std::string &config_file,
               const RuntimeOption &custom_option = RuntimeOption(),
               const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER,
                          Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_rknpu_backends = {Backend::RKNPU2};
    valid_ascend_backends = {Backend::LITE};
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv8"; }
};

class ULTRAINFER_DECL RTMDet : public PPDetBase {
public:
  RTMDet(const std::string &model_file, const std::string &params_file,
         const std::string &config_file,
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/RTMDet"; }
};

class ULTRAINFER_DECL CascadeRCNN : public PPDetBase {
public:
  CascadeRCNN(const std::string &model_file, const std::string &params_file,
              const std::string &config_file,
              const RuntimeOption &custom_option = RuntimeOption(),
              const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER};
    initialized = Initialize();
  }

  virtual std::string ModelName() const {
    return "PaddleDetection/CascadeRCNN";
  }
};

class ULTRAINFER_DECL PSSDet : public PPDetBase {
public:
  PSSDet(const std::string &model_file, const std::string &params_file,
         const std::string &config_file,
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/PSSDet"; }
};

class ULTRAINFER_DECL RetinaNet : public PPDetBase {
public:
  RetinaNet(const std::string &model_file, const std::string &params_file,
            const std::string &config_file,
            const RuntimeOption &custom_option = RuntimeOption(),
            const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/RetinaNet"; }
};

class ULTRAINFER_DECL PPYOLOESOD : public PPDetBase {
public:
  PPYOLOESOD(const std::string &model_file, const std::string &params_file,
             const std::string &config_file,
             const RuntimeOption &custom_option = RuntimeOption(),
             const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/PPYOLOESOD"; }
};

class ULTRAINFER_DECL FCOS : public PPDetBase {
public:
  FCOS(const std::string &model_file, const std::string &params_file,
       const std::string &config_file,
       const RuntimeOption &custom_option = RuntimeOption(),
       const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/FCOS"; }
};

class ULTRAINFER_DECL TTFNet : public PPDetBase {
public:
  TTFNet(const std::string &model_file, const std::string &params_file,
         const std::string &config_file,
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/TTFNet"; }
};

class ULTRAINFER_DECL TOOD : public PPDetBase {
public:
  TOOD(const std::string &model_file, const std::string &params_file,
       const std::string &config_file,
       const RuntimeOption &custom_option = RuntimeOption(),
       const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/TOOD"; }
};

class ULTRAINFER_DECL GFL : public PPDetBase {
public:
  GFL(const std::string &model_file, const std::string &params_file,
      const std::string &config_file,
      const RuntimeOption &custom_option = RuntimeOption(),
      const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/GFL"; }
};

class ULTRAINFER_DECL PaddleDetectionModel : public PPDetBase {
public:
  PaddleDetectionModel(const std::string &model_file,
                       const std::string &params_file,
                       const std::string &config_file,
                       const RuntimeOption &custom_option = RuntimeOption(),
                       const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    CheckArch();
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER,
                          Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_timvx_backends = {Backend::LITE};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_rknpu_backends = {Backend::RKNPU2};
    valid_ascend_backends = {Backend::LITE};
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetectionModel"; }
};

class ULTRAINFER_DECL PPYOLOER : public PPDetBase {
public:
  PPYOLOER(const std::string &model_file, const std::string &params_file,
           const std::string &config_file,
           const RuntimeOption &custom_option = RuntimeOption(),
           const ModelFormat &model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                  model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::OPENVINO, Backend::ORT,
                          Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
    valid_timvx_backends = {Backend::LITE};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_rknpu_backends = {Backend::RKNPU2};
    valid_ascend_backends = {Backend::LITE};
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PPYOLOER"; }
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
