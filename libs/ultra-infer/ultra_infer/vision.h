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

#include "ultra_infer/core/config.h"
#ifdef ENABLE_VISION
#include "ultra_infer/vision/classification/contrib/resnet.h"
#include "ultra_infer/vision/classification/contrib/yolov5cls/yolov5cls.h"
#include "ultra_infer/vision/classification/ppcls/model.h"
#include "ultra_infer/vision/classification/ppshitu/ppshituv2_det.h"
#include "ultra_infer/vision/classification/ppshitu/ppshituv2_rec.h"
#include "ultra_infer/vision/detection/contrib/fastestdet/fastestdet.h"
#include "ultra_infer/vision/detection/contrib/nanodet_plus.h"
#include "ultra_infer/vision/detection/contrib/rknpu2/model.h"
#include "ultra_infer/vision/detection/contrib/scaledyolov4.h"
#include "ultra_infer/vision/detection/contrib/yolor.h"
#include "ultra_infer/vision/detection/contrib/yolov5/yolov5.h"
#include "ultra_infer/vision/detection/contrib/yolov5lite.h"
#include "ultra_infer/vision/detection/contrib/yolov5seg/yolov5seg.h"
#include "ultra_infer/vision/detection/contrib/yolov6.h"
#include "ultra_infer/vision/detection/contrib/yolov7/yolov7.h"
#include "ultra_infer/vision/detection/contrib/yolov7end2end_ort.h"
#include "ultra_infer/vision/detection/contrib/yolov7end2end_trt.h"
#include "ultra_infer/vision/detection/contrib/yolov8/yolov8.h"
#include "ultra_infer/vision/detection/contrib/yolox.h"
#include "ultra_infer/vision/detection/ppdet/model.h"
#include "ultra_infer/vision/facealign/contrib/face_landmark_1000.h"
#include "ultra_infer/vision/facealign/contrib/pfld.h"
#include "ultra_infer/vision/facealign/contrib/pipnet.h"
#include "ultra_infer/vision/facedet/contrib/centerface/centerface.h"
#include "ultra_infer/vision/facedet/contrib/retinaface.h"
#include "ultra_infer/vision/facedet/contrib/scrfd.h"
#include "ultra_infer/vision/facedet/contrib/ultraface.h"
#include "ultra_infer/vision/facedet/contrib/yolov5face.h"
#include "ultra_infer/vision/facedet/contrib/yolov7face/yolov7face.h"
#include "ultra_infer/vision/facedet/ppdet/blazeface/blazeface.h"
#include "ultra_infer/vision/faceid/contrib/adaface/adaface.h"
#include "ultra_infer/vision/faceid/contrib/insightface/model.h"
#include "ultra_infer/vision/generation/contrib/animegan.h"
#include "ultra_infer/vision/headpose/contrib/fsanet.h"
#include "ultra_infer/vision/keypointdet/pptinypose/pptinypose.h"
#include "ultra_infer/vision/matting/contrib/modnet.h"
#include "ultra_infer/vision/matting/contrib/rvm.h"
#include "ultra_infer/vision/matting/ppmatting/ppmatting.h"
#include "ultra_infer/vision/ocr/ppocr/classifier.h"
#include "ultra_infer/vision/ocr/ppocr/dbcurvedetector.h"
#include "ultra_infer/vision/ocr/ppocr/dbdetector.h"
#include "ultra_infer/vision/ocr/ppocr/ppocr_v2.h"
#include "ultra_infer/vision/ocr/ppocr/ppocr_v3.h"
#include "ultra_infer/vision/ocr/ppocr/ppocr_v4.h"
#include "ultra_infer/vision/ocr/ppocr/ppstructurev2_layout.h"
#include "ultra_infer/vision/ocr/ppocr/ppstructurev2_table.h"
#include "ultra_infer/vision/ocr/ppocr/recognizer.h"
#include "ultra_infer/vision/ocr/ppocr/structurev2_layout.h"
#include "ultra_infer/vision/ocr/ppocr/structurev2_ser_vi_layoutxlm.h"
#include "ultra_infer/vision/ocr/ppocr/structurev2_table.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"
#include "ultra_infer/vision/ocr/ppocr/uvdocwarpper.h"
#include "ultra_infer/vision/perception/paddle3d/caddn/caddn.h"
#include "ultra_infer/vision/perception/paddle3d/centerpoint/centerpoint.h"
#include "ultra_infer/vision/perception/paddle3d/petr/petr.h"
#include "ultra_infer/vision/perception/paddle3d/smoke/smoke.h"
#include "ultra_infer/vision/segmentation/ppseg/model.h"
#include "ultra_infer/vision/sr/ppsr/model.h"
#include "ultra_infer/vision/tracking/pptracking/model.h"

#endif

#include "ultra_infer/vision/visualize/visualize.h"
