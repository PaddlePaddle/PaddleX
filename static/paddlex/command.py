# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from six import text_type as _text_type
import argparse
import sys
import os
import os.path as osp
import paddlex.utils.logging as logging


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        "-m",
        type=_text_type,
        default=None,
        help="define model directory path")
    parser.add_argument(
        "--save_dir",
        "-s",
        type=_text_type,
        default=None,
        help="path to save inference model")
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of PaddleX")
    parser.add_argument(
        "--export_inference",
        "-e",
        action="store_true",
        default=False,
        help="export inference model for C++/Python deployment")
    parser.add_argument(
        "--export_onnx",
        "-eo",
        action="store_true",
        default=False,
        help="export onnx model for deployment")
    parser.add_argument(
        "--onnx_opset",
        "-oo",
        type=int,
        default=10,
        help="when use paddle2onnx, set onnx opset version to export")
    parser.add_argument(
        "--data_conversion",
        "-dc",
        action="store_true",
        default=False,
        help="convert the dataset to the standard format")
    parser.add_argument(
        "--source",
        "-se",
        type=_text_type,
        default=None,
        help="define dataset format before the conversion")
    parser.add_argument(
        "--to",
        "-to",
        type=_text_type,
        default=None,
        help="define dataset format after the conversion")
    parser.add_argument(
        "--pics",
        "-p",
        type=_text_type,
        default=None,
        help="define pictures directory path")
    parser.add_argument(
        "--annotations",
        "-a",
        type=_text_type,
        default=None,
        help="define annotations directory path")
    parser.add_argument(
        "--fixed_input_shape",
        "-fs",
        default=None,
        help="export inference model with fixed input shape:[w,h]")
    parser.add_argument(
        "--split_dataset",
        "-sd",
        action="store_true",
        default=False,
        help="split dataset with the split value")
    parser.add_argument(
        "--format",
        "-f",
        default=None,
        help="define dataset format(ImageNet/COCO/VOC/Seg)")
    parser.add_argument(
        "--dataset_dir",
        "-dd",
        type=_text_type,
        default=None,
        help="define the path of dataset to be splited")
    parser.add_argument(
        "--val_value",
        "-vv",
        default=None,
        help="define the value of validation dataset(E.g 0.2)")
    parser.add_argument(
        "--test_value",
        "-tv",
        default=None,
        help="define the value of test dataset(E.g 0.1)")

    return parser


def main():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    import paddlex as pdx

    if len(sys.argv) < 2:
        print("Use command 'paddlex -h` to print the help information\n")
        return
    parser = arg_parser()
    args = parser.parse_args()

    if args.version:
        print("PaddleX-{}".format(pdx.__version__))
        print("Repo: https://github.com/PaddlePaddle/PaddleX.git")
        print("Email: paddlex@baidu.com")
        return

    if args.export_inference:
        assert args.model_dir is not None, "--model_dir should be defined while exporting inference model"
        assert args.save_dir is not None, "--save_dir should be defined to save inference model"

        fixed_input_shape = None
        if args.fixed_input_shape is not None:
            fixed_input_shape = eval(args.fixed_input_shape)
            assert len(
                fixed_input_shape
            ) == 2, "len of fixed input shape must == 2, such as [224,224]"
        else:
            fixed_input_shape = None

        model = pdx.load_model(args.model_dir, fixed_input_shape)
        model.export_inference_model(args.save_dir)

    if args.export_onnx:
        assert args.model_dir is not None, "--model_dir should be defined while exporting onnx model"
        assert args.save_dir is not None, "--save_dir should be defined to create onnx model"

        model = pdx.load_model(args.model_dir)

        if model.status == "Normal" or model.status == "Prune":
            logging.error(
                "Only support inference model, try to export model first as below,",
                exit=False)
            logging.error(
                "paddlex --export_inference --model_dir model_path --save_dir infer_model"
            )
        save_file = os.path.join(args.save_dir, 'paddle2onnx_model.onnx')
        pdx.converter.export_onnx_model(model, save_file, args.onnx_opset)

    if args.data_conversion:
        assert args.source is not None, "--source should be defined while converting dataset"
        assert args.to is not None, "--to should be defined to confirm the taregt dataset format"
        assert args.pics is not None, "--pics should be defined to confirm the pictures path"
        assert args.annotations is not None, "--annotations should be defined to confirm the annotations path"
        assert args.save_dir is not None, "--save_dir should be defined to store taregt dataset"
        if args.source not in ['labelme', 'jingling', 'easydata']:
            logging.error(
                "The source format {} is not one of labelme/jingling/easydata".
                format(args.source),
                exit=False)
        if args.to not in ['PascalVOC', 'MSCOCO', 'SEG', 'ImageNet']:
            logging.error(
                "The to format {} is not one of PascalVOC/MSCOCO/SEG/ImageNet".
                format(args.to),
                exit=False)
        if args.source == 'labelme' and args.to == 'ImageNet':
            logging.error(
                "The labelme dataset can not convert to the ImageNet dataset.",
                exit=False)
        if args.source == 'jingling' and args.to == 'PascalVOC':
            logging.error(
                "The jingling dataset can not convert to the PascalVOC dataset.",
                exit=False)
        if not osp.exists(args.save_dir):
            os.makedirs(args.save_dir)
        pdx.tools.convert.dataset_conversion(args.source, args.to, args.pics,
                                             args.annotations, args.save_dir)

    if args.split_dataset:
        assert args.dataset_dir is not None, "--dataset_dir should be defined while spliting dataset"
        assert args.format is not None, "--format should be defined while spliting dataset"
        assert args.val_value is not None, "--val_value should be defined while spliting dataset"

        dataset_dir = args.dataset_dir
        dataset_format = args.format.lower()
        val_value = float(args.val_value)
        test_value = float(args.test_value
                           if args.test_value is not None else 0)
        save_dir = dataset_dir

        if not dataset_format in ["coco", "imagenet", "voc", "seg"]:
            logging.error(
                "The dataset format is not correct defined.(support COCO/ImageNet/VOC/Seg)"
            )
        if not osp.exists(dataset_dir):
            logging.error("The path of dataset to be splited doesn't exist.")
        if val_value <= 0 or val_value >= 1 or test_value < 0 or test_value >= 1 or val_value + test_value >= 1:
            logging.error("The value of split is not correct.")

        pdx.tools.split.dataset_split(dataset_dir, dataset_format, val_value,
                                      test_value, save_dir)

    


if __name__ == "__main__":
    main()
