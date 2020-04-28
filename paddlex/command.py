from six import text_type as _text_type
import argparse
import sys


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
        model = pdx.load_model(args.model_dir)
        model.export_inference_model(args.save_dir)


if __name__ == "__main__":
    main()
