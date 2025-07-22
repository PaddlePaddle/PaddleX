# 打包可执行文件说明

本说明适用于通过Pyinstaller打包PaddleX项目。
对于Nuitka由于其打包流程问题暂未支持。

## 环境准备

- **已根据[paddlex安装文档](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)完成安装**
- **已安装 pyinstaller**

安装 pyinstaller：

```bash
pip install pyinstaller
```

## 安装脚本

```bash
import paddlex
import importlib.metadata
import argparse
import subprocess
import shlex
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', required=True, help='Please provide your file name, e.g. main.py.')
parser.add_argument('--nvidia', type=lambda x: x.lower() == 'true', default=False, help='Whether to add NVIDIA dependencies, default is false.')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

try:
    args = parser.parse_args()
except SystemExit:
    parser.print_help()
    sys.exit(2)
except Exception as e:
    print(e)
    parser.print_help()
    sys.exit(1)

main_file = args.file

user_deps = [dist.metadata["Name"] for dist in importlib.metadata.distributions()]
deps_all = list(paddlex.utils.deps.DEP_SPECS.keys())
deps_need = [dep for dep in user_deps if dep in deps_all]

cmd = [
    f"pyinstaller {main_file}",
    "--collect-data paddlex",
    "--collect-binaries paddle"
]
if args.nvidia:
    cmd.append("--collect-binaries nvidia")

for dep in deps_need:
    cmd.append(f"--copy-metadata {dep}")

final_cmd = " ".join(cmd)
print(final_cmd)

try:
    result = subprocess.run(shlex.split(final_cmd), check=True)
except subprocess.CalledProcessError as e:
    print("Installation failed:", e)
```
## 测试环境

- **操作系统：Win 11**
- **Python：3.10.18**
- **PaddlePaddle：3.0.0**
- **PaddleX：3.1.3**
- **Pyinstaller：6.14.2**

## 使用说明

### 脚本参数

- **--file 或 -f**

    必选。你的打包文件名（如 main.py）。

-  **--nvidia**

    可选项。是否将 NVIDIA 相关依赖文件一同打包到可执行文件目录下。默认为 False。如果您的系统环境变量路径中已包含 NVIDIA 相应动态链接库的路径，则无需开启此选项。

### 安装脚本调用示例

```bash
python install_script.py --file main.py
python install_script.py --file main.py --nvidia true
```

### 运行结果

- 安转脚本将执行类似如下命令：

    pyinstaller main.py --collect-data paddlex --collect-binaries paddle [--copy-metadata xxx …]，其中 --copy-metadata xxx 会根据当前环境已安装的paddlex需要的依赖动态添加。

- 可执行文件将生成在当前路径的dist文件夹中，包含可执行文件和相关打包依赖库。

## 常见问题

- 如果出报错信息出现<code>RuntimeError: xxx requires additional dependencies</code>，确认已按照环境准备部分说明正确安装环境。

- 如果报错信息出现 CUDA、cuDNN 相关动态链接库找不到时，请检查系统环境变量中是否正确添加NVIDIA相关库路径或者考虑在运行安装脚本时添加<code>--nvidia true</code>，将NVIDIA相关依赖打包进可执行文件目录中。
