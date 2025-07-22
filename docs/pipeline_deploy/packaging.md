# 打包可执行文件指南

本说明适用于通过PyInstaller打包PaddleX项目。

> 由于Nuikta的打包原理与PaddleX不适配，当前暂不支持通过Nuikta进行打包。

## 准备环境

- **根据[PaddleX安装文档](../installation/installation.md)完成安装**

- **安装PyInstaller**

安装PyInstaller：

```bash
pip install pyinstaller
```
> 请确认用于打包的环境与当前准备环境一致，以避免因依赖差异导致打包后的程序出现异常。

## 打包脚本

将下方python脚本拷贝后存成py文件，文件名可以为install_script.py。

```python
import paddlex
import importlib.metadata
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True, help='Your file name, e.g. main.py.')
parser.add_argument('--nvidia', action='store_true', help='Whether to include NVIDIA CUDA and cuDNN dependencies. Default is false.')

args = parser.parse_args()

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
    result = subprocess.run((final_cmd), check=True)
except subprocess.CalledProcessError as e:
    print("Installation failed:", e)
    sys.exit(1)
```


### 打包脚本参数

| 参数         | 是否必需 | 说明                                                                                                               | 默认值   |
|--------------|----------|--------------------------------------------------------------------------------------------------------------------|---------|
| --file   | 必须     | 你的打包文件名（如 main.py）。                                                                                        |       |
| --nvidia     | 可选     | 是否将NVIDIA的CUDA、cuDNN相关依赖库一同打包到可执行文件目录下。如果系统环境变量路径已包含NVIDIA的CUDA、cuDNN相关依赖库则无需开启。 | False   |

### 打包脚本调用示例

```bash
python install_script.py --file main.py
python install_script.py --file main.py --nvidia
```

### 运行结果

- 安转脚本将执行类似如下命令：

    pyinstaller main.py --collect-data paddlex --collect-binaries paddle [--copy-metadata xxx …]，其中--copy-metadata xxx 会根据当前环境已安装的PaddleX需要的依赖动态添加。

- 可执行文件将生成在当前路径的dist文件夹中，包含可执行文件和相关打包依赖库。

## 附录

### 测试环境

- **操作系统：Win 11**

- **Python：3.10.18**

- **PaddlePaddle：3.0.0**

- **PaddleX：3.1.3**

- **PyInstaller：6.14.2**

### 常见问题

- 如果出报错信息出现 <code>RuntimeError: xxx requires additional dependencies</code> ，请确认已按照准备环境部分说明正确安装环境。

- 如果报错信息出现CUDA、cuDNN相关动态链接库找不到，请检查系统环境变量中是否正确添加NVIDIA的CUDA、cuDNN相关依赖库路径或者考虑在运行打包脚本时添加 <code>--nvidia</code> ，将NVIDIA的CUDA、cuDNN相关依赖库打包进可执行文件目录中。
