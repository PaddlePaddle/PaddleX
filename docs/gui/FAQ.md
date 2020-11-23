# PaddleX客户端常见问题

## 1. 训练出错，提示『训练任务异常中止，请查阅错误日志具体确认原因』？
请按照以下步骤来查找原因

- 1.首先根据提示，找到错误日志，根据日志提示判断原因
- 2.如无法确定原因，测a)尝试重新训练，看是否能正常训练； b)调低batchsize（同时按比例调低学习率）排除是否是显存不足原因导致
- 3.如第2步仍然失败，请前往GitHub提ISSUE，a) 描述清楚问题 b) 贴上训练参数截图 c) 附上错误日志   https://github.com/PaddlePaddle/PaddleX/issues
- 4.如无Github帐号，则可加QQ群1045148026在线咨询工程师

## 2. 没有使用GPU，使用CPU，错误日志仍然提示"cuda error"
部分Windows机型由于GPU型号或驱动较老，导致训练时无法使用GPU，还会导致使用不了CPU，针对此情况，可采取如下方式解决
- 1.在PaddleX客户端安装目录下，删除"paddle"文件夹
- 2.下载paddlepaddle-cpu（压缩文件可在[百度网盘](https://pan.baidu.com/s/1GrzLCuzuw-PAEx4BELnc0w)下载，提取码iaor，约57M)，下载解压后，将目前中的paddle文件夹拷贝至PaddleX客户端安装目录下即可
- 3.重新启动PaddleX客户端，替换后客户端仅支持使用CPU训练模型

## 3. 如何升级PaddleX客户端
PaddleX客户端目前需用户手动下载最新版升级，旧版本可直接删除原安装目录即可。升级前请备份好工作空间下的3个workspace.*.pb文件，避免升级过程可能导致项目信息丢失。

PaddleX更新历史和下载地址: https://www.paddlepaddle.org.cn/paddlex/download
