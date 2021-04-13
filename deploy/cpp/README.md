## 编译方法

1. 进入当前目录,即`PaddleX/deploy/cpp`
2. 下载Linux预测库 [下载地址](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html)(需高于2.0版本)
3. 执行编译脚本`sh script/build.sh`

## 待做事项
1. 代码目录待整理
- 删除cmake目录
- 单独提供方法编译生成lib
- 单独提供依赖lib的demo，提供两种方式编译demo
- 大量函数缺少错误处理返回


