#OpenVINO部署常见问题

## Q1转模型过程中出现"ModuleNotFoundError: No module named 'mo'"  

原因：该问题主要是因为在安装OpenVINO之后未初始化OpenVINO环境  
解决方案：找到OpenVINO初始化环境脚本，运行后即可以解决此问题  

### Linux系统初始化OpenVINO环境
1)root用户安装，以OpenVINO 2021.1版本为例，运行如下命令即可初始化  

```
source /opt/intel/openvino_2021/bin/setupvars.sh
```
  
2)非root用户安装，以OpenVINO 2021.1版本、用户名为paddlex为例，运行如下命令即可初始化
  
```
source /home/paddlex/intel/openvino_2021/bin/setupvar.sh
```
  
### Window系统初始化OpenVINO环境
以OpenVINO 2021.1版本为例，执行如下命令即可初始化OpenVINO环境  

```
cd C:\Program Files (x86)\Intel\openvino_2021\bin\
setupvars.bat
```

**说明**：更多初始化OpenVINO环境的细节请参考[OpenVINO官网](https://docs.openvinotoolkit.org/latest/index.html)
