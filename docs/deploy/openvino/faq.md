# FAQ

## Q1:"ModuleNotFoundError: No module named 'mo'"  
cause: the main reason for this problem is that the openvino environment is not initialized after installing openvino  

Solution: find the openvino initialization environment script and run it to solve this problem

### Linux system initializes openvino environment

1)Root user installation, take openvino 2021.1 as an example, run the following command to initialize  

```
source /opt/intel/openvino_2021/bin/setupvars.sh
```
  
2)For non root user installation, take openvino 2021.1 with username paddlex as an example, run the following command to initialize  

```
source /home/paddlex/intel/openvino_2021/bin/setupvar.sh
```

### Window system initializes openvino environment
Take openvino 2021.1 as an example, execute the following command to initialize the openvino environment  
```
cd C:\Program Files (x86)\Intel\openvino_2021\bin\
setupvars.bat
```  

**Note**: for more details about initializing openvino environment, please refer to [openvino official website](https://docs.openvinotoolkit.org/latest/index.html)


## Q2:"convert failed, please export paddle inference model by fixed_input_shape"

Reason: the problem is that --fixed_input_shape is not added when exporting the paddle inference model
  
Solution: add --fixed_input_shape when exporting the paddle inference model

[how to export paddle inference model](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/deploy/export_model.md)
