## 2.1.1.1 下载Anaconda     
在Anaconda官网[(https://www.anaconda.com/distribution/)](https://www.anaconda.com/distribution/)选择“Windows”，并选择与所需python相对应的Anaconda版本进行下载（PaddlePaddle要求安装的Anaconda版本为64-bit）

## 2.1.1.2 安装Anaconda 
打开下载的安装包（以.exe为后缀），根据引导完成安装，在安装过程中可以修改安装路径，具体如下图所示：
<div align=center><img width="580" height="400" src="./pics/anaconda1.png"/></div>                  
【注意】默认安装在Windows当前用户主目录下           

## 2.1.1.3 使用Anaconda  

- 点击Windows系统左下角的Windows图标，打开：所有程序->Anaconda3/2（64-bit）->Anaconda Prompt      
- 在命令行中执行下述命令
```cmd
# 创建一个名为mypaddle的环境，指定python版本是3.5
conda create -n mypaddle python=3.5
# 创建好后，使用activate进入环境
conda activate mypaddle
python --version
# 若上述命令行出现Anaconda字样，则表示安装成功
# 退出环境
conda deactivate
```
