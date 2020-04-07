## 2.2.1.1 下载Anaconda         
Ubuntu图形界面下：在Anaconda官网[(https://www.anaconda.com/distribution/)](https://www.anaconda.com/distribution/)选择“Linux”，并选择与所需python相对应的Anaconda版本进行下载             
Ubuntu命令行界面下：使用”wget“进行下载
```cmd
# Anaconda2
wget https://repo.anaconda.com/archive/Anaconda2-2019.07-Linux-x86_64.sh --no-check-certificate
# Anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh --no-check-certificate
```
## 2.2.1.2 安装Anaconda 

***步骤一：安装***       
在Anaconda安装包所在路径执行下述命令行
```cmd
# 运行所下载的Anaconda，例如：
bash ./Anaconda3-2019.07-Linux-x86_64.sh
```
【注意】安装过程中一直回车即可，直至出现设置路径时可对安装路径进行修改，否则默认安装在Ubuntu当前用户主目录下        
***步骤二：设置环境变量***     
在命令行中执行下述命令
```cmd
# 将anaconda的bin目录加入PATH
# 根据安装路径的不同，修改”~/anaconda3/bin“
echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc
# 更新bashrc以立即生效
source ~/.bashrc
```
## 2.2.1.3 使用Anaconda         
在命令行中执行下述命令
```cmd
# 创建一个名为mypaddle的环境，指定python版本是3.5
conda create -n mypaddle python=3.5
# 创建好后，使用activate进入环境
source activate mypaddle
python --version
# 若上述命令行出现Anaconda字样，则表示安装成功
# 退出环境
source deactivate
```
