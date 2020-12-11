# Labelme Installation and Startup

LabelMe is an open source annotation tool, which can be used to label object detection, instance segmentation and semantic segmentation.

## 1. Install Anaconda

Anaconda is recommended to install Python dependencies. Experienced developers can skip this step.For installation of anaconda, please refer to [Document](../../appendix/anaconda_install.md)。

After installing Anaconda and creating the environment, proceed to the next steps

## 2. Install Labelme

After entering Python environment, execute the following command
```
conda activate my_paddlex
conda install pyqt
pip install labelme
```

## 3. Start Labelme

Enter the Python environment where LabelMe is installed and execute the following command to start LabelMe
```
conda activate my_paddlex
labelme
```
