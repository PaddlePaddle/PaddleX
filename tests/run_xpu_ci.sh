wget https://paddle-model-ecology.bj.bcebos.com/paddlex/tmp/PaddleTest.tar
tar -xf PaddleTest.tar && rm -rf  PaddleTest.tar 
xpu-smi 
python --version
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Xpu-P800-SelfBuiltPypiUse/latest/paddlepaddle_xpu-0.0.0-cp310-cp310-linux_x86_64.whl
python -c "import paddle; paddle.version.show()"
cp -r PaddleTest/models/PaddleX/ci ./
export DEVICE_ID=6,7
export PADDLE_PDX_DISABLE_DEV_MODEL_WL=true
export MEM_SIZE=32
export DEVICE_TYPE=xpu
rm -rf  ci/pr_list.txt
mv ci/pr_list_xpu.txt  ci/pr_list.txt
export PIP_DEFAULT_RETRIES=1
bash ci/ci_run.sh
