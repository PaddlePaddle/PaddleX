# Set huawei ascend toolkit correctly.
HUAWEI_ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit/latest"
HUAWEI_ASCEND_DRIVER_PATH="/usr/local/Ascend/driver"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HUAWEI_ASCEND_DRIVER_PATH/lib64/driver:$HUAWEI_ASCEND_DRIVER_PATH/lib64:$HUAWEI_ASCEND_DRIVER_PATH/lib64/stub:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/lib64:$HUAWEI_ASCEND_TOOLKIT_HOME/opp/op_proto/built-in
export PYTHONPATH=$PYTHONPATH:$HUAWEI_ASCEND_TOOLKIT_HOME/fwkacllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/acllib/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/python/site-packages:$HUAWEI_ASCEND_TOOLKIT_HOME/pyACL/python/site-packages/acl
export PATH=$PATH:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/ccec_compiler/bin:${HUAWEI_ASCEND_TOOLKIT_HOME}/acllib/bin:$HUAWEI_ASCEND_TOOLKIT_HOME/atc/bin
export ASCEND_AICPU_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME
export ASCEND_OPP_PATH=$HUAWEI_ASCEND_TOOLKIT_HOME/opp
export TOOLCHAIN_HOME=$HUAWEI_ASCEND_TOOLKIT_HOME/toolkit
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3

echo "===== Finish Initializing Environment for Ascend Deployment ====="
