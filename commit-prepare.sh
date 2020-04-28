path=$(cd `dirname $0`; pwd)
cd $path

pip install pre-commit
pip install yapf
pre-commit install
