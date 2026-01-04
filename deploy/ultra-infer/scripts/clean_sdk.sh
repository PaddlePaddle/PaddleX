origin_install_dir=$1

rm -rf $origin_install_dir/include/onnx $origin_install_dir/include/paddle2onnx
mv $origin_install_dir/lib $origin_install_dir/lib_bak
mkdir $origin_install_dir/lib
cp $origin_install_dir/lib_bak/*ultra_infer* $origin_install_dir/lib
rm -rf $origin_install_dir/lib_bak
