# copy to /usr/local/neuware
dir1=/opt/code_chap_5_student/env/neuware/include
dir2=/opt/code_chap_5_student/env/neuware/lib64

cp $dir1/*.h /usr/local/neuware/include
cp $dir2/libcnml.so /usr/local/neuware/lib64
cp $dir2/libcnplugin.so /usr/local/neuware/lib64
cp $dir2/libcnrt.so /usr/local/neuware/lib64