# copy to tensorflow
dir3=/opt/code_chap_5_student/code_chap_5_1_student/src/tf-implementation/tf-add-power-diff
dir4=/opt/code_chap_5_student/env/tensorflow-v1.10/tensorflow

cp $dir3/cwise_op_power_difference* $dir4/core/kernels/
cp $dir3/BUILD $dir4/core/kernels/
cp $dir3/mlu_stream.h $dir4/stream_executor/mlu/
cp $dir3/mlu_lib_ops.* $dir4/stream_executor/mlu/mlu_api/lib_ops/
cp $dir3/mlu_ops.h $dir4/stream_executor/mlu/mlu_api/ops/
cp $dir3/power_difference.cc $dir4/stream_executor/mlu/mlu_api/ops/
cp $dir3/math_ops.cc $dir4/core/ops/