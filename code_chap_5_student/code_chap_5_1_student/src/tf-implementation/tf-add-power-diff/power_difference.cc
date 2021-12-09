/*Copyright 2018 Cambricon*/
/*
  powerdifference内容的代码补全受https://sanzo.top/AI-Computing-Systems/lab05/启发
  参照env/tensorflow-v1.10/tensorflow/stream_executor/mlu/mlu_api/ops/目录下的squared_difference.cc文件进行补全
*/
#if CAMBRICON_MLU

#include "tensorflow/stream_executor/mlu/mlu_api/lib_ops/mlu_lib_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/ops/mlu_ops.h"
#include "tensorflow/stream_executor/mlu/mlu_api/tf_mlu_intf.h"

#define INVALID_INDEX -1

namespace stream_executor {
namespace mlu {
namespace ops {


struct OpIndex {
  int broadcast_1_index = INVALID_INDEX;
  int broadcast_2_index = INVALID_INDEX;
};

Status MLUPowerDifference::CreateMLUOp(std::vector<MLUTensor *> &inputs,
                            std::vector<MLUTensor *> &outputs, void *param) {
  TF_PARAMS_CHECK(inputs.size() > 1, "Missing input");
  TF_PARAMS_CHECK(outputs.size() > 0, "Missing output");

  MLUBaseOp *power_difference_op_ptr = nullptr;
  MLUTensor *input1 = inputs.at(0);
  MLUTensor *input2 = inputs.at(1);
  int power_c = *((int*)param);
  MLUTensor *output = outputs.at(0);
  int len = 1;

  //TODO: 数据准备
  lib::MLUTensorUtil input1_util(input1);
  lib::MLUTensorUtil input2_util(input2);
  lib::MLUTensorUtil output_util(output);

  int output_dims = output_util.dims();

  std::vector<int> output_shape(output_dims);
  for (int i = 0; i < output_dims; ++i) {
    output_shape[i] = output_util.dim_size(i);
    len *= output_shape[0];
  }

  MLUTensor* final_input1 = input1;
  MLUTensor* final_input2 = input2;

  struct OpIndex* op_index = (struct OpIndex*) std::malloc(sizeof(struct OpIndex));
  op_index->broadcast_1_index = INVALID_INDEX;
  op_index->broadcast_2_index = INVALID_INDEX;

  int idx = INVALID_INDEX;

  if(!output_util.IsSameSize(input1_util)){
    MLUTensor* tmp1;
    //TODO：创建MLUTensor
    TF_STATUS_CHECK(lib::CreateMLUTensor(&tmp1, MLU_Tensor, input1_util.dtype(), output_shape));

    MLULOG(3) << "CreateBroadcastOp, broadcast1 input: "
              << lib::MLUTensorUtil(input1).DebugString()
              << ", broadcast1 output: "
              << lib::MLUTensorUtil(tmp1).DebugString();

    MLUBaseOp* broadcast_op_ptr_1;
    //TODO: 创建BroadcastOp
    TF_STATUS_CHECK(lib::CreateBroadcastOp(&broadcast_op_ptr_1, input1, tmp1));

    base_ops_.push_back(broadcast_op_ptr_1);
    intmd_tensors_.push_back(tmp1);
    final_input1 = tmp1;
    op_index->broadcast_1_index = ++idx;
  }

  if (!output_util.IsSameSize(input2_util)) {
    MLUTensor* tmp2;
    //TODO：创建MLUTensor
    TF_STATUS_CHECK(lib::CreateMLUTensor(&tmp2, MLU_TENSOR, input2_util.dtype(), output_shape));

    MLULOG(3) << "CreateBroadcastOp, broadcast2 input: "
              << lib::MLUTensorUtil(input2).DebugString()
              << ", broadcast2 output: "
              << lib::MLUTensorUtil(tmp2).DebugString();

    MLUBaseOp* broadcast_op_ptr_2;
    //TODO: 创建BroadcastOp
    TF_STATUS_CHECK(lib::CreateBroadcastOp(&broadcast_op_ptr_2, input2, tmp2));

    base_ops_.push_back(broadcast_op_ptr_2);
    intmd_tensors_.push_back(tmp2);
    final_input2 = tmp2;
    op_index->broadcast_2_index = ++idx;
  }

  MLULOG(3) << "CreatePowerDifferenceOp, input1: "
            << lib::MLUTensorUtil(final_input1).DebugString()
            << ", input2: " << lib::MLUTensorUtil(final_input2).DebugString()
            << ", output: " << lib::MLUTensorUtil(output).DebugString();

  //TODO：调用MLULib层实现好的CreatePowerDifferenceOp
  //code_chap_5_1_student/src/tf-implementation/tf-add-power-diff/mlu_lib_ops.h
  /*params {
    MLUBaseOp** op,
    MLUTensor* input1,
    MLUTensor* input2,
    int input3,
    MLUTensor* output,
    int len}*/
  TF_STATUS_CHECK(lib::CreatePowerDifferenceOp(&power_difference_op_ptr, final_input1, final_input2, power_c, output, len));

  base_ops_.push_back(power_difference_op_ptr);
  extra_ = static_cast <void*>(op_index);
  return Status::OK();
}

Status MLUPowerDifference::Compute(const std::vector<void *> &inputs,
                        const std::vector<void *> &outputs, cnrtQueue_t queue) {
  
  void* input1 = inputs.at(0);
  void* input2 = inputs.at(1);
  void* output = outputs.at(0);
  void* broadcast_1_addr;
  void* broadcast_2_addr;

  struct OpIndex* op_index = static_cast<struct OpIndex*>(extra_);

  if(op_index->broadcast_1_index != INVALID_INDEX){
    MLUBaseOp* broadcast_op_ptr_1 = base_ops_.at(op_index->broadcast_1_index);
    size_t broadcast_1_size;
    cnmlGetTensorSize_V2(intmd_tensors_.at(op_index->broadcast_1_index), &broadcast_1_size);
    cnrtMalloc(&broadcast_1_addr, broadcast_1_size);

    lib::ComputeBroadcastOp(broadcast_op_ptr_1, queue, input1, broadcast_1_addr);
  }
  else{
    broadcast_1_addr = input1;
  }

   if(op_index->broadcast_2_index != INVALID_INDEX){
    MLUBaseOp* broadcast_op_ptr_2 = base_ops_.at(op_index->broadcast_2_index);
    size_t broadcast_2_size;
    cnmlGetTensorSize_V2(intmd_tensors_.at(op_index->broadcast_2_index), &broadcast_2_size);
    cnrtMalloc(&broadcast_2_addr, broadcast_2_size);

    lib::ComputeBroadcastOp(broadcast_op_ptr_2, queue, input2, broadcast_2_addr);
  }
  else{
    broadcast_2_addr = input2;
  }

  MLUBaseOp* power_difference_op = base_ops_.at(base_ops_.size() - 1);

  /*
      ComputePowerDifferenceOp
      params {
              MLUBaseOp* op,
              MLUCnrtQueue* queue,
              void* input1,
              void* input2,
              void* output
      }
  */
  TF_STATUS_CHECK(lib::ComputePowerDifferenceOp(power_difference_op, queue, broadcast_1_addr, broadcast_2_addr, output));
  TF_CNRT_CHECK(cnrtSyncQueue(queue));

  if(op_index->broadcast_1_index != INVALID_INDEX){
    cnrtFree(broadcast_1_addr);
  }
  if(op_index->broadcast_2_index != INVALID_INDEX){
    cnrtFree(broadcast_2_addr);
  }


  
  return Status::OK();
}

}  // namespace ops
}  // namespace mlu
}  // namespace stream_executor

#endif  // CAMBRICON_MLU
