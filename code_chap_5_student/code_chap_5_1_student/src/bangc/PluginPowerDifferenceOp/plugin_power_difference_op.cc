/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#include "cnplugin.h"
#include "plugin_power_difference_kernel.h"

typedef uint16_t half;
#if (FLOAT_MODE == 1)
typedef float DType;
#elif (FLOAT_MODE == 0)     // NOLINT
typedef half DType;
#endif

cnmlStatus_t cnmlCreatePluginPowerDifferenceOpParam(
  cnmlPluginPowerDifferenceOpParam_t *param,
  // TODO：添加变量
  half* input1, half* input2, int pow, half* output, int len
) {
  *param = new cnmlPluginPowerDifferenceOpParam();
  // TODO：配置变量
  (*param)->input1 = input1;
  (*param)->input2 = input2;
  (*param)->pow = pow;
  (*param)->output = output;
  (*param)->len = len;

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlDestroyPluginPowerDifferenceOpParam(
  cnmlPluginPowerDifferenceOpParam_t *param
) {
  delete (*param);
  *param = nullptr;

  return CNML_STATUS_SUCCESS;
}

cnmlStatus_t cnmlCreatePluginPowerDifferenceOp(
  cnmlBaseOp_t *op,
  // TODO：添加变量
  //调用方式可在code_chap_5_1_student/src/tf-implementation/tf-add-power-diff/mlu_lib_ops.cc下找到
  cnmlTensor_t *inputs_ptr,
  int pow,
  cnmlTensor_t *outputs_ptr,
  int len
  
) {
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  // TODO：配置变量
  //cnrt相关函数在code_chap_5_1_student/src/offline/include/cnrt.h中
  /*
      input0, input1  
  */
  cnrtKernelParamsBufferMarkInput(params);
  cnrtKernelParamsBufferMarkInput(params);
  //   pow
  cnrtKernelParamsBufferAddParam(params, &pow, sizeof(int));
  //   output
  cnrtKernelParamsBufferMarkOutput(params);
  //   len
  cnrtKernelParamsBufferAddParam(params, &len, sizeof(int));

  //数据转换
  void **interfacePtr = reinterpret_cast<void **>(&PowerDifferenceKernel); 
  /*
      cnml函数相关路径code_chap_5_2_student/neuware/include/cnml.h
  */

  cnmlCreatePluginOp(op,
                     "PowerDifference",
                     interfacePtr,
                     params,
                     inputs_ptr,
                     2,
                     outputs_ptr,
                     1,
                     nullptr,
                     0
                     );
  cnrtDestroyKernelParamsBuffer(params);
  return CNML_STATUS_SUCCESS;
}

/*
    params: {
        op, inputs_ptr, outputs_ptr, queue
    }
*/
cnmlStatus_t cnmlComputePluginPowerDifferenceOpForward(
  cnmlBaseOp_t op,
  // TODO：添加变量
  void** inputs_ptr,
  void** outputs_ptr,
  cnrtQueue_t queue
) {
  // TODO：完成Compute函数
  //cnml函数相关路径code_chap_5_2_student/neuware/include/cnml.h
  cnmlComputePluginOpForward_V4(op,
                                nullptr,
                                inputs_ptr,
                                2,
                                nullptr,
                                outputs_ptr,
                                1,
                                queue,
                                nullptr);
  return CNML_STATUS_SUCCESS;
}

