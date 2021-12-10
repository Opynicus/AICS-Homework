#include "inference.h"
#include "cnrt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stdlib.h"
#include <sys/time.h>
#include <time.h>

namespace StyleTransfer{

typedef unsigned short half;

void cnrtConvertFloatToHalfArray(uint16_t* x, const float* y, int len) {
    for (int i = 0; i < len; i++){
        cnrtConvertFloatToHalf(x+i,y[i]);
    }
}

void cnrtConvertHalfToFloatArray(float* x, const uint16_t* y, int len) {
    for (int i = 0; i < len; i++){
        cnrtConvertHalfToFloat(x+i,y[i]);
    }
}

Inference :: Inference(std::string offline_model){
    offline_model_ = offline_model;
}

void Inference :: run(DataTransfer* DataT){

    // TODO:load model
    cnrtInit(0);
    cnrtModel_t model;
    cnrtLoadModel(&model, offline_model_.c_str());
    // TODO:set current device
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev);
    // TODO:load extract function
    cnrtFunction_t function;
    cnrtCreateFunction(&function);
    cnrtExtractFunction(&function, model, "subnet0");
    int inputNum, outputNum;
    int64_t *inputSizeS, *outputSizeS;
    cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
    cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);
    //printf("input num :%d\n output num : %d\n",inputNum,outputNum);
    //printf("input size: %lld\noutput size : %lld\n",inputSizeS[0],outputSizeS[0]);
    
    //inputNum,outputNum=1
    //inputsize 256 * 256 * 3 * sizeof(half)
    //outputsize 256 * 256 * 3 * sizeof(half)

    // TODO:prepare data on cpu
    float* input_data = (float*)(malloc(256 * 256 * 3 * sizeof(float)));
    float* output_data = (float*)(malloc(256 * 256 * 3 * sizeof(float)));
    DataT->output_data = (float*)(malloc(256 * 256 * 3 * sizeof(float)));

    half* inputCpuPtrS = (half*)malloc(256 * 256 * 3 * sizeof(half));
    half* outputCpuPtrS = (half*)malloc(256 * 256 * 3 * sizeof(half));

    // TODO:allocate I/O data memory on MLU
    // TODO:prepare input buffer
    //cnrtMalloc，对MLU专用，普通的malloc会error
    void *inputMluPtrS, *outputMluPtrS;
    cnrtMalloc(&inputMluPtrS, 256 * 256 * 3 * sizeof(half));
    cnrtMalloc(&outputMluPtrS, 256 * 256 * 3 * sizeof(half));
    for(int i = 0; i < 256 * 256; i++)//{C, H, W} -> {H, W, C}
        for(int j = 0; j < 3; j++)
            input_data[i * 3 + j] = DataT->input_data[256 * 256 * j + i];  
    cnrtConvertFloatToHalfArray(inputCpuPtrS, input_data, 256 * 256 * 3);
    cnrtMemcpy(inputMluPtrS, inputCpuPtrS, 256 * 256 * 3 * sizeof(half), CNRT_MEM_TRANS_DIR_HOST2DEV);

    // TODO:setup runtime ctx
    cnrtRuntimeContext_t ctx;
    cnrtCreateRuntimeContext(&ctx, function, NULL);

    // TODO:bind device
    cnrtSetRuntimeContextDeviceId(ctx, 0);
    cnrtInitRuntimeContext(ctx, NULL);

    // TODO:compute offline
    cnrtQueue_t queue;
    cnrtRuntimeContextCreateQueue(ctx, &queue);

    // invoke
    void* param[2];
    param[0] = inputMluPtrS;
    param[1] = outputMluPtrS;
    cnrtInvokeRuntimeContext(ctx, (void**)param, queue, NULL);

    // sync
    cnrtSyncQueue(queue);

    // copy mlu result to cpu
    cnrtMemcpy(outputCpuPtrS, outputMluPtrS, 256 * 256 * 3 * sizeof(half), CNRT_MEM_TRANS_DIR_DEV2HOST);
    cnrtConvertHalfToFloatArray(output_data, outputCpuPtrS, 256 * 256 * 3);
    for(int i = 0; i < 256 * 256; i++)
        for(int j = 0; j < 3; j++)
            DataT->output_data[256 * 256 * j + i] = output_data[i * 3 + j];

    
    // TODO:free memory spac
    cnrtFree(inputMluPtrS);
    cnrtFree(outputMluPtrS);
    cnrtDestroyQueue(queue);
    free(inputCpuPtrS);
    free(outputCpuPtrS);
    cnrtDestroy();
}

} // namespace StyleTransfer
