/**
 * Host program for GPT2 inference
*/

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }


#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include "train_gpt2.h"
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

// int main(int argc, char* argv[]) {
//     // TARGET_DEVICE macro needs to be passed from gcc command line
//     if (argc != 2) {
//         std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
//         return EXIT_FAILURE;
//     }

//     std::string xclbinFilename = argv[1];

//     /*******************************
//     * Open and load the bitstream
//     ********************************/
//     std::cout << "INFO: Reading " << xclbinFilename << std::endl;
//     FILE* fp;
//     if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
//         printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
//         exit(EXIT_FAILURE);
//     }
//     // Load xclbin
//     std::cout << "Loading: '" << xclbinFilename << "'\n";
//     std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
//     bin_file.seekg(0, bin_file.end);
//     unsigned nb = bin_file.tellg();
//     bin_file.seekg(0, bin_file.beg);
//     char* buf = new char[nb];
//     bin_file.read(buf, nb);

//     // Creating Program from Binary File
//     cl::Program::Binaries bins;
//     bins.push_back({buf, nb});
//     bool valid_device = false;

//     std::vector<cl::Device> devices;
//     cl_int err;
//     cl::Context context;
//     cl::CommandQueue q;
//     cl::Kernel krnl_GPT2;
//     std::vector<cl::Platform> platforms;
//     bool found_device = false;

//     // traversing all Platforms To find Xilinx Platform and targeted
//     // Device in Xilinx Platform.
//     // Note: if you know the platform name, you can directly look for that
//     cl::Platform::get(&platforms);
//     for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
//         cl::Platform platform = platforms[i];
//         std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
//         if (platformName == "Xilinx") {
//             devices.clear();
//             platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
//             if (devices.size()) {
//                 found_device = true;
//                 break;
//             }
//         }
//     }
//     if (found_device == false) {
//         std::cout << "Error: Unable to find Target Device " << std::endl;
//         return EXIT_FAILURE;
//     }

//     // Use the first available device

//     auto device = devices[0];
//     // Creating Context and Command Queue for selected Device
//     OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
//     OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
//     std::cout << "Trying to program device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

//     // Program the device
//     cl::Program program(context, {device}, bins, nullptr, &err);
//     if (err != CL_SUCCESS) {
//         std::cout << "Failed to program device with xclbin file!\n";
//     } else {
//         std::cout << "Device: program successful!\n";
//         OCL_CHECK(err, krnl_GPT2 = cl::Kernel(program, "gpt2_forward", &err));
//         valid_device = true;
//     }
    
//     if (!valid_device) {
//         std::cout << "Failed to program any device found, exit!\n";
//         exit(EXIT_FAILURE);
//     }

//     /********************************************
//     * Prepare data
//     **********************************************/
//     GPT2 model;
//     float *model_params_memory = NULL, *model_acts_memory;
//     ParameterTensors model_params;
//     ActivationTensors model_acts;

//     printf("Size of struct: %lu bytes\n", sizeof(model));

//     // load additional information that we will use for debugging and error checking
//     FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
//     if (state_file == NULL) { printf("Error opening state file\n"); return 1; }
//     int state_header[256];
//     freadCheck(state_header, sizeof(int), 256, state_file);
//     if (state_header[0] != 20240327) { printf("Bad magic state file\n"); return 1; }
//     if (state_header[1] != 2) {
//         printf("Bad version in state file\n");
//         printf("---> HINT: try to re-run `python train_gpt2.py`\n");
//         return 1;
//     }
//     int B = state_header[2]; // batch size, e.g. 4
//     int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
//     printf("[State]\n");
//     printf("batch_size: %d\n", B);
//     printf("seq_len: %d\n", T);

//     gpt2_build_from_checkpoint(&model, &model_params, &model_params_memory, &model_acts, &model_acts_memory, B, T, "gpt2_124M.bin");

//     // ensure the model was initialized or error out
//     if (model_params_memory == NULL) {
//         printf("%p\n", model_params_memory);
//         printf("Error: model was not initialized properly.\n");
//         exit(1);
//     }

//     int C = model.config.channels;
//     int V = model.config.vocab_size;
//     int Vp = model.config.padded_vocab_size;
//     int maxT = model.config.max_seq_len;
//     int L = model.config.num_layers;

//     // inputs and expected outputs, only used for error checking
//     int* x = (int*) malloc(B * T * sizeof(int));
//     int* y = (int*) malloc(B * T * sizeof(int));
//     float* expected_logits = (float*) malloc(B * T * V * sizeof(float));
//     float* expected_loss = (float*) malloc(1 * sizeof(float));

//     // read reference information from Python
//     freadCheck(x, sizeof(int), B*T, state_file);
//     freadCheck(y, sizeof(int), B*T, state_file);
//     freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
//     freadCheck(expected_loss, sizeof(float), 1, state_file);
//     fcloseCheck(state_file);

//     /********************************************
//     * Allocate buffers on device and prepare data
//     **********************************************/
//     printf("%p\n, &model");

//     /*
//     OCL_CHECK(err, cl::Buffer buffer_model(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 604 * 4, &model, &err));
//     OCL_CHECK(err, cl::Buffer buffer_wte(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 38597376 * 4, model_params.wte, &err));
//     OCL_CHECK(err, cl::Buffer buffer_wpe(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 786432 * 4, model_params.wpe, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1w(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, model_params.ln1w, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1b(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, model_params.ln1b, &err));
//     OCL_CHECK(err, cl::Buffer buffer_qkvw(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 21233664 * 4, model_params.qkvw, &err));
//     OCL_CHECK(err, cl::Buffer buffer_qkvb(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 27648 * 4, model_params.qkvb, &err));
//     OCL_CHECK(err, cl::Buffer buffer_attprojw(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 7077888 * 4, model_params.attprojw, &err));
//     OCL_CHECK(err, cl::Buffer buffer_attprojb(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, model_params.attprojb, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2w(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, model_params.ln2w, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2b(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, model_params.ln2b, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcw(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 28311552 * 4, model_params.fcw, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcb(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 36864 * 4, model_params.fcb, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcprojw(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 28311552 * 4, model_params.fcprojw, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcprojb(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, model_params.fcprojb, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnfw(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 768 * 4, model_params.lnfw, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnfb(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 768 * 4, model_params.lnfb, &err));
//     OCL_CHECK(err, cl::Buffer buffer_encoded(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 196608 * 4, model_acts.encoded, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.ln1, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1_mean(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 3072 * 4, model_acts.ln1_mean, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1_rstd(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 3072 * 4, model_acts.ln1_rstd, &err));
//     OCL_CHECK(err, cl::Buffer buffer_qkv(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 7077888 * 4, model_acts.qkv, &err));
//     OCL_CHECK(err, cl::Buffer buffer_atty(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.atty, &err));
//     OCL_CHECK(err, cl::Buffer buffer_preatt(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.preatt, &err));
//     OCL_CHECK(err, cl::Buffer buffer_att(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.att, &err));
//     OCL_CHECK(err, cl::Buffer buffer_attproj(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.attproj, &err));
//     OCL_CHECK(err, cl::Buffer buffer_residual2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.residual2, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.ln2, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2_mean(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 3072 * 4, model_acts.ln2_mean, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2_rstd(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 3072 * 4, model_acts.ln2_rstd, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fch(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 9437184 * 4, model_acts.fch, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fch_gelu(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 9437184 * 4, model_acts.fch_gelu, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcproj(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.fcproj, &err));
//     OCL_CHECK(err, cl::Buffer buffer_residual3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, model_acts.residual3, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnf(context, CL_MEM_READ_WRITE, 196608 * 4, model_acts.lnf, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnf_mean(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, model_acts.lnf_mean, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnf_rstd(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, model_acts.lnf_rstd, &err));
//     OCL_CHECK(err, cl::Buffer buffer_logits(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 12865792 * 4, model_acts.logits, &err));
//     OCL_CHECK(err, cl::Buffer buffer_probs(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 12865792 * 4, model_acts.probs, &err));
//     OCL_CHECK(err, cl::Buffer buffer_losses(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, model_acts.losses, &err));
//     OCL_CHECK(err, cl::Buffer buffer_inputs(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, x, &err));
//     OCL_CHECK(err, cl::Buffer buffer_targets(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, y, &err));
//     */
    
//     printf("good\n");
//     // These commands will allocate memory on the Device. The cl::Buffer objects can
//     // be used to reference the memory locations on the device.
//     OCL_CHECK(err, cl::Buffer buffer_model(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 604 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_wte(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 38597376 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_wpe(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 786432 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1w(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1b(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_qkvw(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 21233664 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_qkvb(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 27648 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_attprojw(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 7077888 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_attprojb(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2w(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2b(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcw(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 28311552 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcb(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 36864 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcprojw(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 28311552 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcprojb(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 9216 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnfw(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 768 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnfb(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 768 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_encoded(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 196608 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1_mean(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 3072 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln1_rstd(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 3072 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_qkv(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 7077888 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_atty(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_preatt(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_att(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_attproj(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_residual2(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2_mean(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 3072 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_ln2_rstd(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 3072 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fch(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 9437184 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fch_gelu(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 9437184 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_fcproj(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_residual3(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 2359296 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnf(context, CL_MEM_READ_WRITE, 196608 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnf_mean(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_lnf_rstd(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_logits(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 12865792 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_probs(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 12865792 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_losses(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_inputs(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, NULL, &err));
//     OCL_CHECK(err, cl::Buffer buffer_targets(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, 256 * 4, NULL, &err));
    
//     // set the kernel Arguments
//     int narg = 0;
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_model));
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_wte)); // (V)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_wpe)); // (maxT)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln1w)); // (L)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln1b)); // (L)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_qkvw)); // (L)) 3*C)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_qkvb)); // (L)) 3*C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_attprojw)); // (L)) C)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_attprojb)); // (L)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln2w)); // (L)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln2b)); // (L)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_fcw)); // (L)) 4*C)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_fcb)); // (L)) 4*C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_fcprojw)); // (L)) C)) 4*C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_fcprojb)); // (L)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_lnfw)); // (C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_lnfb)); // (C)    
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_encoded)); // (B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln1)); // (L)) B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln1_mean)); // (L)) B)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln1_rstd)); // (L)) B)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_qkv)); // (L)) B)) T)) 3*C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_atty)); // (L)) B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_preatt)); // (L)) B)) NH)) T)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_att)); // (L)) B)) NH)) T)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_attproj)); // (L)) B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_residual2)); // (L)) B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln2)); // (L)) B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln2_mean)); // (L)) B)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_ln2_rstd)); // (L)) B)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_fch)); // (L)) B)) T)) 4*C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_fch_gelu)); // (L)) B)) T)) 4*C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_fcproj)); // (L)) B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_residual3)); // (L)) B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_lnf)); // (B)) T)) C)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_lnf_mean)); // (B)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_lnf_rstd)); // (B)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_logits)); // (B)) T)) V)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_probs)); // (B)) T)) V)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_losses)); // (B)) T)
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_inputs));
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, buffer_targets));
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, B));
//     OCL_CHECK(err, err = krnl_GPT2.setArg(narg++, T));

//     printf("mem allocation successful\n");

//     /*
//     // We then need to map our OpenCL buffers to get the pointers
//     int* ptr_x;
//     int* ptr_y;
//     int* ptr_z; //the result
//     OCL_CHECK(err,
//               ptr_x = (int*)q.enqueueMapBuffer(buffer_x, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes, NULL, NULL, &err));
//     OCL_CHECK(err,
//               ptr_y = (int*)q.enqueueMapBuffer(buffer_y, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes, NULL, NULL, &err));
//     OCL_CHECK(err, ptr_z = (int*)q.enqueueMapBuffer(buffer_z, CL_TRUE, CL_MAP_READ, 0, size_in_bytes, NULL,
//                                                          NULL, &err));
//     int test;
    
//     // Initialize the vectors used in the test
//     for (int i = 0; i < vector_length; i++) {
//         ptr_x[i] = rand() % 1024;
//         ptr_y[i] = rand() % 1024;
//     }
    
//     // Data will be migrated to kernel space
//     OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_x, buffer_y}, 0)); // 0 means from host
//     std::cin >> test;
//     OCL_CHECK(err, q.finish());

//     std::cout << "Copied data to device "<<std::endl;
//     // Run the kernel 
//     // finish previous commands

//     const unsigned long int start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

//     // Launch the Kernel
//     OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));

//     // wait for its execution
//     OCL_CHECK(err, q.finish());
//     const unsigned long int end_time_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

//     std::cout << "Kernel execution completed in (usecs): " << end_time_us - start_time_us << std::endl;


//     // The result of the previous kernel execution will need to be retrieved in
//     // order to view the results. This call will transfer the data from FPGA to
//     // source_results vector
//     OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_z}, CL_MIGRATE_MEM_OBJECT_HOST));

//     OCL_CHECK(err, q.finish());


//     // Verify the results
//     int match = 0;
//     for (int i = 0; i < vector_length; i++) {
//         int host_result = ptr_x[i] + ptr_y[i];
//         if (ptr_z[i] != host_result) {
//             printf(error_message.c_str(), i, host_result, ptr_z[i]);
//             match = 1;
//             break;
//         }
//     }

//     // Cleanup
//     OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_x, ptr_x));
//     OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_y, ptr_y));
//     OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_z, ptr_z));
//     OCL_CHECK(err, err = q.finish());

//     std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
//     return (match ? EXIT_FAILURE : EXIT_SUCCESS);
//     */

//     free(x);
//     free(y);
//     free(expected_logits);
//     free(expected_loss);
//     gpt2_free(model_params_memory, model_acts_memory);
// }

int main(int argc, char** argv) {
    // TARGET_DEVICE macro needs to be passed from gcc command line
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        return EXIT_FAILURE;
    }

    const char* xclbin_filename = argv[1];

    // 1. Initialize device and load FPGA binary
    xrt::device device(0); // Use first available device
    xrt::uuid xclbin_uuid = device.load_xclbin(xclbin_filename);

    // 2. Create kernel and command queue
    std::string kernel_name = "gpt2_forward";
    xrt::kernel kernel(device, xclbin_uuid, kernel_name);
    xrt::run run(kernel);

    // 3.
    /********************************************
    * Prepare data
    **********************************************/
    GPT2 model;
    float *model_params_memory = NULL, *model_acts_memory;
    ParameterTensors model_params;
    ActivationTensors model_acts;

    printf("Size of struct: %lu bytes\n", sizeof(model));

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    if (state_file == NULL) { printf("Error opening state file\n"); return 1; }
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { printf("Bad magic state file\n"); return 1; }
    if (state_header[1] != 2) {
        printf("Bad version in state file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        return 1;
    }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    gpt2_build_from_checkpoint(&model, &model_params, &model_params_memory, &model_acts, &model_acts_memory, B, T, "gpt2_124M.bin");

    // ensure the model was initialized or error out
    if (model_params_memory == NULL) {
        printf("%p\n", model_params_memory);
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    int C = model.config.channels;
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int maxT = model.config.max_seq_len;
    int L = model.config.num_layers;

    // inputs and expected outputs, only used for error checking
    int* x = (int*) malloc(B * T * sizeof(int));
    int* y = (int*) malloc(B * T * sizeof(int));
    float* expected_logits = (float*) malloc(B * T * V * sizeof(float));
    float* expected_loss = (float*) malloc(1 * sizeof(float));

    // read reference information from Python
    freadCheck(x, sizeof(int), B*T, state_file);
    freadCheck(y, sizeof(int), B*T, state_file);
    freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
    freadCheck(expected_loss, sizeof(float), 1, state_file);
    fcloseCheck(state_file);

    /********************************************
    * Allocate buffers on device and prepare data
    **********************************************/
    printf("%p\n%p\n%p\n%p\n%p\n", &model, model_params.wte, model_params.wpe, model_params.ln1w, model_params.ln1b);

    // 4. Create device buffer (aligned to 4096 bytes for best performance)
    xrt::bo buffer_model(device, 604 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(0));
    xrt::bo buffer_wte(device, 38597376 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(1));
    xrt::bo buffer_wpe(device, 786432 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(2));
    xrt::bo buffer_ln1w(device, 9216 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(3));
    xrt::bo buffer_ln1b(device, 9216 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(4));
    xrt::bo buffer_qkvw(device, 21233664 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(5));
    xrt::bo buffer_qkvb(device, 27648 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(6));
    xrt::bo buffer_attprojw(device, 7077888 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(7));
    xrt::bo buffer_attprojb(device, 9216 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(8));
    xrt::bo buffer_ln2w(device, 9216 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(9));
    xrt::bo buffer_ln2b(device, 9216 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(10));
    xrt::bo buffer_fcw(device, 28311552 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(11));
    xrt::bo buffer_fcb(device, 36864 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(12));
    xrt::bo buffer_fcprojw(device, 28311552 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(13));
    xrt::bo buffer_fcprojb(device, 9216 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(14));
    xrt::bo buffer_lnfw(device, 768 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(15));
    xrt::bo buffer_lnfb(device, 768 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(16));
    xrt::bo buffer_encoded(device, 196608 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(17));
    xrt::bo buffer_ln1(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(18));
    xrt::bo buffer_ln1_mean(device, 3072 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(19));
    xrt::bo buffer_ln1_rstd(device, 3072 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(20));
    xrt::bo buffer_qkv(device, 7077888 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(21));
    xrt::bo buffer_atty(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(22));
    xrt::bo buffer_preatt(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(23));
    xrt::bo buffer_att(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(24));
    xrt::bo buffer_attproj(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(25));
    xrt::bo buffer_residual2(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(26));
    xrt::bo buffer_ln2(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(27));
    xrt::bo buffer_ln2_mean(device, 3072 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(28));
    xrt::bo buffer_ln2_rstd(device, 3072 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(29));
    xrt::bo buffer_fch(device, 9437184 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(30));
    xrt::bo buffer_fch_gelu(device, 9437184 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(31));
    xrt::bo buffer_fcproj(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(32));
    xrt::bo buffer_residual3(device, 2359296 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(33));
    xrt::bo buffer_lnf(device, 196608 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(34));
    xrt::bo buffer_lnf_mean(device, 256 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(35));
    xrt::bo buffer_lnf_rstd(device, 256 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(36));
    xrt::bo buffer_logits(device, 12865792 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(37));
    xrt::bo buffer_probs(device, 12865792 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(38));
    xrt::bo buffer_losses(device, 256 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(39));
    xrt::bo buffer_inputs(device, 256 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(40));
    xrt::bo buffer_targets(device, 256 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(41));

    // 5. Transfer data to device
    buffer_model.write(&model);
    buffer_model.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_wte.write(model_params.wte);
    buffer_wte.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_wpe.write(model_params.wpe);
    buffer_wpe.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln1w.write(model_params.ln1w);
    buffer_ln1w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln1b.write(model_params.ln1b);
    buffer_ln1b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_qkvw.write(model_params.qkvw);
    buffer_qkvw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_qkvb.write(model_params.qkvb);
    buffer_qkvb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_attprojw.write(model_params.attprojw);
    buffer_attprojw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_attprojb.write(model_params.attprojb);
    buffer_attprojb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln2w.write(model_params.ln2w);
    buffer_ln2w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln2b.write(model_params.ln2b);
    buffer_ln2b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fcw.write(model_params.fcw);
    buffer_fcw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fcb.write(model_params.fcb);
    buffer_fcb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fcprojw.write(model_params.fcprojw);
    buffer_fcprojw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fcprojb.write(model_params.fcprojb);
    buffer_fcprojb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_lnfw.write(model_params.lnfw);
    buffer_lnfw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_lnfb.write(model_params.lnfb);
    buffer_lnfb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_encoded.write(model_acts.encoded);
    buffer_encoded.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln1.write(model_acts.ln1);
    buffer_ln1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln1_mean.write(model_acts.ln1_mean);
    buffer_ln1_mean.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln1_rstd.write(model_acts.ln1_rstd);
    buffer_ln1_rstd.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_qkv.write(model_acts.qkv);
    buffer_qkv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_atty.write(model_acts.atty);
    buffer_atty.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_preatt.write(model_acts.preatt);
    buffer_preatt.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_att.write(model_acts.att);
    buffer_att.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_attproj.write(model_acts.attproj);
    buffer_attproj.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_residual2.write(model_acts.residual2);
    buffer_residual2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln2.write(model_acts.ln2);
    buffer_ln2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln2_mean.write(model_acts.ln2_mean);
    buffer_ln2_mean.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_ln2_rstd.write(model_acts.ln2_rstd);
    buffer_ln2_rstd.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fch.write(model_acts.fch);
    buffer_fch.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fch_gelu.write(model_acts.fch_gelu);
    buffer_fch_gelu.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fcproj.write(model_acts.fcproj);
    buffer_fcproj.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_residual3.write(model_acts.residual3);
    buffer_residual3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_lnf.write(model_acts.lnf);
    buffer_lnf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_lnf_mean.write(model_acts.lnf_mean);
    buffer_lnf_mean.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_lnf_rstd.write(model_acts.lnf_rstd);
    buffer_lnf_rstd.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_logits.write(model_acts.logits);
    buffer_logits.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_probs.write(model_acts.probs);
    buffer_probs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_losses.write(model_acts.losses);
    buffer_losses.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_inputs.write(x);
    buffer_inputs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_targets.write(y);
    buffer_targets.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    printf("mem allocation successful\n");

    // 6. Set kernel arguments and execute
    int nargs = 0;
    run.set_arg(nargs++, buffer_model);
    run.set_arg(nargs++, buffer_wte);
    run.set_arg(nargs++, buffer_wpe);
    run.set_arg(nargs++, buffer_ln1w);
    run.set_arg(nargs++, buffer_ln1b);
    run.set_arg(nargs++, buffer_qkvw);
    run.set_arg(nargs++, buffer_qkvb);
    run.set_arg(nargs++, buffer_attprojw);
    run.set_arg(nargs++, buffer_attprojb);
    run.set_arg(nargs++, buffer_ln2w);
    run.set_arg(nargs++, buffer_ln2b);
    run.set_arg(nargs++, buffer_fcw);
    run.set_arg(nargs++, buffer_fcb);
    run.set_arg(nargs++, buffer_fcprojw);
    run.set_arg(nargs++, buffer_fcprojb);
    run.set_arg(nargs++, buffer_lnfw);
    run.set_arg(nargs++, buffer_lnfb);
    run.set_arg(nargs++, buffer_encoded);
    run.set_arg(nargs++, buffer_ln1);
    run.set_arg(nargs++, buffer_ln1_mean);
    run.set_arg(nargs++, buffer_ln1_rstd);
    run.set_arg(nargs++, buffer_qkv);
    run.set_arg(nargs++, buffer_atty);
    run.set_arg(nargs++, buffer_preatt);
    run.set_arg(nargs++, buffer_att);
    run.set_arg(nargs++, buffer_attproj);
    run.set_arg(nargs++, buffer_residual2);
    run.set_arg(nargs++, buffer_ln2);
    run.set_arg(nargs++, buffer_ln2_mean);
    run.set_arg(nargs++, buffer_ln2_rstd);
    run.set_arg(nargs++, buffer_fch);
    run.set_arg(nargs++, buffer_fch_gelu);
    run.set_arg(nargs++, buffer_fcproj);
    run.set_arg(nargs++, buffer_residual3);
    run.set_arg(nargs++, buffer_lnf);
    run.set_arg(nargs++, buffer_lnf_mean);
    run.set_arg(nargs++, buffer_lnf_rstd);
    run.set_arg(nargs++, buffer_logits);
    run.set_arg(nargs++, buffer_probs);
    run.set_arg(nargs++, buffer_losses);
    run.set_arg(nargs++, buffer_inputs);
    run.set_arg(nargs++, buffer_targets);
    run.set_arg(nargs++, B);
    run.set_arg(nargs++, T);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    run.start();
    
    // 7. Wait for kernel completion
    run.wait();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    std::cout << "Kernel execution completed successfully!\n Time elapsed in s: " << time_elapsed_s << std::endl;

    std::cout << "Synchronize the output buffer data from the device" << std::endl;
    buffer_model.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffer_logits.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::cout << "Read the output data\n";
    buffer_model.read(&model);
    buffer_logits.read( model_acts.logits);

    // error checking at step 0 for reference activations/gradients
    // at this point, target should be equal to expected_logits, let's compare
    int logits_ok = 1;
    float* calculated_logits = model_acts.logits;
    float max_diff = 0.0f;
    for (int bt = 0; bt < B*T; bt++) {
        for (int v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
            int i = bt * Vp + v; // linearized index, using Vp
            if (i < 10) {
                printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
            }
            float diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
            max_diff = fmaxf(max_diff, diff);
            if (diff >= 1e-2f) {
                printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                printf("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                logits_ok = 0;
                bt = B*T; // to break out of both loops
                break;
            }
        }
    }
    if(!logits_ok) { printf("NOT "); }
    printf("OK (LOGITS), max_diff = %e\n", max_diff);

    // compare the achieved loss
    if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
        printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
    } else {
        printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
    }
    
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    gpt2_free(model_params_memory, model_acts_memory);

}