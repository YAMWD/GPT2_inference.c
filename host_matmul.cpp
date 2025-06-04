/**
 * Host program for GPT2 c_fc
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

// Compute the Frobenius norm of a matrix
float frobenius_norm(float *matrix, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += matrix[i] * matrix[i];
    }
    return sqrtf(sum);
}

int check_tensor(float *a, float *b, float *Diff, int n, const char* label) {
    int print_upto = 5;
    int ok = 1;
    float maxdiff = 0.0f;
    float totdiff = 0.0f;
    float tol = 2e-2f;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        float diff = fabsf(a[i] - b[i]);
        Diff[i] = diff;
        totdiff += diff;

        // keep track of the overall error
        ok = ok && (diff <= tol);
        if (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        if (i < print_upto) {
            if (diff <= tol) {
                if (i < print_upto) { printf("OK "); }
            } else {
                if (i < print_upto) { printf("NOT OK "); }
            }
            printf("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    if (ok) {
        printf("TENSOR OK, maxdiff = %e\n", maxdiff);
    } else {
        printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }

    // Compute Frobenius norms
    float norm_A = frobenius_norm(b, n);
    float norm_diff = frobenius_norm(Diff, n);

    free(Diff);

    const float epsilon = 1e-8f; // to avoid division by zero
    printf("relative diff: %f\n\n", norm_diff / (norm_A + epsilon));

    return ok;
}

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
    std::string kernel_name = "matmul_forward";
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
    float *inputs = (float*) malloc(B * T * C * sizeof(float));
    float *c_fc_outputs = (float*) malloc(B * T * 4 * C * sizeof(float));
    float *expected_outputs = (float*) malloc(B * T * 4 * C * sizeof(float));
    float *diff = (float*) malloc(B * T * 4 * C * sizeof(float));


    FILE *c_fc_state_file = fopen("gpt2_124M_block_0_c_fc_debug_state.bin", "rb");
    // read reference information from Python
    freadCheck(inputs, sizeof(float), B * T * C, c_fc_state_file);
    freadCheck(expected_outputs, sizeof(float), B * T * 4 * C, c_fc_state_file);
    fcloseCheck(c_fc_state_file);

    float halton_sequence_base_2[NUM_WIDTH][SN_UNIT];
    float halton_sequence_base_3[NUM_WIDTH][SN_UNIT];

    init_halton(halton_sequence_base_2, 2);
    init_halton(halton_sequence_base_3, 3);

    /********************************************
    * Allocate buffers on device and prepare data
    **********************************************/

    // 4. Create device buffer (aligned to 4096 bytes for best performance)
    xrt::bo buffer_c_fc_outputs(device, 786432, XRT_BO_FLAGS_NONE, kernel.group_id(0));
    xrt::bo buffer_inp(device, 196608, XRT_BO_FLAGS_NONE, kernel.group_id(1));
    xrt::bo buffer_fcw(device, 28311552 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(2));
    xrt::bo buffer_fcb(device, 36864 * 4, XRT_BO_FLAGS_NONE, kernel.group_id(3));
    xrt::bo buffer_rn_seq_1(device, SN_LEN * 4, XRT_BO_FLAGS_NONE, kernel.group_id(4));
    xrt::bo buffer_rn_seq_2(device, SN_LEN * 4, XRT_BO_FLAGS_NONE, kernel.group_id(5));


    // 5. Transfer data to device
    buffer_c_fc_outputs.write(c_fc_outputs);
    buffer_c_fc_outputs.sync(XCL_BO_SYNC_BO_TO_DEVICE);    
    buffer_inp.write(inputs);
    buffer_inp.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fcw.write(model_params.fcw);
    buffer_fcw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_fcb.write(model_params.fcb);
    buffer_fcb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_rn_seq_1.write(halton_sequence_base_2);
    buffer_rn_seq_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);    
    buffer_rn_seq_2.write(halton_sequence_base_3);
    buffer_rn_seq_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    printf("mem allocation successful\n");

    // 6. Set kernel arguments and execute
    int nargs = 0;
    run.set_arg(nargs++, buffer_c_fc_outputs);
    run.set_arg(nargs++, buffer_inp);
    run.set_arg(nargs++, buffer_fcw);
    run.set_arg(nargs++, buffer_fcb);
    run.set_arg(nargs++, buffer_rn_seq_1);
    run.set_arg(nargs++, buffer_rn_seq_2);
    run.set_arg(nargs++, B);
    run.set_arg(nargs++, T);
    run.set_arg(nargs++, C);
    run.set_arg(nargs++, 4 * C);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    printf("run start with SN_LEN: %d\n", SN_LEN);
    run.start();
    
    // 7. Wait for kernel completion
    run.wait();

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    std::cout << "Kernel execution completed successfully!\n Time elapsed in s: " << time_elapsed_s << std::endl;

    std::cout << "Synchronize the output buffer data from the device" << std::endl;
    buffer_c_fc_outputs.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::cout << "Read the output data\n";
    buffer_c_fc_outputs.read(c_fc_outputs);

    check_tensor(c_fc_outputs, expected_outputs, diff, B * T * 4 * C, "c_fc_out");

    // free everything
    free(inputs);
    free(c_fc_outputs);
    free(expected_outputs);
    gpt2_free(model_params_memory, model_acts_memory);

}