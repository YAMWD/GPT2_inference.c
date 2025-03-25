#include "train_gpt2.h"

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char* label) {
    int print_upto = 5;
    int ok = 1;
    float maxdiff = 0.0f;
    float tol = 2e-2f;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        float diff = fabsf(a[i] - b[i]);

        if (diff > tol)
        {
            printf("%d\n", i);
            printf("%f %f\n", a[i], b[i]);
            break;
        }

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
    return ok;
}

int main(int argc, char *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    float *model_params_memory = NULL, *model_acts_memory;
    ParameterTensors model_params;
    ActivationTensors model_acts;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    if (state_file == NULL) { printf("Error opening state file\n"); return 1; }
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    fcloseCheck(state_file);
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
    int NH = model.config.num_heads;

    // inputs and expected outputs, only used for error checking
    float *inputs = (float*) malloc(B * T * C * sizeof(float));
    float *qkv_outputs = (float*) malloc(B * T * 3 * C * sizeof(float));
    float *c_attn_outputs = (float*) malloc(B * T * C * sizeof(float));
    float *c_proj_outputs = (float*) malloc(B * T * C * sizeof(float));
    float *expected_outputs = (float*) malloc(B * T * C * sizeof(float));

    FILE *attn_state_file = fopen("gpt2_124M_block_0_attn_debug_state.bin", "rb");
    // read reference information from Python
    freadCheck(inputs, sizeof(float), B * T * C, attn_state_file);
    freadCheck(expected_outputs, sizeof(float), B * T * C, attn_state_file);
    fcloseCheck(attn_state_file);

    for (int i = 0; i < 10; ++i)
        printf("%f\n", inputs[i]);
    // fflush(stdout);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    attn_block_forward(
        c_proj_outputs, c_attn_outputs, qkv_outputs,
        inputs,
        model_params.qkvw, 
        model_params.qkvb, 
        model_acts.preatt, // (B, NH, T, T)
        model_acts.att, // (B, NH, T, T)
        model_params.attprojw, 
        model_params.attprojb, 
        B, T, C, NH);

    printf("attn done\n");
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("time eplased: %lf\n", time_elapsed_s);

    check_tensor(c_proj_outputs, expected_outputs, B * T * C, "attn_out");

    // free everything
    free(inputs);
    free(qkv_outputs);
    free(c_attn_outputs);
    free(c_proj_outputs);
    free(expected_outputs);
    gpt2_free(model_params_memory, model_acts_memory);

    return 0;
}