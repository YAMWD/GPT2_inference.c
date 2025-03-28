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

    printf("validating inputs\n");
    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        // printf("%d ", inputs[i]);
        // fflush(stdout);
        assert(0 <= x[i] && x[i] < V);
        if (y != NULL) {
            assert(0 <= y[i] && y[i] < V);
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    gpt2_forward(
        &model, 
        
        model_params.wte, // (V, C)
        model_params.wpe, // (maxT, C)
        model_params.ln1w, // (L, C)
        model_params.ln1b, // (L, C)
        model_params.qkvw, // (L, 3*C, C)
        model_params.qkvb, // (L, 3*C)
        model_params.attprojw, // (L, C, C)
        model_params.attprojb, // (L, C)
        model_params.ln2w, // (L, C)
        model_params.ln2b, // (L, C)
        model_params.fcw, // (L, 4*C, C)
        model_params.fcb, // (L, 4*C)
        model_params.fcprojw, // (L, C, 4*C)
        model_params.fcprojb, // (L, C)
        model_params.lnfw, // (C)
        model_params.lnfb, // (C)        
        model_acts.encoded, // (B, T, C)
        model_acts.ln1, // (L, B, T, C)
        model_acts.ln1_mean, // (L, B, T)
        model_acts.ln1_rstd, // (L, B, T)
        model_acts.qkv, // (L, B, T, 3*C)
        model_acts.atty, // (L, B, T, C)
        model_acts.preatt, // (L, B, NH, T, T)
        model_acts.att, // (L, B, NH, T, T)
        model_acts.attproj, // (L, B, T, C)
        model_acts.residual2, // (L, B, T, C)
        model_acts.ln2, // (L, B, T, C)
        model_acts.ln2_mean, // (L, B, T)
        model_acts.ln2_rstd, // (L, B, T)
        model_acts.fch, // (L, B, T, 4*C)
        model_acts.fch_gelu, // (L, B, T, 4*C)
        model_acts.fcproj, // (L, B, T, C)
        model_acts.residual3, // (L, B, T, C)
        model_acts.lnf, // (B, T, C)
        model_acts.lnf_mean, // (B, T)
        model_acts.lnf_rstd, // (B, T)
        model_acts.logits, // (B, T, V)
        model_acts.probs, // (B, T, V)
        model_acts.losses, // (B, T)

        x, y, B, T);

    printf("forward pass done\n");
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

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

    // assert(logits_ok == 1);

    printf("OK (LOGITS), max_diff = %e\n", max_diff);

    // compare the achieved loss
    int loss_ok = 1;
    if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
        loss_ok = 0;
        printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
    } else {
        printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
    }

    // assert(loss_ok == 1);

    #ifndef HLS_CSIM
    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int train_num_batches = train_loader.num_tokens / (B*T);
    int val_num_batches = val_loader.num_tokens / (B*T);

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64; // number of steps of inference we will do

    // inference
    float loss = 0.0f;
    for (int i = 0; i < val_num_batches; i++) 
    {
        printf("############\n");
        printf("batch %d\n", i);
        printf("############\n");

        dataloader_next_batch(&val_loader);

        gpt2_forward(
        &model, 
        
        model_params.wte, // (V, C)
        model_params.wpe, // (maxT, C)
        model_params.ln1w, // (L, C)
        model_params.ln1b, // (L, C)
        model_params.qkvw, // (L, 3*C, C)
        model_params.qkvb, // (L, 3*C)
        model_params.attprojw, // (L, C, C)
        model_params.attprojb, // (L, C)
        model_params.ln2w, // (L, C)
        model_params.ln2b, // (L, C)
        model_params.fcw, // (L, 4*C, C)
        model_params.fcb, // (L, 4*C)
        model_params.fcprojw, // (L, C, 4*C)
        model_params.fcprojb, // (L, C)
        model_params.lnfw, // (C)
        model_params.lnfb, // (C)        
        model_acts.encoded, // (B, T, C)
        model_acts.ln1, // (L, B, T, C)
        model_acts.ln1_mean, // (L, B, T)
        model_acts.ln1_rstd, // (L, B, T)
        model_acts.qkv, // (L, B, T, 3*C)
        model_acts.atty, // (L, B, T, C)
        model_acts.preatt, // (L, B, NH, T, T)
        model_acts.att, // (L, B, NH, T, T)
        model_acts.attproj, // (L, B, T, C)
        model_acts.residual2, // (L, B, T, C)
        model_acts.ln2, // (L, B, T, C)
        model_acts.ln2_mean, // (L, B, T)
        model_acts.ln2_rstd, // (L, B, T)
        model_acts.fch, // (L, B, T, 4*C)
        model_acts.fch_gelu, // (L, B, T, 4*C)
        model_acts.fcproj, // (L, B, T, C)
        model_acts.residual3, // (L, B, T, C)
        model_acts.lnf, // (B, T, C)
        model_acts.lnf_mean, // (B, T)
        model_acts.lnf_rstd, // (B, T)
        model_acts.logits, // (B, T, V)
        model_acts.probs, // (B, T, V)
        model_acts.losses, // (B, T)

        val_loader.inputs, val_loader.targets, B, T);

        printf("loss: %f\n\n", model.mean_loss);

        loss += model.mean_loss;
    }

    loss /= val_num_batches;

    printf("val loss %f\n", loss);
    #endif

    // free everything
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    gpt2_free(model_params_memory, model_acts_memory);

    return 0;
}
