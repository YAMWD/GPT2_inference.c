#define TESTING
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
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    int C = model.config.channels;
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int maxT = model.config.max_seq_len;
    int L = model.config.num_layers;

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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    gpt2_forward(&model, x, y, B, T);

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
    if(!logits_ok) { printf("NOT "); }
    printf("OK (LOGITS), max_diff = %e\n", max_diff);

    // compare the achieved loss
    if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
        printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
    } else {
        printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
    }

    // free everything
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    gpt2_free(&model);
    return 0;
}
