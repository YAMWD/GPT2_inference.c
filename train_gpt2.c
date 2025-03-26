/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include "train_gpt2.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"

    // explicitly state the value of loop vars so HLS can calculate latency 
    for (int b = 0; b < B; b++) {
    #pragma HLS loop_tripcount min=4 max=4 avg=4
        for (int t = 0; t < T; t++) {
        #pragma HLS loop_tripcount min=64 max=64 avg=64
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
            #pragma HLS loop_tripcount min=768 max=768 avg=768
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted

    #ifdef TESTING_LN
    printf("defined LN marco\n");
    #pragma HLS INTERFACE m_axi port = out depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = mean depth = 256 bundle = gmem
    #pragma HLS INTERFACE m_axi port = rstd depth = 256 bundle = gmem
    #pragma HLS INTERFACE m_axi port = inp depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = weight depth = 768 bundle = gmem
    #pragma HLS INTERFACE m_axi port = bias depth = 768 bundle = gmem

    #pragma HLS INTERFACE s_axilite port = B
    #pragma HLS INTERFACE s_axilite port = T
    #pragma HLS INTERFACE s_axilite port = C

    #endif

    // explicitly state the value of loop vars so HLS can calculate latency 
    // B = 4;
    // T = 64;
    // C = 768;

    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
    #pragma HLS loop_tripcount min=4 max=4 avg=4
        for (int t = 0; t < T; t++) {
        #pragma HLS loop_tripcount min=64 max=64 avg=64
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
            #pragma HLS loop_tripcount min=768 max=768 avg=768
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
            #pragma HLS loop_tripcount min=768 max=768 avg=768
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
            #pragma HLS loop_tripcount min=768 max=768 avg=768
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            // mean[b * T + t] = m;
            // rstd[b * T + t] = s;
        }
    }
}

void attn_block_forward(float *out, float *c_attn_out, float *qkv_out, 
                        float *inp, 
                        float *qkvw, float *qkvb,
                        float *preatt, float *att,
                        float *attprojw, float *attprojb,
                        int B, int T, int C, int NH)
{
    #ifdef TESTING_ATTN
    printf("defined ATTN marco\n");

    #pragma HLS INTERFACE m_axi port = out depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = c_attn_out depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = qkv_out depth = 589824 bundle = gmem
    #pragma HLS INTERFACE m_axi port = inp depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = qkvw depth = 1769472 bundle = gmem
    #pragma HLS INTERFACE m_axi port = qkvb depth = 2304 bundle = gmem
    #pragma HLS INTERFACE m_axi port = preatt depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = att depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = attprojw depth = 589824 bundle = gmem
    #pragma HLS INTERFACE m_axi port = attprojb depth = 768 bundle = gmem


    #pragma HLS INTERFACE s_axilite port = B
    #pragma HLS INTERFACE s_axilite port = T
    #pragma HLS INTERFACE s_axilite port = C
    #pragma HLS INTERFACE s_axilite port = NH

    // explicitly state the value of loop vars so HLS can calculate latency 
    B = 4;
    T = 64;
    C = 768;
    NH = 12;

    #endif

    matmul_forward(
        qkv_out, 
        inp, 
        qkvw, 
        qkvb, 
        B, T, C, 3 * C);

    attention_forward(    
        c_attn_out,   
        preatt, // (B, NH, T, T)
        att, // (B, NH, T, T)
        qkv_out,
        B, T, C, NH);

    matmul_forward(
        out, 
        c_attn_out,
        attprojw, 
        attprojb, 
        B, T, C, C);
}

void mlp_block_forward(float *c_proj_outputs, float *c_fc_gelu_outputs, float *c_fc_outputs, 
                        float *inputs, 
                        float *fcw, float *fcb,
                        float *fcprojw, float *fcprojb,
                        int B, int T, int C)
{
    #ifdef TESTING_MLP
    printf("defined MLP marco\n");

    #pragma HLS INTERFACE m_axi port = c_proj_outputs depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = c_fc_gelu_outputs depth = 786432 bundle = gmem
    #pragma HLS INTERFACE m_axi port = c_fc_outputs depth = 786432 bundle = gmem
    #pragma HLS INTERFACE m_axi port = inputs depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcw depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcb depth = 3072 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcprojw depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcprojb depth = 768 bundle = gmem


    #pragma HLS INTERFACE s_axilite port = B
    #pragma HLS INTERFACE s_axilite port = T
    #pragma HLS INTERFACE s_axilite port = C

    // explicitly state the value of loop vars so HLS can calculate latency 
    B = 4;
    T = 64;
    C = 768;

    #endif

    matmul_forward(
        c_fc_outputs,
        inputs,
        fcw,
        fcb,
        B, T, C, 4 * C);

    gelu_forward(
        c_fc_gelu_outputs,
        c_fc_outputs,
        B * T * 4 * C);

    matmul_forward(
        c_proj_outputs,
        c_fc_gelu_outputs,
        fcprojw,
        fcprojb,
        B, T, 4 * C, C);
}

void matmul_forward(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.

    // explicitly state the value of loop vars so HLS can calculate latency 
    // B = 4;
    // T = 64;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
    #pragma HLS loop_tripcount min=4 max=4 avg=4
        for (int t = 0; t < T; t++) {
        #pragma HLS loop_tripcount min=64 max=64 avg=64
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
            #pragma HLS loop_tripcount min=768 max=50304
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                #pragma HLS loop_tripcount min=768 max=3072
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

/*
void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}
*/

void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
    #pragma HLS loop_tripcount min=4 max=4 avg=4
        for (int t = 0; t < T; t++) {
        #pragma HLS loop_tripcount min=64 max=64 avg=64
            for (int h = 0; h < NH; h++) {
            #pragma HLS loop_tripcount min=12 max=12 avg=12
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                #pragma HLS loop_tripcount min=64 max=64 avg=64
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
#pragma float_control(pop)

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // output: probs are (B, T, Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B, T, Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    // #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
    #pragma HLS loop_tripcount min=4 max=4 avg=4
        for (int t = 0; t < T; t++) {
        #pragma HLS loop_tripcount min=64 max=64 avg=64
            printf("processing token %d in batch %d\n", t, b);
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
            #pragma HLS loop_tripcount min=50257 max=50257 avg=50257
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            printf("maxVal calculated\n");
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
            #pragma HLS loop_tripcount min=50257 max=50257 avg=50257
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            printf("prob calculated\n");
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
            #pragma HLS loop_tripcount min=50257 max=50257 avg=50257
                probs_bt[i] /= sum;
            }
            printf("prob normed\n");
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
            #pragma HLS loop_tripcount min=50304 max=50304 avg=50304
                probs_bt[i] = 0.0f;
            }
            printf("forced padded probs to be zero\n");
        }
    }
}

void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int Vp) {
    // output: losses is (B, T) of the individual losses at each position
    // input: probs are (B, T, Vp) of the probabilities
    // input: targets is (B, T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
    #pragma HLS loop_tripcount min=4 max=4 avg=4
        for (int t = 0; t < T; t++) {
        #pragma HLS loop_tripcount min=64 max=64 avg=64
            // loss = -log(probs[target])
            float* probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // num_parameters = 124475904;
    // malloc all parameters all at once
    // float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    void *p;
    posix_memalign(&p, 4096, num_parameters * sizeof(float));
    float *params_memory = (float *)p;
    //float* params_memory = (float*)aligned_alloc(num_parameters, num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }

    return params_memory;
}

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    void *p;
    posix_memalign(&p, 4096, num_activations * sizeof(float));
    float *acts_memory = (float *)p;
    //float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

void gpt2_build_from_checkpoint(GPT2 *model, ParameterTensors *model_params, float **model_params_memory, ActivationTensors *model_acts, float **model_acts_memory, int B, int T, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    *model_params_memory = malloc_and_point_parameters(model_params, model->param_sizes);
    freadCheck(*model_params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // allocate space for all the activations if needed (done here, lazily)
    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;
    // and now allocate the space
    fill_in_activation_sizes(model->act_sizes, model->config, B, T);
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += model->act_sizes[i];
    }
    printf("num_activations: %zu\n", num_activations);
    model->num_activations = num_activations;
    *model_acts_memory = malloc_and_point_activations(model_acts, model->act_sizes);
    // also create memory for caching inputs and targets
    // model->inputs = (int*)mallocCheck(B * T * sizeof(int));
    // model->targets = (int*)mallocCheck(B * T * sizeof(int)); // might be unused if we never have targets but it's small

    // other inits
    // *model_acts_memory = NULL;
    // model->grads_memory = NULL;
    // model->m_memory = NULL;
    // model->v_memory = NULL;
    // model->grads_acts_memory = NULL;
    // model->inputs = NULL;
    // model->targets = NULL;
    // model->batch_size = 0;
    // model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(
    GPT2 *model, 

    float* wte, // (V, C)
    float* wpe, // (maxT, C)
    float* ln1w, // (L, C)
    float* ln1b, // (L, C)
    float* qkvw, // (L, 3*C, C)
    float* qkvb, // (L, 3*C)
    float* attprojw, // (L, C, C)
    float* attprojb, // (L, C)
    float* ln2w, // (L, C)
    float* ln2b, // (L, C)
    float* fcw, // (L, 4*C, C)
    float* fcb, // (L, 4*C)
    float* fcprojw, // (L, C, 4*C)
    float* fcprojb, // (L, C)
    float* lnfw, // (C)
    float* lnfb, // (C)    
    float* encoded, // (B, T, C)
    float* ln1, // (L, B, T, C)
    float* ln1_mean, // (L, B, T)
    float* ln1_rstd, // (L, B, T)
    float* qkv, // (L, B, T, 3*C)
    float* atty, // (L, B, T, C)
    float* preatt, // (L, B, NH, T, T)
    float* att, // (L, B, NH, T, T)
    float* attproj, // (L, B, T, C)
    float* residual2, // (L, B, T, C)
    float* ln2, // (L, B, T, C)
    float* ln2_mean, // (L, B, T)
    float* ln2_rstd, // (L, B, T)
    float* fch, // (L, B, T, 4*C)
    float* fch_gelu, // (L, B, T, 4*C)
    float* fcproj, // (L, B, T, C)
    float* residual3, // (L, B, T, C)
    float* lnf, // (B, T, C)
    float* lnf_mean, // (B, T)
    float* lnf_rstd, // (B, T)
    float* logits, // (B, T, V)
    float* probs, // (B, T, V)
    float* losses, // (B, T)
    
    int *inputs, 
    int *targets, 
    size_t B, 
    size_t T) {
    #pragma HLS INTERFACE m_axi port = model depth = 1 bundle = gmem

    #pragma HLS INTERFACE m_axi port = wte depth = 38597376 bundle = gmem
    #pragma HLS INTERFACE m_axi port = wpe depth = 786432 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln1w depth = 9216 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln1b depth = 9216 bundle = gmem
    #pragma HLS INTERFACE m_axi port = qkvw depth = 21233664 bundle = gmem
    #pragma HLS INTERFACE m_axi port = qkvb depth = 27648 bundle = gmem
    #pragma HLS INTERFACE m_axi port = attprojw depth = 7077888 bundle = gmem
    #pragma HLS INTERFACE m_axi port = attprojb depth = 9216 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln2w depth = 9216 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln2b depth = 9216 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcw depth = 28311552 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcb depth = 36864 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcprojw depth = 28311552 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcprojb depth = 9216 bundle = gmem
    #pragma HLS INTERFACE m_axi port = lnfw depth = 768 bundle = gmem
    #pragma HLS INTERFACE m_axi port = lnfb depth = 768 bundle = gmem
    #pragma HLS INTERFACE m_axi port = encoded depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln1 depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln1_mean depth = 3072 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln1_rstd depth = 3072 bundle = gmem
    #pragma HLS INTERFACE m_axi port = qkv depth = 7077888 bundle = gmem
    #pragma HLS INTERFACE m_axi port = atty depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = preatt depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = att depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = attproj depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = residual2 depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln2 depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln2_mean depth = 3072 bundle = gmem
    #pragma HLS INTERFACE m_axi port = ln2_rstd depth = 3072 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fch depth = 9437184 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fch_gelu depth = 9437184 bundle = gmem
    #pragma HLS INTERFACE m_axi port = fcproj depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = residual3 depth = 2359296 bundle = gmem
    #pragma HLS INTERFACE m_axi port = lnf depth = 196608 bundle = gmem
    #pragma HLS INTERFACE m_axi port = lnf_mean depth = 256 bundle = gmem
    #pragma HLS INTERFACE m_axi port = lnf_rstd depth = 256 bundle = gmem
    #pragma HLS INTERFACE m_axi port = logits depth = 12865792 bundle = gmem
    #pragma HLS INTERFACE m_axi port = probs depth = 12865792 bundle = gmem
    #pragma HLS INTERFACE m_axi port = losses depth = 256 bundle = gmem

    #pragma HLS INTERFACE m_axi port = inputs depth = 256 bundle = gmem
    #pragma HLS INTERFACE m_axi port = targets depth = 256 bundle = gmem
 
    #pragma HLS INTERFACE s_axilite port=B
    #pragma HLS INTERFACE s_axilite port=T
    // #pragma HLS INTERFACE s_axilite port=return

    // targets are optional and could be NULL

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    ParameterTensors model_params;
    model_params.wte = wte; // (V, C)
    model_params.wpe = wpe; // (maxT, C)
    model_params.ln1w = ln1w; // (L, C)
    model_params.ln1b = ln1b; // (L, C)
    model_params.qkvw = qkvw; // (L, 3*C, C)
    model_params.qkvb = qkvb; // (L, 3*C)
    model_params.attprojw = attprojw; // (L, C, C)
    model_params.attprojb = attprojb; // (L, C)
    model_params.ln2w = ln2w; // (L, C)
    model_params.ln2b = ln2b; // (L, C)
    model_params.fcw = fcw; // (L, 4*C, C)
    model_params.fcb = fcb; // (L, 4*C)
    model_params.fcprojw = fcprojw; // (L, C, 4*C)
    model_params.fcprojb = fcprojb; // (L, C)
    model_params.lnfw = lnfw; // (C)
    model_params.lnfb = lnfb; // (C)

    ActivationTensors model_acts;
    model_acts.encoded = encoded; // (B, T, C)
    model_acts.ln1 = ln1; // (L, B, T, C)
    model_acts.ln1_mean = ln1_mean; // (L, B, T)
    model_acts.ln1_rstd = ln1_rstd; // (L, B, T)
    model_acts.qkv = qkv; // (L, B, T, 3*C)
    model_acts.atty = atty; // (L, B, T, C)
    model_acts.preatt = preatt; // (L, B, NH, T, T)
    model_acts.att = att; // (L, B, NH, T, T)
    model_acts.attproj = attproj; // (L, B, T, C)
    model_acts.residual2 = residual2; // (L, B, T, C)
    model_acts.ln2 = ln2; // (L, B, T, C)
    model_acts.ln2_mean = ln2_mean; // (L, B, T)
    model_acts.ln2_rstd = ln2_rstd; // (L, B, T)
    model_acts.fch = fch; // (L, B, T, 4*C)
    model_acts.fch_gelu = fch_gelu; // (L, B, T, 4*C)
    model_acts.fcproj = fcproj; // (L, B, T, C)
    model_acts.residual3 = residual3; // (L, B, T, C)
    model_acts.lnf = lnf; // (B, T, C)
    model_acts.lnf_mean = lnf_mean; // (B, T)
    model_acts.lnf_rstd = lnf_rstd; // (B, T)
    model_acts.logits = logits; // (B, T, V)
    model_acts.probs = probs; // (B, T, V)
    model_acts.losses = losses; // (B, T)

    printf("validating inputs\n");
    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        printf("%d ", inputs[i]);
        // fflush(stdout);
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model_params; // for brevity
    ActivationTensors acts = model_acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    for (int l = 0; l < L; l++) {
    #pragma HLS loop_tripcount min=12 max=12 avg=12
        printf("computing layer %d\n", l);

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        printf("computing LN\n");
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        printf("computing matmul\n");
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        printf("computing attn\n");
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        printf("computing matmul\n");
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        printf("computing res\n");
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        printf("computing LN\n");
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        printf("computing matmul\n");
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        printf("computing gelu\n");
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        printf("computing matmul\n");
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        printf("computing res\n");
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    printf("computing LN\n");
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    printf("computing matmul\n");
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    printf("computing softmax\n");
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);
    printf("softmax done\n");    

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        printf("computing CE loss\n");
        crossentropy_forward(model_acts.losses, model_acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model_acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
    printf("all the computation are done\n");
}

void gpt2_free(float *model_params_memory, float *model_acts_memory) {
    free(model_params_memory);
    // free(model->grads_memory);
    // free(model->m_memory);
    // free(model->v_memory);
    free(model_acts_memory);
    // free(model->grads_acts_memory);
    // free(model->inputs);
    // free(model->targets);
}

// #ifndef TESTING
// // if we are TESTING (see test_gpt2.c), we'll skip the int main below
// // ----------------------------------------------------------------------------
// // sampler

// unsigned int random_u32(uint64_t *state) {
//     // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
//     *state ^= *state >> 12;
//     *state ^= *state << 25;
//     *state ^= *state >> 27;
//     return (*state * 0x2545F4914F6CDD1Dull) >> 32;
// }
// float random_f32(uint64_t *state) { // random float32 in [0,1)
//     return (random_u32(state) >> 8) / 16777216.0f;
// }

// int sample_mult(float* probabilities, int n, float coin) {
//     // sample index from probabilities (they must sum to 1!)
//     // coin is a random number in [0, 1), usually from random_f32()
//     float cdf = 0.0f;
//     for (int i = 0; i < n; i++) {
//         cdf += probabilities[i];
//         if (coin < cdf) {
//             return i;
//         }
//     }
//     return n - 1; // in case of rounding errors
// }

// // ----------------------------------------------------------------------------
// // main training loop
// int main() {

//     // build the GPT-2 model from a checkpoint
//     GPT2 model;
//     gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

//     // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
//     const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
//     const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
//     const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
//     const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
//     const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
//     const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
//     int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
//     int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
//     DataLoader train_loader, val_loader;
//     dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
//     dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
//     printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
//     printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
//     int train_num_batches = 5;
//     int val_num_batches = 5;

//     // build the Tokenizer
//     Tokenizer tokenizer;
//     tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

//     // some memory for generating samples from the model
//     uint64_t rng_state = 1337;
//     int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
//     const int genT = 64; // number of steps of inference we will do

//     // inference
//     float train_loss = 0.0f;
//     for (int i = 0; i < train_num_batches; i++) {
//         dataloader_next_batch(&train_loader);
//         gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
//         train_loss += model.mean_loss;
//     }
//     train_loss /= train_num_batches;
//     printf("train loss %f\n", train_loss);

//     /*
//     // train
//     struct timespec start, end;
//     for (int step = 0; step <= 40; step++) {

//         // once in a while estimate the validation loss
//         if (step % 10 == 0) {
//             float val_loss = 0.0f;
//             dataloader_reset(&val_loader);
//             for (int i = 0; i < val_num_batches; i++) {
//                 dataloader_next_batch(&val_loader);
//                 gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
//                 val_loss += model.mean_loss;
//             }
//             val_loss /= val_num_batches;
//             printf("val loss %f\n", val_loss);
//         }

//         // once in a while do model inference to print generated text
//         if (step > 0 && step % 20 == 0) {
//             // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
//             for(int i = 0; i < B * T; ++i) {
//                 gen_tokens[i] = tokenizer.eot_token;
//             }
//             // now sample from the model autoregressively
//             printf("generating:\n---\n");
//             for (int t = 1; t < genT; t++) {
//                 // note that inference is very wasteful here because for each token
//                 // we re-calculate the forward pass for all of (B,T) positions from scratch
//                 // but the inference here is just for sanity checking anyway
//                 // and we can maybe optimize a bit more later, with careful tests
//                 gpt2_forward(&model, gen_tokens, NULL, B, T);
//                 // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
//                 // we're in principle running B "inference streams" in parallel here
//                 // but only using position 0
//                 // get the Vp-dimensional vector probs[0, t-1, :]
//                 float* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
//                 float coin = random_f32(&rng_state);
//                 // note we're only sampling from the first V elements, ignoring padding
//                 // (the probabilities in the padded region should be zero anyway)
//                 int next_token = sample_mult(probs, model.config.vocab_size, coin);
//                 gen_tokens[t] = next_token;
//                 // print the generated token, either using the Tokenizer or a fallback
//                 if (tokenizer.init_ok) {
//                     const char* token_str = tokenizer_decode(&tokenizer, next_token);
//                     safe_printf(token_str);
//                 } else {
//                     // fall back to printing the token id
//                     printf("%d ", next_token);
//                 }
//                 fflush(stdout);
//             }
//             printf("\n---\n");
//         }

//         // do a training step
//         clock_gettime(CLOCK_MONOTONIC, &start);
//         dataloader_next_batch(&train_loader);
//         gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
//         gpt2_zero_grad(&model);
//         gpt2_backward(&model);
//         gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
//         clock_gettime(CLOCK_MONOTONIC, &end);
//         double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
//         printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
//     }
//     */

//     // free
//     dataloader_free(&train_loader);
//     dataloader_free(&val_loader);
//     tokenizer_free(&tokenizer);
//     gpt2_free(&model);
//     free(gen_tokens);
//     return 0;
// }
// #endif
