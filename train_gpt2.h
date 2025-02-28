#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    // ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    // float* params_memory;
    size_t num_parameters;
    /*
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    */
    // the activations of the model, and their sizes
    // ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    //float* acts_memory;
    size_t num_activations;
    /*
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    */
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int inputs[4 * 64]; // the input tokens for the current forward pass
    int targets[4 * 64]; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C); 

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C);

void matmul_forward(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC);

void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH);

void gelu_forward(float* out, float* inp, int N);

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp);

void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int Vp);

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
    size_t T);

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config);

float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes);

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T);

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes);

void gpt2_build_from_checkpoint(GPT2 *model, ParameterTensors *model_params, float **model_params_memory, ActivationTensors *model_acts, float **model_acts_memory, int B, int T, const char* checkpoint_path);

void gpt2_free(float *model_params_memory, float *model_acts_memory);
