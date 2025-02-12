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

void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T);

