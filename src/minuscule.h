#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

#ifndef MINUSCULE_H
#define MINUSCULE_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct {
    uint32_t rows;
    uint32_t cols;
    float* values;
} Matrix;

Matrix mat_new(uint32_t rows, uint32_t cols);
void mat_free(Matrix* m);
uint32_t mat_elem_count(Matrix const* m);

void mat_ensure_shape(Matrix* m, uint32_t rows, uint32_t cols);
bool mat_same_shape(Matrix const* a, Matrix const* b);
void mat_zero(Matrix* m);

float mat_at_const(Matrix const* m, uint32_t row, uint32_t col);
float* mat_at(Matrix* m, uint32_t row, uint32_t col);

void mat_add_into(Matrix const* a, Matrix* b);
void mat_sub_into(Matrix const* a, Matrix* b);
void mat_mulf(Matrix* m, float v);

void mat_copy_into(Matrix const* src, Matrix* dest);
void mat_move_into(Matrix* src, Matrix* dest);

void mat_mul_into(Matrix const* a, Matrix const* b, Matrix* out, bool transpose_a, bool transpose_b, bool zero);

typedef struct {
    Matrix momentum;
    Matrix variance;
} AdamW;

typedef struct {
    Matrix value;
    Matrix gradient;
    AdamW optimizer;
} Parameter;

Parameter parameter_new(uint32_t rows, uint32_t cols);
void parameter_free(Parameter* parameter);
void parameter_step(Parameter* parameter, float lr, float betta_1, float betta_2, float lambda, uint32_t t);

typedef enum {
    LAYER_DENSE,
    LAYER_ACTIVATION,
    LAYER_LAYERNORM,
    LAYER_SOFTMAX,
} LayerType;

typedef struct {
    Parameter weights;
    Parameter biases;
    Matrix cached_activation;
    Matrix temp_output;
    Matrix temp_gradient;
} DenseLayer;

typedef float(*Function)(float x);

typedef struct {
    Function forward;
    Function derivative;
    Matrix cached_activation;
} ActivationLayer;

typedef struct {
    Parameter scale;
    Parameter shift;
    Matrix cached_input;
    Matrix cached_output;
    float cached_mean, cached_std;
} LaynormLayer;

typedef struct {
    float temp;
    Matrix cached_output;
} SoftmaxLayer;

typedef struct {
    LayerType type;
    uint32_t fan_in;
    uint32_t fan_out;

    union {
        DenseLayer       dense;
        ActivationLayer  activation;
        LaynormLayer     layernorm;
        SoftmaxLayer     softmax;
    } data;
} Layer;

typedef enum {
    LOSS_MSE,
    LOSS_MAE,
    LOSS_BINARY_CROSS_ENTROPY,
    LOSS_CATEGORICAL_CROSS_ENTROPY, // can only be used with > 1 output features
} LossFunction;

typedef enum {
    LEARNING_RATE_CONSTANT,
    LEARNING_RATE_EXPONENTIAL,
} LearningRateType;

typedef struct {
    float learning_rate;
} ConstantLearningRate;

typedef struct {
    float lr_begin;
    float lr_end;
    float exponent;
    uint32_t expected_steps;
} ExponentialLearningRate;

typedef struct {
    LearningRateType type;
    union {
        ConstantLearningRate     constant;
        ExponentialLearningRate  exponential;
    } data;
} LearningRate;

LearningRate constant_learning_rate(float lr);
LearningRate exponential_learning_rate(float lr_begin, float lr_end, float exponent, uint32_t expected_steps);

float get_learning_rate(LearningRate learning_rate, uint32_t step);

typedef struct {
    LearningRate learning_rate;
    LossFunction loss_function;
    float betta_1;
    float betta_2;
    float lambda;
    uint32_t mini_batch;
    uint32_t t, optimizer_step;

    uint32_t layer_count;
    Layer* layers;
} MLP;

typedef enum {
    DENSE_INIT_XAVIER,
    DENSE_INIT_KAIMING,
} DenseInitOption;

Layer dense_new(uint32_t fan_in, uint32_t fan_out, DenseInitOption init_option);
void dense_free(DenseLayer* dense);
void dense_forward (DenseLayer* dense, Matrix* inout, bool train);
void dense_backward(DenseLayer* dense, Matrix* inout);
void dense_step    (DenseLayer* dense, float lr, float betta_1, float betta_2, float lambda, uint32_t t);

Layer activation_new(uint32_t fan_in, Function forward, Function derivative);
void activation_free(ActivationLayer* activation);
void activation_forward (ActivationLayer* activation, Matrix* inout, bool train);
void activation_backward(ActivationLayer* activation, Matrix* inout);

Layer layernorm_new(uint32_t fan_in);
void layernorm_free(LaynormLayer* layernorm);
void layernorm_forward (LaynormLayer* layernorm, Matrix* inout, bool train);
void layernorm_backward(LaynormLayer* layernorm, Matrix* inout);
void layernorm_step    (LaynormLayer* layernorm, float lr, float betta_1, float betta_2, float lambda, uint32_t t);

Layer softmax_new(uint32_t fan_in, float temp);
void softmax_free(SoftmaxLayer* softmax);
void softmax_forward (SoftmaxLayer* softmax, Matrix* inout, bool train);
void softmax_backward(SoftmaxLayer* softmax, Matrix* inout);

void layer_forward (Layer* layer, Matrix* inout, bool train);
void layer_backward(Layer* layer, Matrix* inout);
void layer_free    (Layer* layer);
void layer_step    (Layer* layer, float lr, float betta_1, float betta_2, float lambda, uint32_t t);

float relu(float x);
float relu_derivative(float x);
float leaky_relu(float x);
float leaky_relu_derivative(float x);
float sigmoid(float x);
float sigmoid_derivative(float x);

MLP mlp_new(void);
void mlp_free(MLP* mlp);

void mlp_push_layer(MLP* mlp, Layer layer);

void mlp_forward(MLP* mlp, Matrix* inout);
void mlp_train(MLP* mlp, Matrix const* input, Matrix const* desired_output);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MINUSCULE_H

// --- Implementation --
#ifdef MINUSCULE_IMPLEMENTATION
#undef MINUSCULE_IMPLEMENTATION

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

Matrix mat_new(uint32_t rows, uint32_t cols) {
    assert(rows > 0);
    assert(cols > 0);

    return (Matrix){
        .rows = rows,
        .cols = cols,
        .values = (float*)calloc(rows * cols, sizeof(float)),
    };
}

void mat_free(Matrix* m) {
    assert(m);
    assert(m->values);
    free(m->values);
    *m = (Matrix){0};
}

uint32_t mat_elem_count(Matrix const* m) {
    assert(m);
    return m->rows * m->cols;
}

void mat_ensure_shape(Matrix* m, uint32_t rows, uint32_t cols) {
    assert(m);

    if (m->rows == rows && m->cols == cols) return;

    if (m->values) mat_free(m);
    *m = mat_new(rows, cols);
}

bool mat_same_shape(Matrix const* a, Matrix const* b) {
    assert(a);
    assert(b);
    return (a->rows == b->rows && a->cols == b->cols);
}

void mat_zero(Matrix* m) {
    assert(m);
    for (uint32_t i = 0; i < mat_elem_count(m); ++i)
        m->values[i] = 0;
}

static inline uint32_t mat_index(uint32_t row, uint32_t col, uint32_t rows) {
    return row + col * rows;
}

float mat_at_const(Matrix const* m, uint32_t row, uint32_t col) {
    assert(m);
    assert(row < m->rows);
    assert(col < m->cols);
    return m->values[mat_index(row, col, m->rows)];
}

float* mat_at(Matrix* m, uint32_t row, uint32_t col) {
    assert(m);
    assert(row < m->rows);
    assert(col < m->cols);
    return &m->values[mat_index(row, col, m->rows)];
}

void mat_add_into(Matrix const* a, Matrix* b) {
    assert(a);
    assert(b);
    assert(mat_same_shape(a, b));

    for (uint32_t i = 0; i < mat_elem_count(a); ++i)
        b->values[i] += a->values[i];
}

void mat_sub_into(Matrix const* a, Matrix* b) {
    assert(a);
    assert(b);
    assert(mat_same_shape(a, b));

    for (uint32_t i = 0; i < mat_elem_count(a); ++i)
        b->values[i] -= a->values[i];
}

void mat_mulf(Matrix* m, float v) {
    assert(m);

    for (uint32_t i = 0; i < mat_elem_count(m); ++i)
        m->values[i] *= v;
}

void mat_copy_into(Matrix const* src, Matrix* dest) {
    assert(src);
    assert(dest);

    mat_ensure_shape(dest, src->rows, src->cols);

    for (uint32_t i = 0; i < mat_elem_count(src); ++i)
        dest->values[i] = src->values[i];
}

void mat_move_into(Matrix* src, Matrix* dest) {
    mat_free(dest);
    *dest = *src;
    *src = (Matrix){0};
}

static inline void mat_mul_into_nn(Matrix const* a, Matrix const* b, Matrix* out) {
    for (uint32_t i = 0; i < a->rows; ++i) {
        for (uint32_t k = 0; k < b->cols; ++k) {
            for (uint32_t j = 0; j < a->cols; ++j) {
                *mat_at(out, i, k) +=
                    mat_at_const(a, i, j) *
                    mat_at_const(b, j, k);
            }
        }
    }
}

static inline void mat_mul_into_nt(Matrix const* a, Matrix const* b, Matrix* out) {
    (void)a;
    (void)b;
    (void)out;
    abort();
}

static inline void mat_mul_into_tn(Matrix const* a, Matrix const* b, Matrix* out) {
    for (uint32_t i = 0; i < a->cols; ++i) {
        for (uint32_t k = 0; k < b->cols; ++k) {
            for (uint32_t j = 0; j < a->rows; ++j) {
                *mat_at(out, i, k) +=
                    mat_at_const(a, j, i) *
                    mat_at_const(b, j, k);
            }
        }
    }
}

static inline void mat_mul_into_tt(Matrix const* a, Matrix const* b, Matrix* out) {
    (void)a;
    (void)b;
    (void)out;
    abort();
}

void mat_mul_into(Matrix const* a, Matrix const* b, Matrix* out, bool transpose_a, bool transpose_b, bool zero) {
    assert(a);
    assert(b);
    assert(out);

    uint32_t a_rows = (transpose_a ? a->cols : a->rows);
    uint32_t a_cols = (transpose_a ? a->rows : a->cols);
    uint32_t b_rows = (transpose_b ? b->cols : b->rows);
    uint32_t b_cols = (transpose_b ? b->rows : b->cols);

    assert(a_cols == b_rows);
    
    mat_ensure_shape(out, a_rows, b_cols);

    if (zero) mat_zero(out);

    uint8_t flags = (uint8_t)((transpose_a << 1) | (transpose_b << 0));

    switch (flags) {
        case (0 << 1) | (0 << 0): mat_mul_into_nn(a, b, out); break;
        case (1 << 1) | (0 << 0): mat_mul_into_tn(a, b, out); break;
        case (0 << 1) | (1 << 0): mat_mul_into_nt(a, b, out); break;
        case (1 << 1) | (1 << 0): mat_mul_into_tt(a, b, out); break;
        default: abort();
    }
}

Parameter parameter_new(uint32_t rows, uint32_t cols) {
    return (Parameter){
        .value     = mat_new(rows, cols),
        .gradient  = mat_new(rows, cols),
        .optimizer = (AdamW){
            .momentum = mat_new(rows, cols),
            .variance = mat_new(rows, cols),
        },
    };
}

void parameter_free(Parameter* parameter) {
    assert(parameter);
    mat_free(&parameter->value);
    mat_free(&parameter->gradient);
    mat_free(&parameter->optimizer.momentum);
    mat_free(&parameter->optimizer.variance);
    *parameter = (Parameter){0};
}

void parameter_step(Parameter* parameter, float lr, float betta_1, float betta_2, float lambda, uint32_t t) {
    assert(parameter);
    assert(mat_same_shape(&parameter->value, &parameter->gradient));
    assert(mat_same_shape(&parameter->value, &parameter->optimizer.momentum));
    assert(mat_same_shape(&parameter->value, &parameter->optimizer.variance));

    const float betta_1_t = 1.0f - powf(betta_1, (float)t);
    const float betta_2_t = 1.0f - powf(betta_2, (float)t);

    const uint32_t n = mat_elem_count(&parameter->value);

    float* const w = parameter->value.values;
    float* const g = parameter->gradient.values;
    float* const m = parameter->optimizer.momentum.values;
    float* const v = parameter->optimizer.variance.values;
    
    for (uint32_t i = 0; i < n; ++i) {
        m[i] = betta_1 * m[i] + (1.0f - betta_1) * g[i];
        v[i] = betta_2 * v[i] + (1.0f - betta_2) * g[i] * g[i];

        float m_hat = m[i] / betta_1_t;
        float v_hat = v[i] / betta_2_t;

        w[i] -= lr * (m_hat / (sqrtf(v_hat) + 1e-5f) + lambda * w[i]);
    }

    mat_zero(&parameter->gradient);
}

Layer dense_new(uint32_t fan_in, uint32_t fan_out, DenseInitOption init_option) {
    Layer layer;
    layer.type    = LAYER_DENSE;
    layer.fan_in  = fan_in;
    layer.fan_out = fan_out;
    layer.data.dense = (DenseLayer){
        .weights = parameter_new(fan_out, fan_in),
        .biases  = parameter_new(fan_out, 1),
        .cached_activation = mat_new(fan_in,  1),
        .temp_gradient     = mat_new(fan_in,  1),
        .temp_output       = mat_new(fan_out, 1),
    };

    switch (init_option) {
        case DENSE_INIT_XAVIER:  {
            float range = sqrtf(6.0f / (float)(fan_in + fan_out));
            for (uint32_t i = 0; i < mat_elem_count(&layer.data.dense.weights.value); ++i)
                layer.data.dense.weights.value.values[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * range;
            break;
        }
        case DENSE_INIT_KAIMING: {
            float range = sqrtf(2.0f / (float)fan_in);
            for (uint32_t i = 0; i < mat_elem_count(&layer.data.dense.weights.value); ++i)
                layer.data.dense.weights.value.values[i] = ((float)rand() / (float)RAND_MAX) * range;
            break;
        }
        default: abort(); // unreachable
    }

    return layer;
}

void dense_free(DenseLayer* dense) {
    assert(dense);
    parameter_free(&dense->weights);
    parameter_free(&dense->biases);
    mat_free(&dense->cached_activation);
    mat_free(&dense->temp_output);
    mat_free(&dense->temp_gradient);
    *dense = (DenseLayer){0};
}

void dense_forward(DenseLayer* dense, Matrix* inout, bool train) {
    assert(dense);
    assert(inout);

    if (train) mat_copy_into(inout, &dense->cached_activation);

    mat_mul_into(&dense->weights.value, inout, &dense->temp_output, false, false, true);
    mat_add_into(&dense->biases.value, &dense->temp_output);

    mat_copy_into(&dense->temp_output, inout);
}

void dense_backward(DenseLayer* dense, Matrix* inout) {
    assert(dense);
    assert(inout);

    mat_mul_into(&dense->weights.value, inout, &dense->temp_gradient, true, false, true);

    mat_add_into(inout, &dense->biases.gradient);

    for (uint32_t j = 0; j < dense->weights.gradient.cols; ++j) {
        for (uint32_t i = 0; i < dense->weights.gradient.rows; ++i) {
            *mat_at(&dense->weights.gradient, i, j) +=
                mat_at_const(&dense->cached_activation, j, 0) *
                mat_at_const(inout, i, 0);
        }
    }

    mat_copy_into(&dense->temp_gradient, inout);
}

void dense_step(DenseLayer* dense, float lr, float betta_1, float betta_2, float lambda, uint32_t t) {
    assert(dense);
    parameter_step(&dense->weights, lr, betta_1, betta_2, lambda, t);
    parameter_step(&dense->biases,  lr, betta_1, betta_2, lambda, t);
}

Layer activation_new(uint32_t fan_in, Function forward, Function derivative) {
    Layer layer;
    layer.type = LAYER_ACTIVATION;
    layer.fan_in  = fan_in;
    layer.fan_out = fan_in;
    layer.data.activation = (ActivationLayer){
        .forward    = forward,
        .derivative = derivative,
        .cached_activation = mat_new(fan_in, 1),
    };
    return layer;
}

void activation_free(ActivationLayer* activation) {
    assert(activation);
    mat_free(&activation->cached_activation);
    *activation = (ActivationLayer){0};
}

void activation_forward(ActivationLayer* activation, Matrix* inout, bool train) {
    assert(activation);
    assert(inout);

    if (train) mat_copy_into(inout, &activation->cached_activation);

    for (uint32_t i = 0; i < mat_elem_count(inout); ++i)
        inout->values[i] = activation->forward(inout->values[i]);
}

void activation_backward(ActivationLayer* activation, Matrix* inout) {
    assert(activation);
    assert(inout);

    for (uint32_t i = 0; i < mat_elem_count(inout); ++i)
        inout->values[i] *= activation->derivative(activation->cached_activation.values[i]);
}

Layer layernorm_new(uint32_t fan_in) {
    Layer layer;
    layer.type    = LAYER_LAYERNORM;
    layer.fan_in  = fan_in;
    layer.fan_out = fan_in;
    layer.data.layernorm = (LaynormLayer){
        .scale = parameter_new(fan_in, 1),
        .shift = parameter_new(fan_in, 1),
        .cached_input  = mat_new(fan_in, 1),
        .cached_output = mat_new(fan_in, 1),
    };
    for (uint32_t i = 0; i < mat_elem_count(&layer.data.layernorm.scale.value); ++i)
        layer.data.layernorm.scale.value.values[i] = 1.0f;
    
    return layer;
}

void layernorm_free(LaynormLayer* layernorm) {
    assert(layernorm);
    parameter_free(&layernorm->scale);
    parameter_free(&layernorm->shift);
    mat_free(&layernorm->cached_input);
    mat_free(&layernorm->cached_output);
    *layernorm = (LaynormLayer){0};
}

static inline float sumf(float const* values, uint32_t count) {
    assert(values);
    
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; ++i)
        sum += values[i];
    return sum;
}

static inline float meanf(float const* values, uint32_t count) {
    assert(values);
    
    return sumf(values, count) / (float)count;
}

static inline float variancef(float const* values, float mean, uint32_t count) {
    assert(values);
    
    float variance = 0.0f;
    for (uint32_t i = 0; i < count; ++i)
        variance += powf(values[i] - mean, 2.0f);
    return variance / (float)count;
}

void layernorm_forward(LaynormLayer* layernorm, Matrix* inout, bool train) {
    assert(layernorm);
    assert(inout);
    
    const uint32_t n = mat_elem_count(inout);
    
    const float mean = meanf(inout->values, n);
    const float std  = sqrtf(variancef(inout->values, mean, n) + 1e-5f);

    if (train) {
        mat_copy_into(inout, &layernorm->cached_input);
        layernorm->cached_mean = mean;
        layernorm->cached_std  = std;
    }

    for (uint32_t i = 0; i < n; ++i) {
        inout->values[i] = (inout->values[i] - mean) / std;
        if (train) layernorm->cached_output.values[i] = inout->values[i];
        inout->values[i] = layernorm->scale.value.values[i] * inout->values[i] + layernorm->shift.value.values[i];
    }
}

void layernorm_backward(LaynormLayer* layernorm, Matrix* inout) {
    assert(layernorm);
    assert(inout);

    const uint32_t n = mat_elem_count(inout);

    mat_add_into(inout, &layernorm->shift.gradient);

    for (uint32_t i = 0; i < n; ++i) {
        layernorm->scale.gradient.values[i] += inout->values[i] * layernorm->cached_output.values[i];
        inout->values[i] *= layernorm->scale.value.values[i];
    }

    float sum = 0.0f;
    float sum_normalized = 0.0f;

    for (uint32_t i = 0; i < n; ++i) {
        sum += inout->values[i];
        sum_normalized += inout->values[i] * layernorm->cached_output.values[i];
    }

    for (uint32_t i = 0; i < n; ++i) {
        inout->values[i] = (inout->values[i] * (float)n - sum - layernorm->cached_output.values[i] * sum_normalized) / ((float)n * layernorm->cached_std);
    }
}

void layernorm_step(LaynormLayer* layernorm, float lr, float betta_1, float betta_2, float lambda, uint32_t t) {
    parameter_step(&layernorm->scale, lr, betta_1, betta_2, lambda, t);
    parameter_step(&layernorm->shift, lr, betta_1, betta_2, lambda, t);
}

Layer softmax_new(uint32_t fan_in, float temp) {
    assert(temp > 0.0f);
    Layer layer;
    layer.type    = LAYER_SOFTMAX;
    layer.fan_in  = fan_in;
    layer.fan_out = fan_in;
    layer.data.softmax = (SoftmaxLayer){
        .temp = temp,
        .cached_output = mat_new(fan_in, 1),
    };
    return layer;
}

void softmax_free(SoftmaxLayer* softmax) {
    assert(softmax);
    mat_free(&softmax->cached_output);
    *softmax = (SoftmaxLayer){0};
}

void softmax_forward(SoftmaxLayer* softmax, Matrix* inout, bool train) {
    assert(softmax);
    assert(inout);
    assert(inout->rows >= 1);

    const uint32_t n = mat_elem_count(inout);

    float max = inout->values[0];

    for (uint32_t i = 1; i < n; ++i)
        if (max < inout->values[i])
            max = inout->values[i];

    float exps[n];

    for (uint32_t i = 0; i < n; ++i)
        exps[i] = expf((inout->values[i] - max) / softmax->temp);

    float sum = sumf(exps, n);

    for (uint32_t i = 0; i < n; ++i)
        inout->values[i] = exps[i] / sum;

    if (train) mat_copy_into(inout, &softmax->cached_output);
}

void softmax_backward(SoftmaxLayer* softmax, Matrix* inout) {
    assert(softmax);
    assert(inout);
    assert(softmax->cached_output.rows == inout->rows);

    const uint32_t n = mat_elem_count(inout);

    float dot = 0.0f;

    for (uint32_t i = 0; i < n; ++i)
        dot += inout->values[i] * softmax->cached_output.values[i];

    for (uint32_t i = 0; i < n; ++i)
        inout->values[i] = softmax->cached_output.values[i] * (inout->values[i] - dot) / softmax->temp;
}

void layer_forward(Layer* layer, Matrix* inout, bool train) {
    assert(layer);
    assert(inout);

    switch (layer->type) {
        case LAYER_DENSE: dense_forward(&layer->data.dense, inout, train); break;
        case LAYER_ACTIVATION: activation_forward(&layer->data.activation, inout, train); break;
        case LAYER_LAYERNORM: layernorm_forward(&layer->data.layernorm, inout, train); break;
        case LAYER_SOFTMAX: softmax_forward(&layer->data.softmax, inout, train); break;
        default: abort(); // unreachable
    }
}

void layer_backward(Layer* layer, Matrix* inout) {
    assert(layer);
    assert(inout);

    switch (layer->type) {
        case LAYER_DENSE: dense_backward(&layer->data.dense, inout); break;
        case LAYER_ACTIVATION: activation_backward(&layer->data.activation, inout); break;
        case LAYER_LAYERNORM: layernorm_backward(&layer->data.layernorm, inout); break;
        case LAYER_SOFTMAX: softmax_backward(&layer->data.softmax, inout); break;
        default: abort(); // unreachable
    }
}

void layer_free(Layer* layer) {
    assert(layer);

    switch (layer->type) {
        case LAYER_DENSE: dense_free(&layer->data.dense); break;
        case LAYER_ACTIVATION: activation_free(&layer->data.activation); break;
        case LAYER_LAYERNORM: layernorm_free(&layer->data.layernorm); break;
        case LAYER_SOFTMAX: softmax_free(&layer->data.softmax); break;
        default: abort(); // unreachable
    }
}

void layer_step(Layer* layer, float lr, float betta_1, float betta_2, float lambda, uint32_t t) {
    assert(layer);

    switch (layer->type) {
        case LAYER_DENSE: dense_step(&layer->data.dense, lr, betta_1, betta_2, lambda, t); break;
        case LAYER_ACTIVATION: break;
        case LAYER_LAYERNORM: layernorm_step(&layer->data.layernorm, lr, betta_1, betta_2, lambda, t); break;
        case LAYER_SOFTMAX: break;
        default: abort(); // unreachable
    }
}

LearningRate constant_learning_rate(float lr) {
    return (LearningRate){
        .type = LEARNING_RATE_CONSTANT,
        .data.constant = (ConstantLearningRate){
            .learning_rate = lr,
        },
    };
}

LearningRate exponential_learning_rate(float lr_begin, float lr_end, float exponent, uint32_t expected_steps) {
    return (LearningRate){
        .type = LEARNING_RATE_EXPONENTIAL,
        .data.exponential = (ExponentialLearningRate){
            .lr_begin       = lr_begin,
            .lr_end         = lr_end,
            .exponent       = exponent,
            .expected_steps = expected_steps,
        },
    };
}

float get_learning_rate(LearningRate learning_rate, uint32_t step) {
    switch (learning_rate.type) {
        case LEARNING_RATE_CONSTANT: return learning_rate.data.constant.learning_rate;
        case LEARNING_RATE_EXPONENTIAL: {
            float t = (float)step / (float)learning_rate.data.exponential.expected_steps;
            if (t > 1.0f) t = 1.0f;
            return learning_rate.data.exponential.lr_end
                + (learning_rate.data.exponential.lr_begin - learning_rate.data.exponential.lr_end)
                * powf(1.0f - t, learning_rate.data.exponential.exponent);
        }
        default: abort();  // unreachable
    }
}

MLP mlp_new(void) {
    return (MLP){
        .learning_rate  = constant_learning_rate(0.005f),
        .loss_function  = LOSS_MSE,
        .betta_1        = 0.9f,
        .betta_2        = 0.999f,
        .lambda         = 0.001f,
        .mini_batch     = 4,

        .t = 1,
        .optimizer_step = 0,
        .layer_count = 0,
        .layers = NULL,
    };
}

float leaky_relu(float x) {
    return (x > 0 ? x : 0.1f * x);
}

float leaky_relu_derivative(float x) {
    return (x > 0 ? 1 : 0.1f);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

void mlp_free(MLP* mlp) {
    assert(mlp);

    for (uint32_t i = 0; i < mlp->layer_count; ++i)
        layer_free(&mlp->layers[i]);
    
    if (mlp->layers) free(mlp->layers);
    *mlp = (MLP){0};
}

void mlp_push_layer(MLP* mlp, Layer layer) {
    assert(mlp);

    uint32_t n = mlp->layer_count;

    mlp->layers = (Layer*)realloc(mlp->layers, ++mlp->layer_count * sizeof(Layer));

    mlp->layers[n] = layer;
}

static inline void mlp_forward_helper(MLP* mlp, Matrix* inout, bool train) {
    assert(mlp);
    assert(inout);

    for (uint32_t i = 0; i < mlp->layer_count; ++i)
        layer_forward(&mlp->layers[i], inout, train);
}

void mlp_forward(MLP* mlp, Matrix* inout) {
    mlp_forward_helper(mlp, inout, false);
}

static inline void mlp_backward(MLP* mlp, Matrix* inout) {
    assert(mlp);
    assert(inout);

    for (uint32_t i = mlp->layer_count; i-- > 0;)
        layer_backward(&mlp->layers[i], inout);
}

void mlp_train(MLP* mlp, Matrix const* input, Matrix const* desired_output) {
    assert(mlp);
    assert(input);
    assert(desired_output);

    Matrix inout = mat_new(input->rows, input->cols);
    mat_copy_into(input, &inout);

    mlp_forward_helper(mlp, &inout, true);

    switch (mlp->loss_function) {
        case LOSS_MSE: {
            const uint32_t n = mat_elem_count(&inout);
            
            for (uint32_t i = 0; i < n; ++i) {
                const float y = desired_output->values[i];
                const float p = inout.values[i];
                
                inout.values[i] = (p - y);
            }
            break;
        }
        case LOSS_MAE: {
            const uint32_t n = mat_elem_count(&inout);
            
            for (uint32_t i = 0; i < n; ++i) {
                const float y = desired_output->values[i];
                const float p = inout.values[i];
                
                inout.values[i] = (p > y ? 1 : -1);
            }
            break;
        }
        case LOSS_BINARY_CROSS_ENTROPY: {
            const uint32_t n = mat_elem_count(&inout);
            
            for (uint32_t i = 0; i < n; ++i) {
                const float y = desired_output->values[i];
                const float p = inout.values[i];

                inout.values[i] = (p - y) / (p * (1 - p));
            }
            break;
        }
        case LOSS_CATEGORICAL_CROSS_ENTROPY: {
            const uint32_t n = mat_elem_count(&inout);
            
            for (uint32_t i = 0; i < n; ++i) {
                const float y = desired_output->values[i];
                const float p = inout.values[i];

                inout.values[i] = -y / p;
            }
            break;
        }
        default: abort();
    }

    mat_mulf(&inout, 1.0f / (float)mlp->mini_batch);

    mlp_backward(mlp, &inout);

    mat_free(&inout);

    float lr = get_learning_rate(mlp->learning_rate, mlp->optimizer_step);

    mlp->optimizer_step++;

    if ((mlp->optimizer_step % mlp->mini_batch) == 0) {
        for (uint32_t i = 0; i < mlp->layer_count; ++i)
            layer_step(&mlp->layers[i], lr, mlp->betta_1, mlp->betta_2, mlp->lambda, mlp->t);

        mlp->t++;
    }
}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MINUSCULE_IMPLEMENTATION