#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

#ifndef MINUSCULE_H
#define MINUSCULE_H

#define MAX_INCOMING 2
#define MAX_OUTGOING 2

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct {
    float* values;
    uint32_t rows;
    uint32_t cols;
} Matrix;

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAMW,
} OptimizerType;

typedef struct {
    float betta_1;
    float betta_2;
    float lambda;
} AdamWConfig;

typedef struct {
    OptimizerType type;
    union {
        AdamWConfig adamw;
    } data;
} OptimizerConfig;

typedef struct {
    Matrix momentum;
    Matrix variance;
    AdamWConfig config;
} AdamW;

typedef struct {
    OptimizerType type;

    union {
        AdamW adamw;
    } data;
} Optimizer;

typedef struct {
    Matrix value;
    Matrix gradient;
    Optimizer optimizer;
} Parameter;

typedef enum {
    LAYER_DENSE,
    LAYER_ACTIVATION,
    LAYER_RMSNORM,
    LAYER_SOFTMAX,
} LayerType;

typedef struct {
    Parameter weights;
    Parameter biases;
    Matrix cached_input;
    Matrix temp_output;
    Matrix temp_gradient;
} DenseLayer;

typedef float(*Function)(float);

typedef struct {
    Function forward;
    Function backward;
    Matrix cached_input;
} Activationlayer;

typedef struct {
    Parameter scale;
    Matrix cached_input;
} RMSNormLayer;

typedef struct {
    float temperatur;
    Matrix cached_output;
} SoftmaxLayer;

typedef struct {
    LayerType type;
    uint32_t fan_in, fan_out;
    Matrix input, output;
    Matrix gradient_input, gradient_output;
    uint32_t outgoing[MAX_OUTGOING];
    uint32_t incoming[MAX_INCOMING];

    union {
        DenseLayer      dense;
        Activationlayer activation;
        RMSNormLayer    rmsnorm;
        SoftmaxLayer    softmax;
    } data;
} Layer;

typedef enum {
    LOSS_MSE,
    LOSS_MAE,
    LOSS_BINARY_CROSS_ENTROPY,
    LOSS_CATEGORICAL_CROSS_ENTROPY, // can only be used with > 1 output feature(s)
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
        ConstantLearningRate constant;
        ExponentialLearningRate exponential;
    } data;
} LearningRate;

typedef struct {
    LearningRate learning_rate;
    LossFunction loss_function;
    Layer* layers;
    
    uint32_t mini_batch;
    uint32_t layer_count;
} MLP;

static inline Matrix mat_new(uint32_t rows, uint32_t cols) {
    assert((rows * cols) > 0);

    return (Matrix){
        .values = calloc(rows * cols, sizeof(float)),
        .rows   = rows,
        .cols   = cols,
    };
}

static inline void mat_free(Matrix* m) {
    assert(m);
    free(m->values);
    *m = (Matrix){0};
}

static inline void mat_ensure_shape(Matrix* m, uint32_t rows, uint32_t cols) {
    assert(m);
    if (m->rows == rows && m->cols == cols) return;
    mat_free(m);
    *m = mat_new(rows, cols);
}

static inline uint32_t _mat_index(uint32_t row, uint32_t col, uint32_t rows) {
    return row + col * rows;
}

static inline float* mat_at(Matrix* m, uint32_t row, uint32_t col) {
    assert(m);
    assert(row < m->rows);
    assert(col < m->cols);
    return &m->values[_mat_index(row, col, m->rows)];
}

static inline float mat_at_const(Matrix const* m, uint32_t row, uint32_t col) {
    assert(m);
    assert(row < m->rows);
    assert(col < m->cols);
    return m->values[_mat_index(row, col, m->rows)];
}

static inline void mat_move_into(Matrix* src, Matrix* dest) {
    assert(src);
    assert(dest);
    *dest = *src;
    *src  = (Matrix){0};
}

static inline uint32_t mat_element(Matrix const* m) {
    assert(m);
    return m->rows * m->cols;
}

static inline void mat_copy_into(Matrix const* src, Matrix* dest) {
    assert(src);
    assert(dest);
    mat_ensure_shape(dest, src->rows, src->cols);

    for (uint32_t i = 0; i < mat_element(src); ++i)
        dest->values[i] = src->values[i];
}

static inline Matrix mat_copy(Matrix const* src) {
    assert(src);
    Matrix result = mat_new(src->rows, src->cols);

    for (uint32_t i = 0; i < mat_element(src); ++i)
        result.values[i] = src->values[i];

    return result;
}

static inline bool mat_same_shape(Matrix const* a, Matrix const* b) {
    return a->rows == b->rows && a->cols == b->cols;
}

static inline void mat_add_into(Matrix const* a, Matrix* b) {
    assert(a);
    assert(b);
    assert(mat_same_shape(a, b));

    for (uint32_t i = 0; i < mat_element(a); ++i)
        b->values[i] += a->values[i];
}

static inline void mat_zero(Matrix* m) {
    assert(m);
    for (uint32_t i = 0; i < mat_element(m); ++i)
        m->values[i] = 0.0f;
}

static inline void mat_sub_into(Matrix const* a, Matrix* b) {
    assert(a);
    assert(b);
    assert(mat_same_shape(a, b));

    for (uint32_t i = 0; i < mat_element(a); ++i)
        b->values[i] -= a->values[i];
}

static inline void mat_mulf(Matrix* m, float f) {
    assert(m);
    for (uint32_t i = 0; i < mat_element(m); ++i)
        m->values[i] *= f;
}

static inline void _mat_mul_into_nn(Matrix const* a, Matrix const* b, Matrix* out) {
    for (uint32_t i = 0; i < a->rows; ++i)
        for (uint32_t k = 0; k < b->cols; ++k)
            for (uint32_t j = 0; j < a->cols; ++j)
                *mat_at(out, i, k) +=
                    mat_at_const(a, i, j) *
                    mat_at_const(b, j, k);
}

static inline void _mat_mul_into_nt(Matrix const* a, Matrix const* b, Matrix* out) {
    abort();
}

static inline void _mat_mul_into_tn(Matrix const* a, Matrix const* b, Matrix* out) {
    for (uint32_t k = 0; k < b->cols; ++k)
        for (uint32_t i = 0; i < a->cols; ++i)
            for (uint32_t j = 0; j < a->rows; ++j)
                *mat_at(out, i, k) +=
                    mat_at_const(a, j, i) *
                    mat_at_const(b, j, k);
}

static inline void _mat_mul_into_tt(Matrix const* a, Matrix const* b, Matrix* out) {
    abort();
}

static inline void mat_mul_into(Matrix const* a, Matrix const* b, Matrix* out, bool transpose_a, bool transpose_b, bool zero) {
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

    uint8_t flag = (transpose_a << 1) | (transpose_b << 0);

    switch (flag) {
        case 0: _mat_mul_into_nn(a, b, out); break;
        case 1: _mat_mul_into_nt(a, b, out); break;
        case 2: _mat_mul_into_tn(a, b, out); break;
        case 3: _mat_mul_into_tt(a, b, out); break;
    }
}

static inline OptimizerConfig adamw_new(float betta_1, float betta_2, float lambda) {
    return (OptimizerConfig){
        .type = OPTIMIZER_ADAMW,
        .data.adamw  = (AdamWConfig){
            .betta_1 = betta_1,
            .betta_2 = betta_2,
            .lambda  = lambda,
        },
    };
}

static inline OptimizerConfig sgd_new(void) {
    return (OptimizerConfig){
        .type = OPTIMIZER_SGD,
    };
}

static inline Optimizer optimizer_new(OptimizerConfig config, uint32_t rows, uint32_t cols) {
    switch (config.type) {
        case OPTIMIZER_SGD: return (Optimizer){.type = OPTIMIZER_SGD};
        case OPTIMIZER_ADAMW: {
            Optimizer optimizer;
            optimizer.type = OPTIMIZER_ADAMW;
            optimizer.data.adamw = (AdamW){
                .momentum = mat_new(rows, cols),
                .variance = mat_new(rows, cols),
                .config = config.data.adamw,
            };
            return optimizer;
        }
    }
}

static inline void optimizer_free(Optimizer* optimizer) {
    assert(optimizer);
    switch (optimizer->type) {
        case OPTIMIZER_SGD: break;
        case OPTIMIZER_ADAMW: {
            mat_free(&optimizer->data.adamw.momentum);
            mat_free(&optimizer->data.adamw.variance);
            break;
        }
    }
    *optimizer = (Optimizer){0};
}

static inline LearningRate constant_learning_rate(float lr) {
    return (LearningRate){
        .type = LEARNING_RATE_CONSTANT,
        .data.constant = (ConstantLearningRate){
            .learning_rate = lr,
        }
    };
}

static inline LearningRate exponential_learning_rate(float lr_begin, float lr_end, float exponent, uint32_t expected_steps) {
    return (LearningRate){
        .type = LEARNING_RATE_EXPONENTIAL,
        .data.exponential   = (ExponentialLearningRate){
            .lr_begin       = lr_begin,
            .lr_end         = lr_end,
            .exponent       = exponent,
            .expected_steps = expected_steps,
        }
    };
}

static inline float learningrate_get_current(LearningRate lr, uint32_t t);
static inline float learningrate_get_base(LearningRate lr);

static inline void sgd_step(Matrix* value, Matrix* gradient, LearningRate lr, uint32_t t) {
    assert(value);
    assert(gradient);
    assert(mat_same_shape(value, gradient));

    mat_mulf(gradient, learningrate_get_current(lr, t));

    mat_sub_into(value, gradient);

    mat_zero(gradient);
}

static inline void adamw_step(Matrix* value, Matrix* gradient, AdamW* adamw, LearningRate lr, uint32_t t) {
    assert(value);
    assert(gradient);
    assert(mat_same_shape(value, gradient));

    const float lambda    = adamw->config.lambda;
    const float betta_1   = adamw->config.betta_1;
    const float betta_2   = adamw->config.betta_2;
    const float betta_1_t = 1.0f - powf(adamw->config.betta_1, (float)t);
    const float betta_2_t = 1.0f - powf(adamw->config.betta_2, (float)t);

    const float alpha = learningrate_get_base(lr);
    const float n     = learningrate_get_current(lr, t);

    float* w = &value->values[0];
    float* g = &gradient->values[0];
    float* m = &adamw->momentum.values[0];
    float* v = &adamw->variance.values[0];
    
    for (uint32_t i = 0; i < mat_element(value); ++i) {
        m[i] = betta_1 * m[i] + (1.0f - betta_1) * g[i];
        v[i] = betta_2 * v[i] + (1.0f - betta_2) * g[i] * g[i];

        const float m_hat = m[i] / (1.0f - betta_1_t);
        const float v_hat = v[i] / (1.0f - betta_2_t);

        w[i] = w[i] - n * (alpha * m_hat / (sqrtf(v_hat + 1e-5f)) + lambda * w[i]);
    }
}

static inline void optimizer_step(Matrix* value, Matrix* gradient, Optimizer* optimizer, LearningRate lr, uint32_t t) {
    assert(value);
    assert(gradient);
    assert(optimizer);
    switch (optimizer->type) {
        case OPTIMIZER_SGD: sgd_step(value, gradient, lr, t); break;
        case OPTIMIZER_ADAMW: adamw_step(value, gradient, &optimizer->data.adamw, lr, t); break;
    }
}

static inline Parameter parameter_new(uint32_t rows, uint32_t cols, OptimizerConfig optimizer_config) {
    return (Parameter){
        .value     = mat_new(rows, cols),
        .gradient  = mat_new(rows, cols),
        .optimizer = optimizer_new(optimizer_config, rows, cols),
    };
}

static inline void parameter_free(Parameter* parameter) {
    assert(parameter);
    mat_free(&parameter->gradient);
    mat_free(&parameter->value);
    optimizer_free(&parameter->optimizer);
    *parameter = (Parameter){0};
}

static inline void parameter_step(Parameter* parameter, LearningRate lr, uint32_t t) {
    assert(parameter);
    optimizer_step(&parameter->value, &parameter->gradient, &parameter->optimizer, lr, t);
}

static inline Layer _layer_new(LayerType type, uint32_t fan_in, uint32_t fan_out) {
    Layer layer;
    layer.type            = LAYER_DENSE;
    layer.fan_in          = fan_in;
    layer.fan_out         = fan_out;
    layer.input           = mat_new(fan_in,  1);
    layer.output          = mat_new(fan_out, 1);
    layer.gradient_input  = mat_new(fan_out, 1);
    layer.gradient_output = mat_new(fan_in,  1);

    for (uint32_t i = 0; i < MAX_INCOMING; ++i) layer.incoming[i] = UINT32_MAX;
    for (uint32_t i = 0; i < MAX_OUTGOING; ++i) layer.outgoing[i] = UINT32_MAX;

    return layer;
}

static inline Layer dense_new(uint32_t fan_in, uint32_t fan_out, OptimizerConfig optimizer_config) {
    Layer layer = _layer_new(LAYER_DENSE, fan_in, fan_out);

    layer.data.dense = (DenseLayer){
        .weights = parameter_new(fan_out, fan_in, optimizer_config),
        .biases  = parameter_new(fan_out, 1, optimizer_config),
        .cached_input  = mat_new(fan_in,  1),
        .temp_output   = mat_new(fan_out, 1),
        .temp_gradient = mat_new(fan_out, fan_in),
    };

    return layer;
}

static inline Layer activation_new(uint32_t fan_in, Function forward, Function backward) {
    Layer layer = _layer_new(LAYER_ACTIVATION, fan_in, fan_in);

    layer.data.activation = (Activationlayer){
        .forward  = forward,
        .backward = backward,
        .cached_input = mat_new(fan_in, 1),
    };
    
    return layer;
}

static inline Layer softmax_new(uint32_t fan_in, float temperatur) {
    Layer layer = _layer_new(LAYER_SOFTMAX, fan_in, fan_in);
    
    layer.data.softmax = (SoftmaxLayer){
        .temperatur = temperatur,
        .cached_output = mat_new(fan_in, 1),
    };
    
    return layer;
}

static inline Layer rmsnorm_new(uint32_t fan_in, OptimizerConfig optimizer_config) {
    Layer layer = _layer_new(LAYER_RMSNORM, fan_in, fan_in);

    layer.data.rmsnorm = (RMSNormLayer){
        .scale  = parameter_new(fan_in, 1, optimizer_config),
        .cached_input = mat_new(fan_in, 1),
    };

    return layer;
}

static inline void dense_free(DenseLayer* dense) {
    assert(dense);
    mat_free(&dense->cached_input);
    mat_free(&dense->temp_output);
    mat_free(&dense->temp_gradient);
    parameter_free(&dense->weights);
    parameter_free(&dense->biases);
    *dense = (DenseLayer){0};
}

static inline void activation_free(Activationlayer* activation) {
    assert(activation);
    mat_free(&activation->cached_input);
    *activation = (Activationlayer){0};
}

static inline void softmax_free(SoftmaxLayer* softmax) {
    assert(softmax);
    mat_free(&softmax->cached_output);
    *softmax = (SoftmaxLayer){0};
}

static inline void rmsnorm_free(RMSNormLayer* rmsnorm) {
    assert(rmsnorm);
    mat_free(&rmsnorm->cached_input);
    parameter_free(&rmsnorm->scale);
    *rmsnorm = (RMSNormLayer){0};
}

static inline void layer_free(Layer* layer) {
    assert(layer);
    mat_free(&layer->input);
    mat_free(&layer->output);
    mat_free(&layer->gradient_input);
    mat_free(&layer->gradient_output);

    switch (layer->type) {
        case LAYER_DENSE:      dense_free(&layer->data.dense); break;
        case LAYER_ACTIVATION: activation_free(&layer->data.activation); break;
        case LAYER_SOFTMAX:    softmax_free(&layer->data.softmax); break;
        case LAYER_RMSNORM:    rmsnorm_free(&layer->data.rmsnorm); break;
    }

    *layer = (Layer){0};
}

static inline void dense_forward(DenseLayer* dense, Matrix const* input, Matrix* output, bool train);
static inline void activation_forward(Activationlayer* activation, Matrix const* input, Matrix* output, bool train);
static inline void softmax_forward(SoftmaxLayer* softmax, Matrix const* input, Matrix* output, bool train);
static inline void rmsnorm_forward(RMSNormLayer* rmsnorm, Matrix const* input, Matrix* output, bool train);

static inline void dense_backward(DenseLayer* dense, Matrix const* input, Matrix* output);
static inline void activation_backward(Activationlayer* activation, Matrix const* input, Matrix* output);
static inline void softmax_backward(SoftmaxLayer* softmax, Matrix const* input, Matrix* output);
static inline void rmsnorm_backward(RMSNormLayer* rmsnorm, Matrix const* input, Matrix* output);

static inline void layer_forward(Layer* layer, Matrix const* input, Matrix* output, bool train) {
    assert(layer);
    assert(input);
    assert(output);
    
    switch (layer->type) {
        case LAYER_DENSE: dense_forward(&layer->data.dense, input, output, train); break;
        case LAYER_ACTIVATION: activation_forward(&layer->data.activation, input, output, train); break;
        case LAYER_SOFTMAX: softmax_forward(&layer->data.softmax, input, output, train); break;
        case LAYER_RMSNORM: rmsnorm_forward(&layer->data.rmsnorm, input, output, train); break;
    }
}

static inline void layer_backward(Layer* layer, Matrix const* input, Matrix* output) {
    assert(layer);
    assert(input);
    assert(output);
    
    switch (layer->type) {
        case LAYER_DENSE: dense_backward(&layer->data.dense, input, output); break;
        case LAYER_ACTIVATION: activation_backward(&layer->data.activation, input, output); break;
        case LAYER_SOFTMAX: softmax_backward(&layer->data.softmax, input, output); break;
        case LAYER_RMSNORM: rmsnorm_backward(&layer->data.rmsnorm, input, output); break;
    }
}

static inline MLP mlp_new(void) {
    return (MLP){
        .learning_rate = constant_learning_rate(0.005f),
        .loss_function = LOSS_MSE,
        .mini_batch    = 4,
    };
}

static inline void mlp_free(MLP* mlp) {
    assert(mlp);

    for (uint32_t i = 0; i < mlp->layer_count; ++i)
        layer_free(&mlp->layers[i]);
    
    if (mlp->layers) free(mlp->layers);

    *mlp = (MLP){0};
}

static inline uint32_t mlp_push_layer(MLP* mlp, Layer layer) {
    assert(mlp);
    mlp->layers = realloc(mlp->layers, ++mlp->layer_count * sizeof(Layer));
    return (mlp->layer_count - 1);
}

static inline void mlp_connect_layer(MLP* mlp, uint32_t output_from, uint32_t into_input) {
    assert(mlp);
    assert(output_from < mlp->layer_count);
    assert(into_input  < mlp->layer_count);
    assert(output_from < into_input); // assert that the activations are always flowing upstream
    assert(mlp->layers[output_from].fan_out == mlp->layers[into_input].fan_in); // check correct dimension size

    // Linear search that searches for empty slots and connects both layers

    bool found_empty_incoming = false;
    bool found_empty_outgoing = false;
    
    for (uint32_t i = 0; i < MAX_OUTGOING; ++i)
        if (mlp->layers[output_from].outgoing[i] == UINT32_MAX) {
            mlp->layers[output_from].outgoing[i] = into_input;
            found_empty_outgoing = true;
            break;
        }

    for (uint32_t i = 0; i < MAX_INCOMING; ++i)
        if (mlp->layers[into_input].incoming[i] == UINT32_MAX) {
            mlp->layers[into_input].incoming[i] = output_from;
            found_empty_incoming = true;
            break;
        }

    assert(found_empty_incoming);
    assert(found_empty_outgoing);
}

static inline void _mlp_forward(MLP* mlp, Matrix* inout, bool train) {
    assert(mlp);
    assert(inout);

    for (uint32_t i = 0; i < mlp->layer_count; ++i)
        mat_zero(&mlp->layers[i].input);

    mat_copy_into(inout, &mlp->layers[0].input);

    for (uint32_t i = 0; i < mlp->layer_count; ++i) {
        layer_forward(&mlp->layers[i], &mlp->layers[i].input, &mlp->layers[i].output, train);

        for (uint32_t c = 0; c < MAX_OUTGOING; ++c) {
            const uint32_t connection = mlp->layers[i].outgoing[c];
            if (connection == UINT32_MAX) continue;
            mat_add_into(&mlp->layers[i].output, &mlp->layers[connection].input);
        }
    }

    mat_copy_into(&mlp->layers[mlp->layer_count - 1].output, inout);
}

static inline void mlp_forward(MLP* mlp, Matrix* inout) {
    _mlp_forward(mlp, inout, false);
}

// static inline void mlp_train(MLP* mlp, Matrix const* input, Matrix const* desired_output) {
//     Matrix activation = mat_copy(input);
//     _mlp_forward(mlp, &activation, true);
// }

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MINUSCULE_IMPLEMENTATION