#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

#ifndef MINUS_H
#define MINUS_H

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
    Matrix param;
    Matrix gradient;
    AdamW optimizer;
} Parameter;

Parameter parameter_new(uint32_t rows, uint32_t cols);
void parameter_free(Parameter* parameter);
void parameter_step(Parameter* parameter, float lr, float betta_1, float betta_2, float lambda, uint32_t t);

typedef enum {
    LAYER_DENSE,
    LAYER_ACTIVATION,
} LayerType;

typedef struct {
    Parameter weights;
    Parameter biases;
    Matrix cached_activation;
} DenseLayer;

typedef float(*Function)(float x);

typedef struct {
    Function forward;
    Function derivative;
    Matrix cached_activation;
} ActivationLayer;

typedef struct {
    LayerType type;
    uint32_t fan_in;
    uint32_t fan_out;

    union {
        DenseLayer dense;
        ActivationLayer activation;
    } data;
} Layer;

typedef struct {
    float lr;
    float betta_1;
    float betta_2;
    float lambda;
    uint32_t mini_batch;
    uint32_t t;

    uint32_t layer_count;
    Layer* layers;
} MLP;

Layer dense_new(uint32_t fan_in, uint32_t fan_out);
void dense_free(DenseLayer* dense);
void dense_forward (DenseLayer* dense, Matrix* inout, bool train, uint32_t fan_in, uint32_t fan_out);
void dense_backward(DenseLayer* dense, Matrix* inout, uint32_t fan_in, uint32_t fan_out);
void dense_step    (DenseLayer* dense, float lr, float betta_1, float betta_2, float lambda, uint32_t t);

Layer activation_new(uint32_t fan_in, Function forward, Function derivative);
void activation_free(ActivationLayer* activation);
void activation_forward (ActivationLayer* activation, Matrix* inout, bool train, uint32_t fan_in, uint32_t fan_out);
void activation_backward(ActivationLayer* activation, Matrix* inout, uint32_t fan_in, uint32_t fan_out);

void layer_forward (Layer* layer, Matrix* inout, bool train, uint32_t fan_in, uint32_t fan_out);
void layer_backward(Layer* layer, Matrix* inout, uint32_t fan_in, uint32_t fan_out);
void layer_free    (Layer* layer);
void layer_step    (Layer* layer, float lr, float betta_1, float betta_2, float lambda, uint32_t t);

MLP mlp_new(void);
void mlp_free(MLP* mlp);

void mlp_push_layer(MLP* mlp, Layer layer);

void mlp_forward(MLP* mlp, Matrix* inout);
void mlp_train(MLP* mlp, Matrix const* input, Matrix const* desired_output);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MINUS_H


// --- Implementation --

#define MINUS_IMPLEMENTATION

#ifdef MINUS_IMPLEMENTATION

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

Matrix mat_new(uint32_t rows, uint32_t cols) {
    assert(rows > 0);
    assert(cols > 0);

    return (Matrix){
        .rows = rows,
        .cols = cols,
        .values = calloc(rows * cols, sizeof(float)),
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

    mat_free(m);
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
    return m->values[mat_index(row, col, m->rows)];
}

float* mat_at(Matrix* m, uint32_t row, uint32_t col) {
    assert(m);
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

    uint8_t flags = (transpose_a << 1) | (transpose_b << 0);

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
        .param     = mat_new(rows, cols),
        .gradient  = mat_new(rows, cols),
        .optimizer = (AdamW){
            .momentum = mat_new(rows, cols),
            .variance = mat_new(rows, cols),
        },
    };
}

void parameter_free(Parameter* parameter) {
    assert(parameter);
    mat_free(&parameter->param);
    mat_free(&parameter->gradient);
    mat_free(&parameter->optimizer.momentum);
    mat_free(&parameter->optimizer.variance);
    *parameter = (Parameter){0};
}

void parameter_step(Parameter* parameter, float lr, float betta_1, float betta_2, float lambda, uint32_t t) {
    assert(parameter);
    assert(mat_same_shape(&parameter->param, &parameter->gradient));
    assert(mat_same_shape(&parameter->param, &parameter->optimizer.momentum));
    assert(mat_same_shape(&parameter->param, &parameter->optimizer.variance));

    const float betta_1_t = 1.0 - powf(betta_1, t);
    const float betta_2_t = 1.0 - powf(betta_2, t);

    const uint32_t n = mat_elem_count(&parameter->param);

    float* const w = parameter->param.values;
    float* const g = parameter->gradient.values;
    float* const m = parameter->optimizer.momentum.values;
    float* const v = parameter->optimizer.variance.values;
    
    for (uint32_t i = 0; i < n; ++i) {
        m[i] = betta_1 * m[i] + (1.0 - betta_1) * g[i];
        v[i] = betta_2 * v[i] + (1.0 - betta_2) * g[i] * g[i];

        float m_hat = m[i] / betta_1_t;
        float v_hat = v[i] / betta_2_t;

        w[i] -= lr * (m_hat / (sqrtf(v_hat) + 1e-5f) + lambda * w[i]);
    }

    mat_zero(&parameter->gradient);
}

Layer dense_new(uint32_t fan_in, uint32_t fan_out) {
    Layer layer;
    layer.type    = LAYER_DENSE;
    layer.fan_in  = fan_in;
    layer.fan_out = fan_out;
    layer.data.dense = (DenseLayer){
        .weights = parameter_new(fan_out, fan_in),
        .biases  = parameter_new(fan_out, 1),
        .cached_activation = mat_new(fan_out, 1),
    };

    for (uint32_t i = 0; i < mat_elem_count(&layer.data.dense.weights.param); ++i)
        layer.data.dense.weights.param.values[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; // TODO: Xavier/Kaiming Initilization
    
    return layer;
}

void dense_free(DenseLayer* dense) {
    assert(dense);
    parameter_free(&dense->weights);
    parameter_free(&dense->biases);
    mat_free(&dense->cached_activation);
    *dense = (DenseLayer){0};
}

void dense_forward(DenseLayer* dense, Matrix* inout, bool train, uint32_t fan_in, uint32_t fan_out) {
    assert(dense);
    assert(inout);

    if (train) mat_copy_into(inout, &dense->cached_activation);

    Matrix temp = mat_new(fan_out, 1);

    mat_mul_into(&dense->weights.param, inout, &temp, false, false, true);
    mat_add_into(&dense->biases.param, &temp);

    mat_move_into(&temp, inout);
}

void dense_backward(DenseLayer* dense, Matrix* inout, uint32_t fan_in, uint32_t fan_out) {
    assert(dense);
    assert(inout);

    Matrix prev_grad = mat_new(inout->rows, 1);
    mat_mul_into(&dense->weights.param, inout, &prev_grad, true, false, true);

    mat_add_into(inout, &dense->biases.gradient);

    for (uint32_t j = 0; j < dense->weights.gradient.cols; ++j) {
        for (uint32_t i = 0; i < dense->weights.gradient.rows; ++i) {
            *mat_at(&dense->weights.gradient, i, j) +=
                mat_at_const(&dense->cached_activation, j, 0) *
                mat_at_const(inout, i, 0);
        }
    }

    mat_move_into(&prev_grad, inout);
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

void activation_forward(ActivationLayer* activation, Matrix* inout, bool train, uint32_t fan_in, uint32_t fan_out) {
    assert(activation);
    assert(inout);

    if (train) mat_copy_into(inout, &activation->cached_activation);

    for (uint32_t i = 0; i < mat_elem_count(inout); ++i)
        inout->values[i] = activation->forward(inout->values[i]);
}

void activation_backward(ActivationLayer* activation, Matrix* inout, uint32_t fan_in, uint32_t fan_out) {
    assert(activation);
    assert(inout);

    for (uint32_t i = 0; i < mat_elem_count(inout); ++i)
        inout->values[i] *= activation->derivative(inout->values[i]);
}

void layer_forward(Layer* layer, Matrix* inout, bool train, uint32_t fan_in, uint32_t fan_out) {
    assert(layer);
    assert(inout);

    switch (layer->type) {
        case LAYER_DENSE: dense_forward(&layer->data.dense, inout, train, fan_in, fan_out); break;
        case LAYER_ACTIVATION: activation_forward(&layer->data.activation, inout, train, fan_in, fan_out); break;
        default: abort(); // unreachable
    }
}

void layer_backward(Layer* layer, Matrix* inout, uint32_t fan_in, uint32_t fan_out) {
    assert(layer);
    assert(inout);

    switch (layer->type) {
        case LAYER_DENSE: dense_backward(&layer->data.dense, inout, fan_in, fan_out); break;
        case LAYER_ACTIVATION: activation_backward(&layer->data.activation, inout, fan_in, fan_out); break;
        default: abort(); // unreachable
    }
}

void layer_free(Layer* layer) {
    assert(layer);

    switch (layer->type) {
        case LAYER_DENSE: dense_free(&layer->data.dense); break;
        case LAYER_ACTIVATION: activation_free(&layer->data.activation); break;
        default: abort(); // unreachable
    }
}

void layer_step(Layer* layer, float lr, float betta_1, float betta_2, float lambda, uint32_t t) {
    assert(layer);

    switch (layer->type) {
        case LAYER_DENSE: dense_step(&layer->data.dense, lr, betta_1, betta_2, lambda, t); break;
        case LAYER_ACTIVATION: break;
        default: abort(); // unreachable
    }
}


MLP mlp_new(void) {
    return (MLP){
        .lr         = 0.005f,
        .betta_1    = 0.9f,
        .betta_2    = 0.999f,
        .lambda     = 0.001f,
        .mini_batch = 4,

        .t = 1,
        .layer_count = 0,
        .layers = NULL,
    };
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

    mlp->layers = realloc(mlp->layers, ++mlp->layer_count * sizeof(Layer));

    mlp->layers[n] = layer;
}

static inline void mlp_forward_helper(MLP* mlp, Matrix* inout, bool train) {
    assert(mlp);
    assert(inout);

    for (uint32_t i = 0; i < mlp->layer_count; ++i)
        layer_forward(&mlp->layers[i], inout, train, mlp->layers[i].fan_in, mlp->layers[i].fan_out);
}

void mlp_forward(MLP* mlp, Matrix* inout) {
    mlp_forward_helper(mlp, inout, false);
}

static inline void mlp_backward(MLP* mlp, Matrix* inout) {
    assert(mlp);
    assert(inout);

    for (uint32_t i = mlp->layer_count; i-- > 0;)
        layer_backward(&mlp->layers[i], inout, mlp->layers[i].fan_in, mlp->layers[i].fan_out);
}

void mlp_train(MLP* mlp, Matrix const* input, Matrix const* desired_output) {
    assert(mlp);
    assert(input);
    assert(desired_output);

    Matrix inout = mat_new(input->rows, input->cols);
    mat_copy_into(input, &inout);

    mlp_forward_helper(mlp, &inout, true);

    mat_sub_into(desired_output, &inout);

    mlp_backward(mlp, &inout);

    mat_free(&inout);

    if ((mlp->t % mlp->mini_batch) == 0)
        for (uint32_t i = 0; i < mlp->layer_count; ++i)
            layer_step(&mlp->layers[i], mlp->lr, mlp->betta_1, mlp->betta_2, mlp->lambda, mlp->t);
    
    mlp->t++;
}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MINUS_IMPLEMENTATION