#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
// #include <pthread.h>

#ifndef countof
#define countof(arr) (sizeof(arr) / sizeof(arr[0]))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MLP_H
#define MLP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;

typedef struct {
    u32 rows;
    u32 cols;
    float* values;
} Matrix;

static inline Matrix mat_empty(void) {
    return (Matrix){
        .rows   = 0,
        .cols   = 0,
        .values = NULL,
    };
}

static inline Matrix mat_new(u32 rows, u32 cols) {
    if (rows == 0 && cols == 0) return mat_empty();
    assert(rows > 0 && cols > 0);

    return (Matrix){
        .rows   = rows,
        .cols   = cols,
        .values = calloc(rows * cols, sizeof(float)),
    };
}

static inline Matrix mat_vec(u32 rows) {
    return mat_new(rows, 1);
}

static inline void mat_free(Matrix* m) {
    assert(m && "Passed NULL matrix into mat_free()");
    assert(m->values && "Passed empty  OR  already freed matrix into mat_free()!");

    free(m->values);

    *m = mat_empty();
}

static inline u32 mat_elem_count(const Matrix* m) {
    assert(m);
    return m->rows * m->cols;
}

static inline void mat_set_values(Matrix* m, float* values, u32 count) {
    assert(mat_elem_count(m) == count);
    
    for (u32 i = 0; i < mat_elem_count(m); ++i)
        m->values[i] = values[i];
}

static inline void mat_zero(Matrix* m) {
    assert(m);

    for (u32 i = 0; i < mat_elem_count(m); ++i)
        m->values[i] = 0.0f;
}

static inline bool mat_is_empty(const Matrix* m) {
    assert(m);
    assert((m->rows == 0 && m->cols == 0 && !m->values) || (m->rows > 0 && m->cols > 0 && m->values));
    return m->rows == 0 && m->cols == 0;
}

static inline Matrix mat_clone(const Matrix* m) {
    Matrix dest = mat_new(m->rows, m->cols);

    for (u32 i = 0; i < mat_elem_count(m); ++i)
        dest.values[i] = m->values[i];
    
    return dest;
}

static inline void mat_ensure_shape(Matrix* m, u32 rows, u32 cols) {
    assert(m);
    if (m->rows == rows && m->cols == cols) {return;}

    if (!mat_is_empty(m)) mat_free(m);

    *m = mat_new(rows, cols);
}

static inline void mat_clone_into(const Matrix* src, Matrix* dest) {
    if (src->rows != dest->rows || src->cols != dest->cols) {
        mat_ensure_shape(dest, src->rows, src->cols);
    }

    for (u32 i = 0; i < mat_elem_count(src); ++i)
        dest->values[i] = src->values[i];
}

static inline void mat_add_into(const Matrix* a, const Matrix* b, Matrix* out) {
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);

    mat_ensure_shape(out, a->rows, a->cols);

    for (u32 i = 0; i < mat_elem_count(a); ++i)
        out->values[i] = a->values[i] + b->values[i];
}

static inline void mat_sub_into(const Matrix* a, const Matrix* b, Matrix* out) {
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);

    mat_ensure_shape(out, a->rows, a->cols);

    for (u32 i = 0; i < mat_elem_count(a); ++i)
        out->values[i] = a->values[i] - b->values[i];
}

static inline u32 mat_index(u32 row, u32 col, u32 rows) {
    return row + col * rows; // Column-major!
}

static inline float mat_at(const Matrix* m, u32 row, u32 col) {
    assert(row < m->rows);
    assert(col < m->cols);
    return m->values[mat_index(row, col, m->rows)];
}

static inline float* mat_at_mut(Matrix* m, u32 row, u32 col) {
    assert(row < m->rows);
    assert(col < m->cols);
    return &m->values[mat_index(row, col, m->rows)];
}

static inline void _mat_mul_into_nn(const Matrix* restrict a, const Matrix* restrict b, Matrix* restrict out) {
    for (i32 i = 0; i < a->rows; ++i) {
        for (i32 k = 0; k < b->cols; ++k) {
            for (i32 j = 0; j < a->cols; ++j) {
                *mat_at_mut(out, i, k) +=
                    mat_at(a, i, j) *
                    mat_at(b, j, k);
            }
        }
    }
}

static inline void _mat_mul_into_tn(const Matrix* restrict a, const Matrix* restrict b, Matrix* restrict out) {
    abort();
}

static inline void _mat_mul_into_nt(const Matrix* restrict a, const Matrix* restrict b, Matrix* restrict out) {
    abort();
}

static inline void _mat_mul_into_tt(const Matrix* restrict a, const Matrix* restrict b, Matrix* restrict out) {
    abort();
}

static inline void mat_mul_into(const Matrix* restrict a, const Matrix* restrict b, Matrix* restrict out, bool a_T, bool b_T, bool zero) {
    assert(a && !mat_is_empty(a));
    assert(b && !mat_is_empty(b));
    assert(out);
    
    u8 flags = (a_T << 1) | (b_T << 0);

    u32 a_rows = (a_T ? a->cols : a->rows);
    u32 a_cols = (a_T ? a->rows : a->cols);
    u32 b_rows = (b_T ? b->cols : b->rows);
    u32 b_cols = (b_T ? b->rows : b->cols);

    assert(a_cols == b_rows);

    mat_ensure_shape(out, a_rows, b_cols);
    if (zero) mat_zero(out);

    switch (flags) {
        case (0 << 1 | 0 << 0): _mat_mul_into_nn(a, b, out); break;
        case (1 << 1 | 0 << 0): _mat_mul_into_tn(a, b, out); break;
        case (0 << 1 | 1 << 0): _mat_mul_into_nt(a, b, out); break;
        case (1 << 1 | 1 << 0): _mat_mul_into_tt(a, b, out); break;
        default: abort();
    }
}

static inline void mat_pretty_print(const Matrix* m, const char* optional_name) {
    printf("%s: [", (optional_name ? optional_name : "Matrix"));

    if (mat_is_empty(m)) {
        printf("EMPTY]\n");
        return;
    } else {
        printf("\n");
    }

    for (u32 i = 0; i < m->rows; ++i) {
        printf("    ");
        for (u32 j = 0; j < m->cols; ++j) {
            printf("%.2f ", mat_at(m, i, j));
        }
        printf("\n");
    }
    
    printf("]\n\n");
}

typedef enum {
    Linear, Biases, ActivationFunction,
} LayerType;

typedef struct {
    Matrix weights;
    Matrix grad_weights;
    Matrix cached_activation;
} LinearLayer;

typedef struct {
    Matrix biases;
    Matrix grad_biases;
    Matrix cached_activation;
} BiasesLayer;

typedef float(*Function)(float);

typedef struct {
    Function forward;
    Function deriavtive;
    Matrix cached_activation;
} ActivationLayer;

typedef struct {
    LayerType layer_type;

    union {
        LinearLayer linear;
        BiasesLayer biases;
        ActivationLayer activation;
    } data;

    bool requires_optimizer;
} Layer;

static inline Layer linear_new(u32 fan_in, u32 fan_out) {
    Layer layer;
    layer.layer_type = Linear;
    layer.requires_optimizer = true;

    layer.data.linear = (LinearLayer){
        .weights           = mat_new(fan_out, fan_in),
        .grad_weights      = mat_new(fan_out, fan_in),
        .cached_activation = mat_new(fan_in, 1),
    };

    return layer;
}

static inline void linear_free(LinearLayer* linear) {
    assert(linear);

    mat_free(&linear->weights);
    mat_free(&linear->grad_weights);
    mat_free(&linear->cached_activation);

    *linear = (LinearLayer){0};
}

static inline void linear_forward(LinearLayer* linear, Matrix* inout, bool train) {
    mat_clone_into(inout, &linear->cached_activation);
    mat_mul_into(&linear->weights, &linear->cached_activation, inout, false, false, true);
}

static inline Layer biases_new(u32 fan_in) {
    Layer layer;
    layer.layer_type = Biases;
    layer.requires_optimizer = true;

    layer.data.biases = (BiasesLayer){
        .biases            = mat_vec(fan_in),
        .grad_biases       = mat_vec(fan_in),
        .cached_activation = mat_vec(fan_in),
    };

    return layer;
}

static inline void biases_free(BiasesLayer* biases) {
    assert(biases);

    mat_free(&biases->biases);
    mat_free(&biases->grad_biases);
    mat_free(&biases->cached_activation);

    *biases = (BiasesLayer){0};
}

static inline void biases_forward(BiasesLayer* biases, Matrix* inout, bool train) {
    if (train) mat_clone_into(inout, &biases->cached_activation);
    mat_add_into(inout, &biases->biases, inout);
}

float relu(float x) {
    return MAX(x, 0.0);
}

float relu_derivative(float x) {
    return (x >= 0.0 ? 1.0 : 0.0);
}

static inline Layer activation_new(u32 fan_in, Function forward, Function derivative) {
    Layer layer;
    layer.layer_type = ActivationFunction;
    layer.requires_optimizer = false;

    layer.data.activation = (ActivationLayer){
        .forward           = forward,
        .deriavtive        = derivative,
        .cached_activation = mat_vec(fan_in),
    };

    return layer;
}

static inline void activation_free(ActivationLayer* activation) {
    assert(activation);
    mat_free(&activation->cached_activation);
    *activation = (ActivationLayer){0};
}

static inline void activation_forward(ActivationLayer* activation, Matrix* inout, bool train) {
    if (train) mat_clone_into(inout, &activation->cached_activation);

    for (i32 i = 0; i < mat_elem_count(inout); ++i)
        inout->values[i] = activation->forward(inout->values[i]);
}

static inline void layer_free(Layer* layer) {
    assert(layer);
    switch (layer->layer_type) {
        case Linear: linear_free(&layer->data.linear); break;
        case Biases: biases_free(&layer->data.biases); break;
        case ActivationFunction: activation_free(&layer->data.activation); break;
        default: abort();
    }
}

static inline void layer_forward(Layer* layer, Matrix* inout /*activation*/, bool train) {
    switch (layer->layer_type) {
        case Linear: {linear_forward(&layer->data.linear, inout, train);} break;
        case Biases: {biases_forward(&layer->data.biases, inout, train);} break;
        case ActivationFunction: {activation_forward(&layer->data.activation, inout, train);} break;
        default: abort();
    }
}

static inline void layer_backward(Layer* layer, Matrix* inout /*gradient*/) {
    abort();
}

typedef struct {
    Matrix* param;
    Matrix* gradient;
} ParamGrad;

static inline ParamGrad layer_get_param_gradient(Layer* layer) {
    switch (layer->layer_type) {
        case Linear: return (ParamGrad){&layer->data.linear.weights, &layer->data.linear.grad_weights}; break;
        case Biases: return (ParamGrad){&layer->data.biases.biases,  &layer->data.biases.grad_biases}; break;
        default: abort();
    }
}

typedef struct {
    u32 fan_in;
    u32 fan_out;
} LayerInfo;

static inline bool info_requires_optimizer(const LayerInfo* info) {
    assert((info->fan_in > 0 && info->fan_out > 0) || (info->fan_in == 0 && info->fan_out == 0));
    return (info->fan_in > 0 && info->fan_out > 0);
}

static inline LayerInfo layer_get_shape(const Layer* layer) {
    assert(layer);
    switch (layer->layer_type) {
        case Linear: return (LayerInfo){.fan_in = layer->data.linear.weights.cols, .fan_out = layer->data.linear.weights.rows}; break;
        case Biases: return (LayerInfo){.fan_in = layer->data.biases.biases .cols, .fan_out = layer->data.biases.biases .rows}; break;
        case ActivationFunction: return (LayerInfo){0}; break;
        default: abort();
    }
}

typedef struct {
    Matrix momentum;
    Matrix varaince;
} AdamW;

static inline AdamW adam_w_new(const LayerInfo layer_info) {
    if (info_requires_optimizer(&layer_info))
        return (AdamW){
            .momentum = mat_new(layer_info.fan_out, layer_info.fan_in),
            .varaince = mat_new(layer_info.fan_out, layer_info.fan_in),
        };
    else
        return (AdamW){
            .momentum = mat_empty(),
            .varaince = mat_empty(),
        };
}

static inline void adam_w_free(AdamW* adam_w) {
    assert(adam_w);

    if (!mat_is_empty(&adam_w->momentum)) mat_free(&adam_w->momentum);
    if (!mat_is_empty(&adam_w->varaince)) mat_free(&adam_w->varaince);

    *adam_w = (AdamW){0};
}

static inline void adam_w_update(AdamW* adam_w, Matrix* param, Matrix* gradient, float betta_1, float betta_2, float lambda, float lr, u32 t) {
    assert(adam_w->momentum.rows == gradient->rows);
    assert(adam_w->momentum.cols == gradient->cols);

    const float betta_1_t = 1.0 - powf(betta_1, t);
    const float betta_2_t = 1.0 - powf(betta_2, t);

    for (u32 i = 0; i < mat_elem_count(gradient); ++i) {
        float  g = gradient->values[i];
        float* m = &adam_w->momentum.values[i];
        float* v = &adam_w->varaince.values[i];
        float* w = &param->values[i];

        *m = betta_1 * *m + (1.0 - betta_1) * g;
        *v = betta_2 * *v + (1.0 - betta_2) * g * g;

        float m_hat = *m / betta_1_t;
        float v_hat = *v / betta_2_t;

        *w -= lr * (m_hat / (sqrtf(v_hat) + 1e-5f) + lambda * *w);
    }
}

typedef struct {
    u32 layer_count;
    float lr;
    float betta_1;
    float betta_2;
    float lambda;
    u32 mini_batch;
    u32 t;
    
    Layer* layers;
    AdamW* adam_w;
} MLP;

static inline MLP mlp_new(void) {
    return (MLP){
        .lr         = 0.005,
        .betta_1    = 0.9,
        .betta_2    = 0.999,
        .lambda     = 0.001,
        .mini_batch = 4,
        .t          = 1,
    };
}

static inline void mlp_free(MLP* mlp) {
    assert(mlp);

    for (u32 i = 0; i < mlp->layer_count; ++i) {
        layer_free(&mlp->layers[i]);
        adam_w_free(&mlp->adam_w[i]);
    }

    if (mlp->layers) free(mlp->layers);
    if (mlp->adam_w) free(mlp->adam_w);

    *mlp = (MLP){0};
}

static inline void mlp_push_layer(MLP* mlp, Layer layer) {
    assert(mlp);
    mlp->layer_count++;

    mlp->layers = realloc(mlp->layers, mlp->layer_count * sizeof(Layer));
    mlp->adam_w = realloc(mlp->adam_w, mlp->layer_count * sizeof(AdamW));
    
    mlp->layers[mlp->layer_count - 1] = layer;
    mlp->adam_w[mlp->layer_count - 1] = adam_w_new(layer_get_shape(&layer));
}

static inline void mlp_forward(MLP* mlp, Matrix* inout) {
    assert(!mat_is_empty(inout));

    for (u32 i = 0; i < mlp->layer_count; ++i)
        layer_forward(&mlp->layers[i], inout, false);
}

static inline void _mlp_apply_gradients(MLP* mlp) {
    assert(mlp);

    for (u32 i = 0; i < mlp->layer_count; ++i) {
        if (!mlp->layers[i].requires_optimizer)
            continue;
        
        ParamGrad param_grad = layer_get_param_gradient(&mlp->layers[i]);
        adam_w_update(&mlp->adam_w[i], param_grad.param, param_grad.gradient, mlp->betta_1, mlp->betta_2, mlp->lambda, mlp->lr, mlp->t);
    }
}

static inline void mlp_train(MLP* mlp, const Matrix* input, const Matrix* desired_output) {
    assert(mlp);
    
    Matrix activation = mat_clone(input);

    for (u32 i = 0; i < mlp->layer_count; ++i)
        layer_forward(&mlp->layers[i], &activation, true);

    mat_sub_into(&activation, desired_output, &activation); // MSE derivative

    for (i32 i = mlp->layer_count - 1; i >= 0; --i)
        layer_backward(&mlp->layers[i], &activation);

    if ((mlp->t % mlp->mini_batch) == 0) {
        _mlp_apply_gradients(mlp);
    }

    mlp->t++;
}

#ifdef __cplusplus
}
#endif

#endif // MLP_H


// ToDo:
// - cosine lr decay