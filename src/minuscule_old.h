#include <math.h>
#include <float.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifndef countof
#define countof(arr) (sizeof(arr) / sizeof(arr[0]))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef CLAMP
#define CLAMP(x, lo, hi) (MIN((hi), MAX((lo), (x))))
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

static inline void mat_move_into(Matrix* src, Matrix* dest) {
    if (!mat_is_empty(dest)) mat_free(dest);

    *dest = *src;
    *src  = mat_empty();
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
    for (i32 k = 0; k < b->cols; ++k) {
        for (i32 j = 0; j < a->cols; ++j) {
            for (i32 i = 0; i < a->rows; ++i) {
                *mat_at_mut(out, i, k) +=
                    mat_at(a, i, j) *
                    mat_at(b, j, k);
            }
        }
    }
}

static inline void _mat_mul_into_tn(const Matrix* restrict a, const Matrix* restrict b, Matrix* restrict out) {
    for (i32 i = 0; i < a->cols; ++i) {
        for (i32 k = 0; k < b->cols; ++k) {
            for (i32 j = 0; j < a->rows; ++j) {
                *mat_at_mut(out, i, k) +=
                    mat_at(a, j, i) *
                    mat_at(b, j, k);
            }
        }
    }
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

static inline void mat_mulf(Matrix* m, float f) {
    for (u32 i = 0; i < mat_elem_count(m); ++i)
        m->values[i] *= f;
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
    Linear, Biases, ActivationFunction, Softmax,
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
    Matrix cached_activation;
    float temp;
} SoftmaxLayer;

typedef struct {
    LayerType layer_type;

    union {
        LinearLayer      linear;
        BiasesLayer      biases;
        ActivationLayer  activation;
        SoftmaxLayer     softmax;
    } data;

    bool requires_optimizer;
} Layer;

typedef enum {
    Xavier, Kaiming
} InitOptions;

static inline float rand_float(float min, float max) {
    return (float)rand() / (float)RAND_MAX * (max - min) + min;
}

static inline Layer linear_new(u32 fan_in, u32 fan_out, InitOptions init_options) {
    Layer layer;
    layer.layer_type = Linear;
    layer.requires_optimizer = true;

    layer.data.linear = (LinearLayer){
        .weights           = mat_new(fan_out, fan_in),
        .grad_weights      = mat_new(fan_out, fan_in),
        .cached_activation = mat_new(fan_in, 1),
    };

    switch (init_options) {
        case Xavier: {
            float range = sqrtf(6.0 / (fan_in + fan_out));
            for (u32 i = 0; i < mat_elem_count(&layer.data.linear.weights); ++i)
                layer.data.linear.weights.values[i] = rand_float(-range, range);
        } break;
        case Kaiming: {
            float range = sqrtf(2.0 / fan_in);
            for (u32 i = 0; i < mat_elem_count(&layer.data.linear.weights); ++i)
                layer.data.linear.weights.values[i] = rand_float(0, range);
        } break;
        default: abort();
    }

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

static inline void linear_backward(LinearLayer* linear, Matrix* inout) {
    assert(linear->weights.rows == inout->rows);
    
    Matrix prev_grad = mat_empty();
    mat_mul_into(&linear->weights, inout, &prev_grad, true, false, true);

    for (u32 j = 0; j < linear->weights.cols; ++j) {
        for (u32 i = 0; i < linear->weights.rows; ++i) {
            *mat_at_mut(&linear->grad_weights, i, j) +=
                mat_at(&linear->cached_activation, j, 0) *
                mat_at(inout, i, 0);
        }
    }
    
    mat_move_into(&prev_grad, inout);
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

static inline void biases_backward(BiasesLayer* biases, Matrix* inout) {
    assert(biases->biases.rows == inout->rows);
    assert(biases->biases.cols == inout->cols);

    mat_add_into(&biases->grad_biases, inout, &biases->grad_biases);
}

float relu(float x) {
    return MAX(x, 0.0);
}

float relu_derivative(float x) {
    return (x >= 0.0 ? 1.0 : 0.0);
}

float leaky_relu(float x) {
    return MAX(x, x * 0.1f);
}

float leaky_relu_derivative(float x) {
    return (x >= 0.0 ? 1.0 : 0.1);
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

static inline void activation_backward(ActivationLayer* activation, Matrix* inout) {
    assert(activation->cached_activation.rows == inout->rows);
    assert(activation->cached_activation.cols == inout->cols);

    for (u32 i = 0; i < mat_elem_count(inout); ++i)
        inout->values[i] *= activation->deriavtive(activation->cached_activation.values[i]);
}

static inline Layer softmax_new(u32 fan_in, float temp) {
    Layer layer;
    layer.layer_type = Softmax;
    layer.requires_optimizer = false;

    layer.data.softmax = (SoftmaxLayer){
        .temp = temp,
        .cached_activation = mat_vec(fan_in),
    };

    return layer;
}

static inline void softmax_free(SoftmaxLayer* softmax) {
    assert(soft);
    mat_free(&softmax->cached_activation);
    *softmax = (SoftmaxLayer){0};
}

static inline void softmax_forward(SoftmaxLayer* softmax, Matrix* inout, bool train) {
    assert(softmax->cached_activation.rows == inout->rows && inout->cols == 1);
    assert(softmax->temp > 0.0f);
    const u32 n = softmax->cached_activation.rows;
    if (train) mat_clone_into(inout, &softmax->cached_activation);

    float max = -FLT_MAX;

    for (u32 i = 0; i < n; ++i)
        if (inout->values[i] > max)
            max = inout->values[i];

    float exps[n];

    for (u32 i = 0; i < n; ++i)
        exps[i] = expf((inout->values[i] - max) / softmax->temp);

    float sum = 0.0f;
    for (u32 i = 0; i < n; ++i)
        sum += exps[i];

    for (u32 i = 0; i < n; ++i)
        inout->values[i] = exps[i] / sum;
}

static inline void softmax_backward(SoftmaxLayer* softmax, Matrix* inout) {
    const u32 n = softmax->cached_activation.rows;
    assert(inout->rows == n && inout->cols == 1);

    float dot = 0.0f;

    for (u32 i = 0; i < n; ++i)
        dot += inout->values[i] * softmax->cached_activation.values[i];

    for (u32 i = 0; i < n; ++i) {
        const float y = softmax->cached_activation.values[i];
        inout->values[i] = y * (inout->values[i] - dot) / softmax->temp;
    }
}

static inline void layer_free(Layer* layer) {
    assert(layer);
    switch (layer->layer_type) {
        case Linear: linear_free(&layer->data.linear); break;
        case Biases: biases_free(&layer->data.biases); break;
        case ActivationFunction: activation_free(&layer->data.activation); break;
        case Softmax: softmax_free(&layer->data.softmax); break;
        default: abort();
    }
}

static inline void layer_forward(Layer* layer, Matrix* inout /*activation*/, bool train) {
    switch (layer->layer_type) {
        case Linear: {linear_forward(&layer->data.linear, inout, train);} break;
        case Biases: {biases_forward(&layer->data.biases, inout, train);} break;
        case ActivationFunction: {activation_forward(&layer->data.activation, inout, train);} break;
        case Softmax: {softmax_forward(&layer->data.softmax, inout, train);} break;
        default: abort();
    }
}

static inline void layer_backward(Layer* layer, Matrix* inout /*gradient*/) {
    switch (layer->layer_type) {
        case Linear: {linear_backward(&layer->data.linear, inout);} break;
        case Biases: {biases_backward(&layer->data.biases, inout);} break;
        case ActivationFunction: {activation_backward(&layer->data.activation, inout);} break;
        case Softmax: {softmax_backward(&layer->data.softmax, inout);} break;
        default: abort();
    }
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
        case Softmax: return (LayerInfo){.fan_in = layer->data.softmax.cached_activation.rows, .fan_out = layer->data.softmax.cached_activation.rows}; break;
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
        mat_zero(param_grad.gradient);
    }
}

static inline void mlp_train(MLP* mlp, const Matrix* input, const Matrix* desired_output) {
    assert(mlp);
    
    Matrix activation = mat_clone(input);

    for (u32 i = 0; i < mlp->layer_count; ++i)
        layer_forward(&mlp->layers[i], &activation, true);

    mat_sub_into(&activation, desired_output, &activation); // MSE derivative
    mat_mulf(&activation, 1.0f / (float)mlp->mini_batch);

    for (i32 i = mlp->layer_count - 1; i >= 0; --i)
        layer_backward(&mlp->layers[i], &activation);

    mat_free(&activation);

    if ((mlp->t % mlp->mini_batch) == 0) {
        _mlp_apply_gradients(mlp);
    }

    mlp->t++;
}


// typedef struct {
//     u32 episode_steps;
//     u32 episode_batches;
    
//     MLP actor;
//     MLP critic;
// } PPO;

// static inline PPO ppo_new(void) {
//     return (PPO){
//         .episode_steps = 128,
        
//         .actor  = mlp_new(),
//         .critic = mlp_new(),
//     };
// }

// static inline void ppo_free(PPO* ppo) {
//     assert(ppo);
//     mlp_free(&ppo->actor);
//     mlp_free(&ppo->critic);
//     *ppo = (PPO){0};
// }


#ifdef __cplusplus
}
#endif

#endif // MLP_H


// ToDo:
// - cosine lr decay
// - add optional multithreading using macros (openMP!)
// - simd
// - better random numbers using pcg32
// - PPO implementation