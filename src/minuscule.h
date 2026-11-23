#include <stdint.h>
#include <stdlib.h>
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
uint32_t mat_elem_count(Matrix* m);

void mat_add_into(Matrix const* a, Matrix const* b, Matrix* out);
void mat_sub_into(Matrix const* a, Matrix const* b, Matrix* out);
void mat_mulf(Matrix* m, float v);

void mat_mul_into(Matrix* const a, Matrix* const b, Matrix* out, bool transpose_a, bool transpose_b, bool zero);

typedef struct {
    Matrix momentum;
    Matrix variance;
} AdamW;

typedef struct {
    Matrix param;
    Matrix gradient;
} Parameter;

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

    union {
        DenseLayer dense;
        ActivationLayer activation;
    } data;
} Layer;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MINUS_H


// --- Implementation --


#ifdef MINUS_IMPLEMENTATION

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // MINUS_IMPLEMENTATION