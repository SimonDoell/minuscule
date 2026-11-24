#include <time.h>
#include <stdio.h>
#define MINUSCULE_IMPLEMENTATION
#include "minuscule.h"

int main(void) {
    srand((unsigned int)time(0));
    
    MLP mlp = mlp_new();
    mlp.learning_rate = exponential_learning_rate(0.010f, 0.0005f, 1.0f, 10000),
    mlp.betta_1       = 0.9f,
    mlp.betta_2       = 0.999f,
    mlp.lambda        = 0.001f,
    mlp.mini_batch    = 16,
    mlp.loss_function = LOSS_MSE;

    mlp_push_layer(&mlp, dense_new(1, 16, DENSE_INIT_XAVIER));
    mlp_push_layer(&mlp, layernorm_new(16));
    mlp_push_layer(&mlp, activation_new(16, tanhf, tanh_derivative));
    mlp_push_layer(&mlp, softmax_new(16, 1.0f));

    for (uint32_t i = 0; i < 15; ++i) {
        mlp_push_layer(&mlp, dense_new(16, 16, DENSE_INIT_XAVIER));
        mlp_push_layer(&mlp, layernorm_new(16));
        mlp_push_layer(&mlp, activation_new(16, tanhf, tanh_derivative));
        mlp_push_layer(&mlp, softmax_new(16, 1.0f));
    }

    mlp_push_layer(&mlp, dense_new(16, 1, DENSE_INIT_XAVIER));

    Matrix input  = mat_new(1, 1);
    Matrix output = mat_new(1, 1);

    for (uint32_t i = 0; i < 10000; ++i) {
        input.values [0] = 0.5f;
        output.values[0] = 0.314159265358979f;
        mlp_train(&mlp, &input, &output);

        mlp_forward(&mlp, &input);

        printf("Value: %f\n", (double)input.values[0]);
    }

    mat_free(&input);
    mat_free(&output);
    mlp_free(&mlp);
    return 0;
}