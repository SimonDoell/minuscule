#include <stdio.h>
#include "minuscule.h"

int main(void) {
    MLP mlp = mlp_new();

    mlp_push_layer(&mlp, linear_new(1, 64));
    mlp_push_layer(&mlp, activation_new(64, relu, relu_derivative));
    mlp_push_layer(&mlp, linear_new(64, 64));
    mlp_push_layer(&mlp, activation_new(64, relu, relu_derivative));
    mlp_push_layer(&mlp, linear_new(64, 1));

    mlp_free(&mlp);
    return 0;
}