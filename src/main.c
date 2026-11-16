#include <stdio.h>
#include "mlp.h"

int main(void) {
    MLP mlp = mlp_new();

    mlp_free(&mlp);
    return 0;
}