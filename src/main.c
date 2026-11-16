#include <stdio.h>
#include "minuscule.h"

int main(void) {
    MLP mlp = mlp_new();

    mlp_free(&mlp);
    return 0;
}