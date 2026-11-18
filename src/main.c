#include <time.h>
#include <stdio.h>
#include "minuscule.h"


int main(void) {
    PPO ppo = ppo_new();

    // State: 4 values:
    // - 0: agent  x
    // - 1: agent  y
    // - 2: target x
    // - 3: target y
    
    // Action: 4 values:
    // - 0: probability move up
    // - 1: probability move down
    // - 2: probability move left
    // - 3: probability move right
    
    // Environment: 5 values
    // - agent x, y
    // - target x, y
    // - size of the world as an integer

    // State -> Action
    mlp_push_layer(&ppo.actor, linear_new(4, 64, Xavier));
    mlp_push_layer(&ppo.actor, biases_new(64));
    mlp_push_layer(&ppo.actor, activation_new(64, leaky_relu, leaky_relu_derivative));
    mlp_push_layer(&ppo.actor, linear_new(64, 64, Xavier));
    mlp_push_layer(&ppo.actor, biases_new(64));
    mlp_push_layer(&ppo.actor, activation_new(64, leaky_relu, leaky_relu_derivative));
    mlp_push_layer(&ppo.actor, linear_new(64, 4, Xavier));
    mlp_push_layer(&ppo.actor, softmax_new(4, 1.0f));

    // State -> expected future Reward
    mlp_push_layer(&ppo.critic, linear_new(4, 64, Xavier));
    mlp_push_layer(&ppo.critic, biases_new(64));
    mlp_push_layer(&ppo.critic, activation_new(64, leaky_relu, leaky_relu_derivative));
    mlp_push_layer(&ppo.critic, linear_new(64, 64, Xavier));
    mlp_push_layer(&ppo.critic, biases_new(64));
    mlp_push_layer(&ppo.critic, activation_new(64, leaky_relu, leaky_relu_derivative));
    mlp_push_layer(&ppo.critic, linear_new(64, 1, Xavier));

    Matrix input  = mat_vec(4);
    Matrix output = mat_vec(4);

    mlp_train(&ppo.actor, &input, &output);

    mat_free(&input);
    mat_free(&output);
    ppo_free(&ppo);
    return 0;
}

// float learn(float x) {
//     return sinf(x * 2.0f * 3.14159265358979f);
// }

// int main(void) {
//     srand(time(0));
    
//     MLP mlp = mlp_new();

//     mlp_push_layer(&mlp, linear_new(1, 64, Xavier));
//     mlp_push_layer(&mlp, biases_new(64));
//     mlp_push_layer(&mlp, activation_new(64, leaky_relu, leaky_relu_derivative));

//     mlp_push_layer(&mlp, linear_new(64, 64, Xavier));
//     mlp_push_layer(&mlp, biases_new(64));
//     mlp_push_layer(&mlp, activation_new(64, leaky_relu, leaky_relu_derivative));

//     mlp_push_layer(&mlp, linear_new(64, 64, Xavier));
//     mlp_push_layer(&mlp, biases_new(64));
//     mlp_push_layer(&mlp, activation_new(64, leaky_relu, leaky_relu_derivative));

//     mlp_push_layer(&mlp, linear_new(64, 64, Xavier));
//     mlp_push_layer(&mlp, biases_new(64));
//     mlp_push_layer(&mlp, activation_new(64, leaky_relu, leaky_relu_derivative));

//     mlp_push_layer(&mlp, linear_new(64, 1, Xavier));

//     Matrix input  = mat_new(1, 1);
//     Matrix output = mat_new(1, 1);

//     for (u32 i = 0; i < 100000; ++i) {
//         input .values[0] = rand_float(0, 1);
//         output.values[0] = learn(input.values[0]);
//         mlp_train(&mlp, &input, &output);

//         if (i % 100 == 0) {
//             float err = 0.0f;
            
//             for (u32 t = 0; t < 100; ++t) {
//                 mat_ensure_shape(&input, 1, 1);
//                 input .values[0] = rand_float(0, 1);
//                 output.values[0] = learn(input.values[0]);

//                 mlp_forward(&mlp, &input);

//                 err += fabsf(input.values[0] - output.values[0]);
//             }

//             err /= (float)100;

//             printf("Err: %f\n", err);
//         }
//     }

//     mat_free(&output);
//     mat_free(&input);
//     mlp_free(&mlp);
//     return 0;
// }