#include <time.h>
#include <stdio.h>
#define MINUS_IMPLEMENTATION
#include "minuscule.h"

static inline float relu(float x) {
    return (x > 0 ? x : 0);
}

static inline float relu_derivative(float x) {
    return (x > 0 ? 1 : 0);
}

int main(void) {
    srand(time(0));
    
    MLP mlp = mlp_new();

    mlp_push_layer(&mlp, dense_new(1, 16, DENSE_INIT_KAIMING));
    mlp_push_layer(&mlp, activation_new(16, relu, relu_derivative));
    mlp_push_layer(&mlp, layernorm_new(16));

    for (uint32_t i = 0; i < 5; ++i) {
        mlp_push_layer(&mlp, dense_new(16, 16, DENSE_INIT_KAIMING));
        mlp_push_layer(&mlp, activation_new(16, relu, relu_derivative));
        mlp_push_layer(&mlp, layernorm_new(16));
    }

    mlp_push_layer(&mlp, dense_new(16, 1, DENSE_INIT_KAIMING));

    Matrix input  = mat_new(1, 1);
    Matrix output = mat_new(1, 1);

    for (uint32_t i = 0; i < 1000; ++i) {
        input.values [0] = 0.5f;
        output.values[0] = 0.314159265358979f;
        mlp_train(&mlp, &input, &output);

        mlp_forward(&mlp, &input);

        printf("Value: %.15f\n", input.values[0]);
    }

    mat_free(&input);
    mat_free(&output);
    mlp_free(&mlp);
    return 0;
}



// #include <time.h>
// #include <stdio.h>
// #include "minuscule.h"

// // State: 4 values:
// // - 0: agent  x
// // - 1: agent  y
// // - 2: target x
// // - 3: target y

// // Action: 4 values:
// // - 0: probability move up
// // - 1: probability move down
// // - 2: probability move left
// // - 3: probability move right

// // Environment: 5 values
// // - agent x, y
// // - target x, y
// // - size of the world as an integer


// typedef struct {
//     u32 agent_x;
//     u32 agent_y;
//     u32 target_x;
//     u32 target_y;
// } State;

// typedef enum {
//     Up, Down, Left, Right,
// } Action;

// typedef struct {
//     u32 agent_x;
//     u32 agent_y;
//     u32 target_x;
//     u32 target_y;

//     u32 size;
// } Env;

// static inline Matrix state_to_input(Env* env, State* state) {
//     Matrix input = mat_vec(4);

//     input.values[0] = (float)state->agent_x  / (float)env->size * 2.0f - 1.0f;
//     input.values[1] = (float)state->agent_y  / (float)env->size * 2.0f - 1.0f;
//     input.values[2] = (float)state->target_x / (float)env->size * 2.0f - 1.0f;
//     input.values[3] = (float)state->target_y / (float)env->size * 2.0f - 1.0f;

//     return input;
// }

// static inline Env env_new(void) {
//     u32 size = 25;

//     return (Env){
//         .agent_x  = rand() % size,
//         .agent_y  = rand() % size,
//         .target_x = rand() % size,
//         .target_y = rand() % size,
//         .size     = size,
//     };
// }

// static inline void env_reset(Env* env) {
//     *env = env_new();
// }

// static inline void env_do_action(Env* env, Action action) {
//     i32 agent_x  = env->agent_x;
//     i32 agent_y  = env->agent_y;
    
//     switch (action) {
//         case Up:    agent_y++; break;
//         case Down:  agent_y--; break;
//         case Right: agent_x++; break;
//         case Left:  agent_x--; break;
//         default: abort();
//     }

//     agent_x = CLAMP(agent_x, 0, env->size - 1);
//     agent_y = CLAMP(agent_y, 0, env->size - 1);

//     env->agent_x = agent_x;
//     env->agent_y = agent_y;
// }

// static inline bool env_terminate(Env* env) {
//     return (env->agent_x == env->target_x) && (env->agent_y == env->target_y);
// }

// static inline State env_get_state(Env* env) {
//     return (State){
//         .agent_x  = env->agent_x,
//         .agent_y  = env->agent_y,
//         .target_x = env->target_x,
//         .target_y = env->target_y,
//     };
// }

// static inline float get_reward(Env* env, Action action) {
//     i32 agent_x  = env->agent_x;
//     i32 agent_y  = env->agent_y;
//     i32 target_x = env->target_x;
//     i32 target_y = env->target_y;
    
//     float prev_delta_x = ((float)agent_x - (float)target_x) / (float)env->size;
//     float prev_delta_y = ((float)agent_y - (float)target_y) / (float)env->size;
//     float prev_dist = sqrtf(prev_delta_x * prev_delta_x + prev_delta_y * prev_delta_y);

//     switch (action) {
//         case Up:    agent_y++; break;
//         case Down:  agent_y--; break;
//         case Right: agent_x++; break;
//         case Left:  agent_x--; break;
//         default: abort();
//     }

//     agent_x = CLAMP(agent_x, 0, env->size - 1);
//     agent_y = CLAMP(agent_y, 0, env->size - 1);

//     float new_delta_x = ((float)agent_x - (float)target_x) / (float)env->size;
//     float new_delta_y = ((float)agent_y - (float)target_y) / (float)env->size;
//     float new_dist = sqrtf(new_delta_x * new_delta_x + new_delta_y * new_delta_y);

//     bool terminate = (agent_x == target_x && agent_y == target_y);
    
//     return -new_dist + (terminate ? 10.0f : -0.1f);
// }

// typedef struct {
//     State state;        // the state from which the action was chosen
//     Action action;      // the chosen action from the state
//     float log_action;   // the log of the probability for the chosen action from actor output
//     float reward;       // the immediate reward from the reward function
//     float value;        // output of the critic network
//     float reward_to_go;
//     float advantage;
//     bool is_terminal;
// } PPOStep;

// static inline float sumf(float* values, u32 count) {
//     float sum = 0.0f;
//     for (u32 i = 0; i < count; ++i)
//         sum += values[i];
//     return sum;
// }

// static inline void ppo_train_episode(MLP* actor, MLP* critic) {
//     const float discount   = 0.9f;
//     const float lambda     = 0.9f;
//     const float epsilon    = 0.2f;
//     const u32 episode_size = 128;

//     u32 successfull_steps;
//     PPOStep steps[episode_size];

//     Env env = env_new();

//     // Collect steps
//     for (u32 s = 0; s < episode_size; ++s) {
//         PPOStep* step = &steps[s];
//         step->is_terminal = false;

//         State state = env_get_state(&env);

//         Matrix actor_input = state_to_input(&env, &state);
//         mlp_forward(actor, &actor_input);

//         assert(fabsf(sumf(actor_input.values, 4) - 1.0f) < 1e-5f);

//         float p = rand_float(0.0f, 1.0f);
        
//         u32 action_index = 0;
//         float cumulative = 0.0f;
        
//         for (u32 i = 0; i < 4; ++i) {
//             cumulative += actor_input.values[i];
        
//             if (p < cumulative || i == 3) {
//                 action_index = i;
//                 break;
//             }
//         }
        
//         Action action = (Action)action_index;
        
//         float reward = get_reward(&env, action);
        
//         env_do_action(&env, action);

//         step->action = action;
//         step->log_action = logf(actor_input.values[action_index]);
//         step->reward = reward;
//         step->state = state;

//         Matrix critic_input = state_to_input(&env, &state);
//         mlp_forward(critic, &critic_input);
//         step->value = critic_input.values[0];

//         successfull_steps = s;
        
//         mat_free(&actor_input);
//         mat_free(&critic_input);
        
//         if (env_terminate(&env)) {
//             step->is_terminal = true;
//             break;
//         }
//     }

//     float next_value = 0.0f;
    
//     if (!steps[successfull_steps].is_terminal) {
//         State next_state = env_get_state(&env);
//         Matrix input = state_to_input(&env, &next_state);
    
//         mlp_forward(critic, &input);
//         next_value = input.values[0];
    
//         mat_free(&input);
//     }

//     for (i32 s = successfull_steps; s >= 0; --s) {
//         if (steps[s].is_terminal)
//         {
//             steps[s].reward_to_go = steps[s].reward;
//             steps[s].advantage    = steps[s].reward - steps[s].value;
//         }
//         else if (s == successfull_steps)
//         {
//             steps[s].reward_to_go = steps[s].reward + discount * next_value;
//             steps[s].advantage    = steps[s].reward + discount * next_value - steps[s].value;
//         }
//         else
//         {
//             steps[s].reward_to_go = steps[s].reward + discount * steps[s + 1].reward_to_go;
//             float delta = steps[s].reward + discount * steps[s + 1].value - steps[s].value;
//             steps[s].advantage = delta + discount * lambda * steps[s + 1].advantage;
//         }
//     }

//     Matrix output = mat_empty();
    
//     // Optimize the critic
//     for (u32 s = 0; s <= successfull_steps; ++s) {
//         mat_ensure_shape(&output, 1, 1);

//         Matrix input = state_to_input(&env, &steps[s].state);
//         output.values[0] = steps[s].reward_to_go;

//         mlp_train(critic, &input, &output);
//         mat_free(&input);
//     }

//     mat_free(&output);

//     // Optimize the actor
// }

// int main(void) {
//     srand(time(0));
    
//     MLP actor  = mlp_new();
//     MLP critic = mlp_new();

//     // Actor: State -> Action
//     mlp_push_layer(&actor, linear_new(4, 64, Xavier));
//     mlp_push_layer(&actor, biases_new(64));
//     mlp_push_layer(&actor, activation_new(64, leaky_relu, leaky_relu_derivative));
//     mlp_push_layer(&actor, linear_new(64, 64, Xavier));
//     mlp_push_layer(&actor, biases_new(64));
//     mlp_push_layer(&actor, activation_new(64, leaky_relu, leaky_relu_derivative));
//     mlp_push_layer(&actor, linear_new(64, 4, Xavier));
//     mlp_push_layer(&actor, softmax_new(4, 1.0f));

//     // Critic: State -> Value (expected future Reward)
//     mlp_push_layer(&critic, linear_new(4, 64, Xavier));
//     mlp_push_layer(&critic, biases_new(64));
//     mlp_push_layer(&critic, activation_new(64, leaky_relu, leaky_relu_derivative));
//     mlp_push_layer(&critic, linear_new(64, 64, Xavier));
//     mlp_push_layer(&critic, biases_new(64));
//     mlp_push_layer(&critic, activation_new(64, leaky_relu, leaky_relu_derivative));
//     mlp_push_layer(&critic, linear_new(64, 1, Xavier));

//     ppo_train_episode(&actor, &critic);

//     mlp_free(&actor);
//     mlp_free(&critic);
//     return 0;
// }

// // float learn(float x) {
// //     return sinf(x * 2.0f * 3.14159265358979f);
// // }

// // int main(void) {
// //     srand(time(0));
    
// //     MLP mlp = mlp_new();

// //     mlp_push_layer(&mlp, linear_new(1, 64, Xavier));
// //     mlp_push_layer(&mlp, biases_new(64));
// //     mlp_push_layer(&mlp, activation_new(64, leaky_relu, leaky_relu_derivative));

// //     mlp_push_layer(&mlp, linear_new(64, 64, Xavier));
// //     mlp_push_layer(&mlp, biases_new(64));
// //     mlp_push_layer(&mlp, activation_new(64, leaky_relu, leaky_relu_derivative));

// //     mlp_push_layer(&mlp, linear_new(64, 64, Xavier));
// //     mlp_push_layer(&mlp, biases_new(64));
// //     mlp_push_layer(&mlp, activation_new(64, leaky_relu, leaky_relu_derivative));

// //     mlp_push_layer(&mlp, linear_new(64, 64, Xavier));
// //     mlp_push_layer(&mlp, biases_new(64));
// //     mlp_push_layer(&mlp, activation_new(64, leaky_relu, leaky_relu_derivative));

// //     mlp_push_layer(&mlp, linear_new(64, 1, Xavier));

// //     Matrix input  = mat_new(1, 1);
// //     Matrix output = mat_new(1, 1);

// //     for (u32 i = 0; i < 100000; ++i) {
// //         input .values[0] = rand_float(0, 1);
// //         output.values[0] = learn(input.values[0]);
// //         mlp_train(&mlp, &input, &output);

// //         if (i % 100 == 0) {
// //             float err = 0.0f;
            
// //             for (u32 t = 0; t < 100; ++t) {
// //                 mat_ensure_shape(&input, 1, 1);
// //                 input .values[0] = rand_float(0, 1);
// //                 output.values[0] = learn(input.values[0]);

// //                 mlp_forward(&mlp, &input);

// //                 err += fabsf(input.values[0] - output.values[0]);
// //             }

// //             err /= (float)100;

// //             printf("Err: %f\n", err);
// //         }
// //     }

// //     mat_free(&output);
// //     mat_free(&input);
// //     mlp_free(&mlp);
// //     return 0;
// // }