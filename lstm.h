#ifndef mj_lstm_h
#define mj_lstm_h

#include <stddef.h>
#include <immintrin.h>
#include <fmaintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double real;
typedef __m256d realv;

typedef struct slstm
{
    size_t num_inputs;
    size_t num_outputs;
    size_t num_hidden_layers;
    size_t* num_hiddens;

    size_t max_state_size;
    size_t max_activation_state_size;

#define MEMBER(name) union { realv* v; real* i; } name;
    MEMBER(memory_cells);
    MEMBER(weights_input);

    MEMBER(weights_input_gate);
    MEMBER(weights_forget_gate);
    MEMBER(weights_output_gate);

    MEMBER(feedback_activations);
    MEMBER(weights_feedback_input);
    MEMBER(weights_feedback_input_gate);
    MEMBER(weights_feedback_forget_gate);
    MEMBER(weights_feedback_output_gate);

    MEMBER(weights_last_layer);

    MEMBER(bias_input);
    MEMBER(bias_input_gate);
    MEMBER(bias_forget_gate);
    MEMBER(bias_output_gate);

    MEMBER(weights_memory_to_output_gate);
    MEMBER(weights_memory_to_input_gate);
    MEMBER(weights_memory_to_forget_gate);
} lstm;

/* Check README.md for documentation on these functions. */

void propagate(lstm* restrict lstm, const real* restrict input, real* restrict output);

lstm* allocate_lstm( size_t num_inputs, size_t num_outputs, const size_t* num_hiddens, size_t num_hidden_layers );
void free_lstm( lstm* lstm );
void reset_lstm( lstm* lstm );

size_t serialized_lstm_size( const lstm* lstm );
void serialize_lstm( const lstm* lstm, real* serialized );
void unserialize_lstm( lstm* lstm, const real* serialized );

#ifdef __cplusplus
}
#endif

#endif

