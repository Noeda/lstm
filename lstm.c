#include "lstm.h"
#include <alloca.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include "exp_avx.h"

#ifdef TESTING
#define TEST(x) x
#else
#define TEST(x)
#endif

int is_valid(const lstm* lstm)
{
    if ( lstm->num_hidden_layers <= 0 ) {
        return 0;
    }
    return 1;
}

static real sigmoid(const real x)
{
    return 1.0 / (1.0 + exp(-x));
}

static void zero_layer( const size_t num_targets
                      , real* restrict target )
{
    for ( size_t i1 = 0; i1 < num_targets; ++i1 ) {
        target[i1] = 0.0;
    }
}

static void zero_layer_avx( const size_t num_targets
                          , __m256d* restrict target )
{
    size_t nt = num_targets/4;
    for ( size_t i1 = 0; i1 < nt; ++i1 ) {
        target[i1] = _mm256_setzero_pd();
    }
}

static void propagate_layer( const size_t num_targets
                           , const size_t num_sources
                           , real* restrict target
                           , const real* restrict source
                           , const real* restrict weights )
{
    for ( size_t i1 = 0; i1 < num_targets; ++i1 ) {
        real* t = &target[i1];
        for ( size_t i2 = 0; i2 < num_sources; ++i2 ) {
            (*t) += weights[i2+i1*num_sources]*source[i2];
        }
    }
}

static void propagate_layer_avx( const size_t num_targets
                               , const size_t num_sources
                               , real* restrict target
                               , const __m256d* restrict source
                               , const __m256d* restrict weights )
{
    int ns = num_sources/4;
    for ( size_t i1 = 0; i1 < num_targets; ++i1 ) {
        real* t = &target[i1];
        for ( size_t i2 = 0; i2 < ns; ++i2 ) {
            size_t off = i2+i1*ns;
            __m256d s = source[i2];
            __m256d w = weights[off];
            __m256d addition = _mm256_mul_pd(s, w);
            (*t) += addition[0];
            (*t) += addition[1];
            (*t) += addition[2];
            (*t) += addition[3];
        }
    }
}

static void memory_cell_layer( const size_t num_items
                             , real* memory_target
                             , real* activation_target
                             , const real* input
                             , const real* input_gate
                             , const real* output_gate
                             , const real* forget_gate )
{
    for ( size_t i1 = 0; i1 < num_items; ++i1 ) {
        memory_target[i1] = memory_target[i1]*forget_gate[i1] +
                            input_gate[i1]*input[i1];
        activation_target[i1] = output_gate[i1]*memory_target[i1];
    }
}

static void memory_cell_layer_avx( const size_t num_items
                                 , __m256d* memory_target
                                 , __m256d* activation_target
                                 , const __m256d* input
                                 , const __m256d* input_gate
                                 , const __m256d* output_gate
                                 , const __m256d* forget_gate )
{
    int ni = num_items/(sizeof(__m256d)/sizeof(real));
    for ( size_t i1 = 0; i1 < ni; ++i1 ) {
        __m256d ii = _mm256_mul_pd(input_gate[i1], input[i1]);
        memory_target[i1] = _mm256_fmadd_pd( memory_target[i1], forget_gate[i1]
                                           , ii );
        activation_target[i1] = _mm256_mul_pd(output_gate[i1], memory_target[i1]);
    }
}

static void propagate_11_layer( const size_t num_items
                              , real* target
                              , const real* source
                              , const real* weights )
{
    for ( size_t i1 = 0; i1 < num_items; ++i1 ) {
        target[i1] += weights[i1]*source[i1];
    }
}

static void propagate_11_layer_avx( const size_t num_items
                                  , __m256d* target
                                  , const __m256d* source
                                  , const __m256d* weights )
{
    const size_t nt = num_items / (sizeof(__m256d)/sizeof(real));
    for ( size_t i1 = 0; i1 < nt; ++i1 ) {
        target[i1] = _mm256_fmadd_pd(weights[i1], source[i1], target[i1]);
    }
}

static void sigmoid_layer( const size_t num_items
                         , real* target )
{
    for ( size_t i1 = 0; i1 < num_items; ++i1 ) {
        target[i1] = sigmoid(target[i1]);
    }
}

static void sigmoid_layer_avx( const size_t num_items
                             , __m256d* target )
{
    __m256d negative = { -1.0, -1.0, -1.0, -1.0 };
    __m256d plus = { 1.0, 1.0, 1.0, 1.0 };
    size_t nt = num_items/(sizeof(__m256d)/sizeof(real));
    for ( size_t i1 = 0; i1 < nt; ++i1 ) {
        target[i1] = _mm256_div_pd(plus, _mm256_add_pd(plus, gmx_mm256_exp_pd(_mm256_mul_pd(target[i1], negative))));
    }
}

static void tanh_layer( const size_t num_items
                      , real* target )
{
    for ( size_t i1 = 0; i1 < num_items; ++i1 ) {
        target[i1] = sigmoid(target[i1])*2-1;
    }
}

static void tanh_layer_avx( const size_t num_items
                          , __m256d* target )
{
    __m256d negative = { -1.0, -1.0, -1.0, -1.0 };
    __m256d plus = { 1.0, 1.0, 1.0, 1.0 };
    __m256d two = { 2.0, 2.0, 2.0, 2.0 };
    int nt = num_items/(sizeof(__m256d)/sizeof(real));
    for ( size_t i1 = 0; i1 < nt; ++i1 ) {
        target[i1] = _mm256_fmsub_pd(_mm256_div_pd(plus, _mm256_add_pd(plus, gmx_mm256_exp_pd(_mm256_mul_pd(target[i1], negative)))), two, plus);
    }
}

static void bias_layer( const size_t num_items
                      , real* target
                      , const real* source )
{
    for ( size_t i1 = 0; i1 < num_items; ++i1 ) {
        target[i1] += source[i1];
    }
}

static void bias_layer_avx( const size_t num_items
                          , __m256d* target
                          , const __m256d* source )
{
    size_t nt = num_items/(sizeof(__m256d)/sizeof(real));
    for ( size_t i1 = 0; i1 < nt; ++i1 ) {
        target[i1] = _mm256_add_pd(target[i1], source[i1]);
    }
}

#define ALIGN32(ptr) \
{ intptr_t ip = (intptr_t) ptr; \
  intptr_t aligned = ip+(32-ip%32); \
  ptr = (real*) aligned; }

void test_propagate_layer( const size_t num_targets
                         , const size_t num_sources
                         , real* restrict target
                         , const real* restrict source
                         , const real* restrict weights )
{
    real* tmp1 = alloca(num_targets*sizeof(real)+32);
    ALIGN32(tmp1);
    real* tmp2 = alloca(num_targets*sizeof(real)+32);
    ALIGN32(tmp2);

    memcpy(tmp1, target, sizeof(real)*num_targets);
    memcpy(tmp2, target, sizeof(real)*num_targets);

    propagate_layer( num_targets, num_sources, tmp1, source, weights );
    propagate_layer_avx( num_targets, num_sources, tmp2, source, weights );

    if ( memcmp(tmp1, tmp2, sizeof(real)*num_targets) ) {
        for ( size_t i1 = 0; i1 < num_targets; ++i1 ) {
            fprintf(stderr, "[%g/%g] ", tmp1[i1], tmp2[i1] );
        }
        fprintf(stderr, "\n");
        fprintf( stderr, "propagate_layer_avx IS BROKEN\n" );
        abort();
    }
}

void test_sigmoid_layer( const size_t num_items, real* target )
{
    real* tmp1 = alloca(num_items*sizeof(real)+32);
    ALIGN32(tmp1);
    real* tmp2 = alloca(num_items*sizeof(real)+32);
    ALIGN32(tmp2);

    memcpy(tmp1, target, sizeof(real)*num_items);
    memcpy(tmp2, target, sizeof(real)*num_items);

    sigmoid_layer( num_items, tmp1 );
    sigmoid_layer_avx( num_items, tmp2 );

    real diff = 0.0;

    for (size_t i1 = 0; i1 < num_items; ++i1 ) {
        diff += fabs(tmp1[i1]-tmp2[i1]);
    }
    if ( diff > 0.00001 ) {
        for (size_t i1 = 0; i1 < num_items; ++i1 ) {
            fprintf(stderr, "[%g/%g (%g)] ", tmp1[i1], tmp2[i1], target[i1]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "sigmoid_layer_avx IS BROKEN %g\n", diff);
        abort();
    }
}

void test_bias_layer( const size_t num_items
                    , real* target
                    , const real* source )
{
    real* tmp1 = alloca(num_items*sizeof(real)+32);
    ALIGN32(tmp1);
    real* tmp2 = alloca(num_items*sizeof(real)+32);
    ALIGN32(tmp2);

    memcpy(tmp1, target, sizeof(real)*num_items);
    memcpy(tmp2, target, sizeof(real)*num_items);

    bias_layer( num_items, tmp1, source );
    bias_layer_avx( num_items, tmp2, source );

    if ( memcmp(tmp1, tmp2, sizeof(real)*num_items) ) {
        fprintf(stderr, "bias_layer_avx IS BROKEN\n");
        abort();
    }
}

void propagate(lstm* lstm, const real* input, real* output)
{
    size_t neuron_offset = 0;
    size_t weight_offset = 0;

    real* state_input       = alloca(lstm->max_state_size*sizeof(real)+32);
    real* state_gate_input  = alloca(lstm->max_state_size*sizeof(real)+32);
    real* state_gate_forget = alloca(lstm->max_state_size*sizeof(real)+32);
    real* state_gate_output = alloca(lstm->max_state_size*sizeof(real)+32);
    real* state_activation  = alloca(lstm->max_activation_state_size*sizeof(real)+32);

    ALIGN32(state_input);
    ALIGN32(state_gate_input);
    ALIGN32(state_gate_forget);
    ALIGN32(state_gate_output);
    ALIGN32(state_activation);

    /* Input to first hidden layer */

    /* Input */
    zero_layer_avx( lstm->num_hiddens[0], state_input );
    propagate_layer( lstm->num_hiddens[0]
                   , lstm->num_inputs
                   , state_input
                   , input
                   , &lstm->weights_input.i[weight_offset] );
    /* Input gate */
    zero_layer_avx( lstm->num_hiddens[0], state_gate_input );
    propagate_layer( lstm->num_hiddens[0]
                   , lstm->num_inputs
                   , state_gate_input
                   , input
                   , &lstm->weights_input_gate.i[weight_offset] );
    /* Forget gate */
    zero_layer_avx( lstm->num_hiddens[0], state_gate_forget );
    propagate_layer( lstm->num_hiddens[0]
                   , lstm->num_inputs
                   , state_gate_forget
                   , input
                   , &lstm->weights_forget_gate.i[weight_offset] );
    /* Output gate */
    zero_layer_avx( lstm->num_hiddens[0], state_gate_output );
    propagate_layer( lstm->num_hiddens[0]
                   , lstm->num_inputs
                   , state_gate_output
                   , input
                   , &lstm->weights_output_gate.i[weight_offset] );

    /* Feedback connections, input */
    propagate_layer_avx( lstm->num_hiddens[0]
                   , lstm->num_hiddens[lstm->num_hidden_layers-1]
                   , state_input
                   , lstm->feedback_activations.i
                   , lstm->weights_feedback_input.i );

    /* Feedback connections, input gate */
    propagate_layer_avx( lstm->num_hiddens[0]
                   , lstm->num_hiddens[lstm->num_hidden_layers-1]
                   , state_gate_input
                   , lstm->feedback_activations.i
                   , lstm->weights_feedback_input_gate.i );

    /* Feedback connections, forget gate */
    propagate_layer_avx( lstm->num_hiddens[0]
                   , lstm->num_hiddens[lstm->num_hidden_layers-1]
                   , state_gate_forget
                   , lstm->feedback_activations.i
                   , lstm->weights_feedback_forget_gate.i );

    /* Output connections, output gate */
    propagate_layer_avx( lstm->num_hiddens[0]
                   , lstm->num_hiddens[lstm->num_hidden_layers-1]
                   , state_gate_output
                   , lstm->feedback_activations.i
                   , lstm->weights_feedback_output_gate.i );

    /* Input peephole */
    propagate_11_layer_avx( lstm->num_hiddens[0]
                      , state_gate_input
                      , &lstm->memory_cells.i[neuron_offset]
                      , &lstm->weights_memory_to_input_gate.i[neuron_offset] );

    /* Forget peephole */
    propagate_11_layer_avx( lstm->num_hiddens[0]
                      , state_gate_forget
                      , &lstm->memory_cells.i[neuron_offset]
                      , &lstm->weights_memory_to_forget_gate.i[neuron_offset] );

    /* Output peephole */
    propagate_11_layer_avx( lstm->num_hiddens[0]
                      , state_gate_output
                      , &lstm->memory_cells.i[neuron_offset]
                      , &lstm->weights_memory_to_output_gate.i[neuron_offset] );

    TEST(test_bias_layer( lstm->num_hiddens[0], state_gate_forget, &lstm->bias_forget_gate.i[neuron_offset] ));

    bias_layer_avx( lstm->num_hiddens[0], state_gate_forget, &lstm->bias_forget_gate.i[neuron_offset] );
    bias_layer_avx( lstm->num_hiddens[0], state_gate_input, &lstm->bias_input_gate.i[neuron_offset] );
    bias_layer_avx( lstm->num_hiddens[0], state_gate_output, &lstm->bias_output_gate.i[neuron_offset] );
    bias_layer_avx( lstm->num_hiddens[0], state_input, &lstm->bias_input.i[neuron_offset] );

    TEST(test_sigmoid_layer( lstm->num_hiddens[0], state_gate_forget ));

    sigmoid_layer_avx( lstm->num_hiddens[0], state_gate_forget );
    sigmoid_layer_avx( lstm->num_hiddens[0], state_gate_input );
    sigmoid_layer_avx( lstm->num_hiddens[0], state_gate_output );
    tanh_layer_avx( lstm->num_hiddens[0], state_input );

    memory_cell_layer_avx( lstm->num_hiddens[0]
                     , &lstm->memory_cells.i[neuron_offset]
                     , state_activation
                     , state_input
                     , state_gate_input
                     , state_gate_output
                     , state_gate_forget );

    neuron_offset += lstm->num_hiddens[0];
    weight_offset += lstm->num_inputs*lstm->num_hiddens[0];

    /* Hidden layers */
    for ( size_t i1 = 1; i1 < lstm->num_hidden_layers; ++i1 )
    {
        /* Input */
        zero_layer_avx( lstm->num_hiddens[i1], state_input );
        TEST(test_propagate_layer( lstm->num_hiddens[i1]
                       , lstm->num_hiddens[i1-1]
                       , state_input
                       , state_activation
                       , &lstm->weights_input.i[weight_offset] ));

        propagate_layer_avx( lstm->num_hiddens[i1]
                           , lstm->num_hiddens[i1-1]
                           , state_input
                           , state_activation
                           , &lstm->weights_input.i[weight_offset] );
        /* Input gate */
        zero_layer_avx( lstm->num_hiddens[i1], state_gate_input );
        propagate_layer_avx( lstm->num_hiddens[i1]
                           , lstm->num_hiddens[i1-1]
                           , state_gate_input
                           , state_activation
                           , &lstm->weights_input_gate.i[weight_offset] );
        /* Forget gate */
        zero_layer_avx( lstm->num_hiddens[i1], state_gate_forget );
        propagate_layer_avx( lstm->num_hiddens[i1]
                           , lstm->num_hiddens[i1-1]
                           , state_gate_forget
                           , state_activation
                           , &lstm->weights_forget_gate.i[weight_offset] );
        /* Output gate */
        zero_layer_avx( lstm->num_hiddens[i1], state_gate_output );
        propagate_layer_avx( lstm->num_hiddens[i1]
                           , lstm->num_hiddens[i1-1]
                           , state_gate_output
                           , state_activation
                           , &lstm->weights_output_gate.i[weight_offset] );

        /* Input peephole */
        propagate_11_layer_avx( lstm->num_hiddens[i1]
                          , state_gate_input
                          , &lstm->memory_cells.i[neuron_offset]
                          , &lstm->weights_memory_to_input_gate.i[neuron_offset] );

        /* Forget peephole */
        propagate_11_layer_avx( lstm->num_hiddens[i1]
                          , state_gate_forget
                          , &lstm->memory_cells.i[neuron_offset]
                          , &lstm->weights_memory_to_forget_gate.i[neuron_offset] );

        /* Output peephole */
        propagate_11_layer_avx( lstm->num_hiddens[i1]
                          , state_gate_output
                          , &lstm->memory_cells.i[neuron_offset]
                          , &lstm->weights_memory_to_output_gate.i[neuron_offset] );

        bias_layer_avx( lstm->num_hiddens[i1], state_gate_forget, &lstm->bias_forget_gate.i[neuron_offset] );
        bias_layer_avx( lstm->num_hiddens[i1], state_gate_input, &lstm->bias_input_gate.i[neuron_offset] );
        bias_layer_avx( lstm->num_hiddens[i1], state_gate_output, &lstm->bias_output_gate.i[neuron_offset] );
        bias_layer_avx( lstm->num_hiddens[i1], state_input, &lstm->bias_input.i[neuron_offset] );

        sigmoid_layer_avx( lstm->num_hiddens[i1], state_gate_forget );
        sigmoid_layer_avx( lstm->num_hiddens[i1], state_gate_input );
        sigmoid_layer_avx( lstm->num_hiddens[i1], state_gate_output );
        tanh_layer_avx( lstm->num_hiddens[i1], state_input );

        memory_cell_layer_avx( lstm->num_hiddens[i1]
                         , &lstm->memory_cells.i[neuron_offset]
                         , state_activation
                         , state_input
                         , state_gate_input
                         , state_gate_output
                         , state_gate_forget );

        neuron_offset += lstm->num_hiddens[i1];
        weight_offset += lstm->num_hiddens[i1]*lstm->num_hiddens[i1-1];
    }

    memcpy( lstm->feedback_activations.i, state_activation, sizeof(real)*lstm->num_hiddens[lstm->num_hidden_layers-1] );

    /* Last hidden layer to output */
    zero_layer( lstm->num_outputs, output );
    propagate_layer( lstm->num_outputs
                   , lstm->num_hiddens[lstm->num_hidden_layers-1]
                   , output
                   , state_activation
                   , lstm->weights_last_layer.i );
    sigmoid_layer( lstm->num_outputs
                 , output );
}

static size_t smax(size_t a, size_t b)
{
    return a > b ? a : b;
}

lstm* allocate_lstm( size_t num_inputs, size_t num_outputs, const size_t* num_hiddens, size_t num_hidden_layers )
{
    if ( num_hidden_layers < 1 )
        return NULL;

    for ( size_t i1 = 0; i1 < num_hidden_layers; ++i1 ) {
        if ( num_hiddens[i1] % 4 != 0 ) {
            return NULL;
        }
    }

    lstm* result = calloc(sizeof(lstm), 1);
    if ( !result )
        return NULL;

    result->num_inputs = num_inputs;
    result->num_outputs = num_outputs;
    result->num_hidden_layers = num_hidden_layers;

    result->max_state_size = smax(num_inputs, num_outputs);
    for ( size_t i1 = 0; i1 < num_hidden_layers; ++i1 ) {
        result->max_state_size = smax(num_hiddens[i1], result->max_state_size);
    }
    result->max_activation_state_size = result->max_state_size;

    size_t num_hidden_neurons = 0;
    size_t num_weights = num_inputs*num_hiddens[0];
    for ( size_t i1 = 0; i1 < num_hidden_layers; ++i1 ) {
        num_hidden_neurons += num_hiddens[i1];
    }
    for ( size_t i1 = 1; i1 < num_hidden_layers; ++i1 ) {
        num_weights += num_hiddens[i1]*num_hiddens[i1-1];
    }

    result->num_hiddens = calloc(sizeof(size_t), num_hidden_layers);
    if ( !result->num_hiddens )
        goto bailout;
    memcpy(result->num_hiddens, num_hiddens, sizeof(size_t)*num_hidden_layers);

#define A(lvalue, sz) { void* ptr = NULL; \
                        int r = posix_memalign(&ptr, 32, sizeof(real)*(sz)); \
                        if ( r != 0 ) goto bailout; \
                        memset(ptr, 0, sizeof(real)*(sz)); \
                        lvalue = (real*) ptr; }

    A(result->memory_cells.i, num_hidden_neurons);
    A(result->weights_input.i, num_weights);
    A(result->weights_input_gate.i, num_weights);
    A(result->weights_forget_gate.i, num_weights);
    A(result->weights_output_gate.i, num_weights);
    A(result->feedback_activations.i, num_hiddens[num_hidden_layers-1]);
    A(result->weights_feedback_input.i, num_hiddens[num_hidden_layers-1]*num_hiddens[0]);
    A(result->weights_feedback_input_gate.i, num_hiddens[num_hidden_layers-1]*num_hiddens[0]);
    A(result->weights_feedback_forget_gate.i, num_hiddens[num_hidden_layers-1]*num_hiddens[0]);
    A(result->weights_feedback_output_gate.i, num_hiddens[num_hidden_layers-1]*num_hiddens[0]);
    A(result->weights_last_layer.i, num_hiddens[num_hidden_layers-1]*num_outputs);
    A(result->bias_input.i, num_hidden_neurons);
    A(result->bias_input_gate.i, num_hidden_neurons);
    A(result->bias_forget_gate.i, num_hidden_neurons);
    A(result->bias_output_gate.i, num_hidden_neurons);
    A(result->weights_memory_to_output_gate.i, num_hidden_neurons);
    A(result->weights_memory_to_input_gate.i, num_hidden_neurons);
    A(result->weights_memory_to_forget_gate.i, num_hidden_neurons);

#undef A
    return result;
bailout:
    free_lstm(result);
    return NULL;
}

size_t serialized_lstm_size( const lstm* lstm )
{
    size_t num_weights = lstm->num_inputs*lstm->num_hiddens[0];
    for ( size_t i1 = 1; i1 < lstm->num_hidden_layers; ++i1 ) {
        num_weights += lstm->num_hiddens[i1]*lstm->num_hiddens[i1-1];
    }
    size_t num_neurons = 0;
    for ( size_t i1 = 0; i1 < lstm->num_hidden_layers; ++i1 ) {
        num_neurons += lstm->num_hiddens[i1];
    }
    size_t num_feedback_weights = lstm->num_hiddens[lstm->num_hidden_layers-1]*lstm->num_hiddens[0];
    return num_weights*4 +
           num_neurons*7 +
           num_feedback_weights*4 +
           lstm->num_hiddens[lstm->num_hidden_layers-1]*lstm->num_outputs;
}

void serialize_lstm( const lstm* lstm, real* serialized )
{
    size_t offset = 0;
    size_t num_weights = lstm->num_inputs*lstm->num_hiddens[0];
    for ( size_t i1 = 1; i1 < lstm->num_hidden_layers; ++i1 ) {
        num_weights += lstm->num_hiddens[i1]*lstm->num_hiddens[i1-1];
    }
    size_t num_neurons = 0;
    for ( size_t i1 = 0; i1 < lstm->num_hidden_layers; ++i1 ) {
        num_neurons += lstm->num_hiddens[i1];
    }

#define APPEND(from_where, num_items) \
    { memcpy( &serialized[offset], from_where, num_items*sizeof(real)); \
      offset += num_items; }
    APPEND(lstm->weights_input.i, num_weights );
    APPEND(lstm->weights_input_gate.i, num_weights );
    APPEND(lstm->weights_forget_gate.i, num_weights );
    APPEND(lstm->weights_output_gate.i, num_weights );
    APPEND(lstm->weights_feedback_input.i, lstm->num_hiddens[0]*lstm->num_hiddens[lstm->num_hidden_layers-1] );
    APPEND(lstm->weights_feedback_input_gate.i, lstm->num_hiddens[0]*lstm->num_hiddens[lstm->num_hidden_layers-1] );
    APPEND(lstm->weights_feedback_forget_gate.i, lstm->num_hiddens[0]*lstm->num_hiddens[lstm->num_hidden_layers-1] );
    APPEND(lstm->weights_feedback_output_gate.i, lstm->num_hiddens[0]*lstm->num_hiddens[lstm->num_hidden_layers-1] );
    APPEND(lstm->weights_last_layer.i, lstm->num_hiddens[lstm->num_hidden_layers-1]*lstm->num_outputs);
    APPEND(lstm->bias_input.i, num_neurons);
    APPEND(lstm->bias_input_gate.i, num_neurons);
    APPEND(lstm->bias_forget_gate.i, num_neurons);
    APPEND(lstm->bias_output_gate.i, num_neurons);
    APPEND(lstm->weights_memory_to_output_gate.i, num_neurons);
    APPEND(lstm->weights_memory_to_input_gate.i, num_neurons);
    APPEND(lstm->weights_memory_to_forget_gate.i, num_neurons);
#undef APPEND
}

void unserialize_lstm( lstm* lstm, const real* serialized )
{
    size_t offset = 0;
    size_t num_weights = lstm->num_inputs*lstm->num_hiddens[0];
    for ( size_t i1 = 1; i1 < lstm->num_hidden_layers; ++i1 ) {
        num_weights += lstm->num_hiddens[i1]*lstm->num_hiddens[i1-1];
    }
    size_t num_neurons = 0;
    for ( size_t i1 = 0; i1 < lstm->num_hidden_layers; ++i1 ) {
        num_neurons += lstm->num_hiddens[i1];
    }

#define APPEND(to_where, num_items) \
    { memcpy(to_where, &serialized[offset], num_items*sizeof(real)); \
      offset += num_items; }
    APPEND(lstm->weights_input.i, num_weights );
    APPEND(lstm->weights_input_gate.i, num_weights );
    APPEND(lstm->weights_forget_gate.i, num_weights );
    APPEND(lstm->weights_output_gate.i, num_weights );
    APPEND(lstm->weights_feedback_input.i, lstm->num_hiddens[0]*lstm->num_hiddens[lstm->num_hidden_layers-1] );
    APPEND(lstm->weights_feedback_input_gate.i, lstm->num_hiddens[0]*lstm->num_hiddens[lstm->num_hidden_layers-1] );
    APPEND(lstm->weights_feedback_forget_gate.i, lstm->num_hiddens[0]*lstm->num_hiddens[lstm->num_hidden_layers-1] );
    APPEND(lstm->weights_feedback_output_gate.i, lstm->num_hiddens[0]*lstm->num_hiddens[lstm->num_hidden_layers-1] );
    APPEND(lstm->weights_last_layer.i, lstm->num_hiddens[lstm->num_hidden_layers-1]*lstm->num_outputs);
    APPEND(lstm->bias_input.i, num_neurons);
    APPEND(lstm->bias_input_gate.i, num_neurons);
    APPEND(lstm->bias_forget_gate.i, num_neurons);
    APPEND(lstm->bias_output_gate.i, num_neurons);
    APPEND(lstm->weights_memory_to_output_gate.i, num_neurons);
    APPEND(lstm->weights_memory_to_input_gate.i, num_neurons);
    APPEND(lstm->weights_memory_to_forget_gate.i, num_neurons);
#undef APPEND
}

void reset_lstm( lstm* lstm )
{
    size_t num_neurons = 0;
    for ( size_t i1 = 0; i1 < lstm->num_hidden_layers; ++i1 ) {
        num_neurons += lstm->num_hiddens[i1];
    }
    memset( lstm->memory_cells.i, 0, sizeof(real)*num_neurons );
    memset( lstm->feedback_activations.i, 0, sizeof(real)*lstm->num_hiddens[lstm->num_hidden_layers-1] );
}

void free_lstm( lstm* lstm )
{
    if ( !lstm )
        return;

    free(lstm->num_hiddens);
    free(lstm->memory_cells.i);
    free(lstm->weights_input.i);
    free(lstm->weights_input_gate.i);
    free(lstm->weights_forget_gate.i);
    free(lstm->weights_output_gate.i);
    free(lstm->feedback_activations.i);
    free(lstm->weights_feedback_input.i);
    free(lstm->weights_feedback_input_gate.i);
    free(lstm->weights_feedback_forget_gate.i);
    free(lstm->weights_feedback_output_gate.i);
    free(lstm->weights_last_layer.i);
    free(lstm->bias_input.i);
    free(lstm->bias_input_gate.i);
    free(lstm->bias_forget_gate.i);
    free(lstm->bias_output_gate.i);
    free(lstm->weights_memory_to_output_gate.i);
    free(lstm->weights_memory_to_input_gate.i);
    free(lstm->weights_memory_to_forget_gate.i);
    memset( lstm, 0, sizeof(*lstm) );
    free(lstm);
}

