Implementation of recurrent LSTM neural networks
================================================

This is a very single-purpose library that implements AVX-optimized LSTM neural
networks for C for Unix platforms. Works at least on Linux.

The neural network architecture currently supported by this library is:

  * An input layer.

  * At least one hidden layer; each hidden layer must contain a multiple of 4
    LSTM nodes.

  * An output layer.

All layers are fully connected to their respective next layer. Additionally,
the last hidden layer is fully connected to first hidden layer. If there is
only one hidden layer (this is common architecture), then that layer is
connected to itself.

LSTM nodes have forget gate, output gate and input gate and peepholes from
memory cells to all of these. All gates also have bias. The trainable
parameters are weights (including all weights to gates and feedback gates from
last layer to first), peepholes and biases.

I'm using this library with non-gradient based optimization which is why this
library does not compute gradients.

License
-------

License is LGPL 2.1.

Compile
-------

You can use the `Makefile`:

    $ make

This should result in a `liblstm.so` file.

You can use `make install` but the Makefile is hard-coded to install to
`/usr/local`. This library is just 3 files so you can also just include the
files in your project or compile by hand.

API
---

API is simplistic. You might see the type `real` in these functions, at the
moment, `real` is always `double`.

    lstm* allocate_lstm( size_t num_inputs, size_t num_outputs, const size_t* num_hiddens, size_t num_hidden_layers );
    void free_lstm( lstm* lstm );

`allocate_lstm` allocates an LSTM network. Returns NULL if out of memory or
configuration is not supported. There must be at least one hidden layer and
each hidden layer must have a multiple of 4 neurons. Use `free_lstm` to free up
your resources.

    void propagate(lstm* restrict lstm, const real* restrict input, real* restrict output);

`propagate` runs one iteration of the network. Inputs are taken from input
vector and outputs are written to your output. Feedback activations are saved
so your network has state.

    void reset_lstm( lstm* lstm );

`reset_lstm` resets all memory cells and feedback activations.

    size_t serialized_lstm_size( const lstm* lstm );
    void serialize_lstm( const lstm* lstm, real* serialized );
    void unserialize_lstm( lstm* lstm, const real* serialized );

These functions are used to serialize and unserialize parameters from an
`lstm`. `serialized_lstm_size` returns the number of parameters and this is the
number of `real`s you need for `serialize_lstm` and `unserialize_lstm`.

`serialize_lstm` takes an `lstm` and writes all parameters to one-dimensional
array. `unserialize_lstm` does the reverse and updates parameters in `lstm`
according to given one-dimensional array.

`lstm` networks created with the same size parameters in `allocate_lstm` have
identical size and you can serialize and unserialize between them.

