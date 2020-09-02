"""Neural machine translation with attention."""
from __future__ import absolute_import, division, print_function

import math

import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl

from nmt import nmt


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer, colocation):
    """Computes the attention and alignments for a given attention_mechanism"""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

    with nmt.colocation_cm(colocation, name='attention'):
        # Reshape from [batch_size, memory_time] to
        #   [batch_size, 1, memory_time]
        expanded_alignments = tf.expand_dims(alignments, 1)
        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is
        #   [batch_size, 1, memory_time]
        # attention_mechanism.values shape is
        #   [batch_size, memory_time, memory_size]
        # the batched matmul is over memory_time, so the output shape is
        #   [batch_size, 1, memory_size].
        # we then squeeze out the singleton dim.
        context = tf.matmul(expanded_alignments, attention_mechanism.values)
        context = tf.squeeze(context, [1])

        if attention_layer is not None:
            attention = attention_layer(tf.concat([cell_output, context], 1))
        else:
            attention = context

    return attention, alignments, next_attention_state


class AttentionWrapper(tf.nn.rnn_cell.RNNCell):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                 cell,
                 attention_mechanism,
                 colocation,
                 attention_layer_size=None,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None,
                 attention_layer=None):
        """Construct the `AttentionWrapper`."""
        super(AttentionWrapper, self).__init__(name=name)
        self._is_multi = False
        if not isinstance(attention_mechanism,
                          tf.contrib.seq2seq.AttentionMechanism):
            raise TypeError(
                "attention_mechanism must be an AttentionMechanism or list of "
                "multiple AttentionMechanism instances, saw type: %s"
                % type(attention_mechanism).__name__)

        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, attention: tf.concat([inputs, attention], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                    "cell_input_fn must be callable, saw type: %s"
                    % type(cell_input_fn).__name__)

        if attention_layer_size is not None and attention_layer is not None:
            raise ValueError(
                "Only one of attention_layer_size and attention_layer "
                "should be set")

        if attention_layer_size is not None:
            self._attention_layer = tf.layers.Dense(
                attention_layer_size,
                name="attention_layer",
                use_bias=False,
                dtype=attention_mechanism.dtype)
            self._attention_layer_size = attention_layer_size
        elif attention_layer is not None:
            self._attention_layer = attention_layer
            self._attention_layer_size = \
                self._attention_layer.compute_output_shape(
                    [None,
                     cell.output_size +
                     attention_mechanism.values.shape[-1].value])[-1].value
        else:
            self._attention_layer = None
            self._attention_layer_size = \
                attention_mechanism.values.get_shape()[-1].value

        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._colocation = colocation
        with tf.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                    final_state_tensor.shape[0].value
                    or tf.shape(final_state_tensor)[0])
                self._initial_cell_state = initial_cell_state

    @property
    def output_size(self):
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.
        Returns:
        An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return tf.contrib.seq2seq.AttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._attention_mechanism.alignments_size,
            attention_state=self._attention_mechanism.state_size,
            alignment_history=())

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
        batch_size: `0D` integer tensor: the batch size.
        dtype: The internal state data type.
        Returns:
        An `AttentionWrapperState` tuple containing zeroed out tensors and,
        possibly, empty `TensorArray` objects.
        Raises:
        ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        name_scope_str = type(self).__name__ + "ZeroState"
        with tf.name_scope(name_scope_str, values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            initial_alignment = self._attention_mechanism.initial_alignments(
                batch_size, dtype)
            return tf.contrib.seq2seq.AttentionWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int32),
                attention=rnn_cell_impl._zero_state_tensors(
                    self._attention_layer_size, batch_size, dtype),
                alignments=initial_alignment,
                attention_state=self._attention_mechanism.initial_state(
                    batch_size, dtype),
                alignment_history=())

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
            `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous
            state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
            `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
            alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell
            output and context through the attention layer (a linear layer with
            `attention_layer_size` outputs).

        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time
                step.
            state: An instance of `AttentionWrapperState` containing
                tensors from the previous time step.
        Returns:
            A tuple `(attention_or_cell_output, next_state)`, where:
            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of `AttentionWrapperState`
                containing the state calculated at this time step.
        Raises:
            TypeError: If `state` is not an instance of `AttentionWrapperState`
        """
        if not isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
            raise TypeError(
                "Expected state to be instance of AttentionWrapperState. "
                "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        previous_attention_state = state.attention_state

        attention, alignments, next_attention_state = _compute_attention(
            self._attention_mechanism, cell_output, previous_attention_state,
            self._attention_layer if self._attention_layer else None,
            self._colocation)

        next_state = tf.contrib.seq2seq.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=next_attention_state,
            alignments=alignments,
            alignment_history=())

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


def _luong_score(query, keys, scale):
    """Implements Luong-style (multiplicative) scoring function.

    This attention has two forms.  The first is standard Luong attention,
    as described in:
    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.  https://arxiv.org/abs/1508.04025

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.
    To enable the second form, call this function with `scale=True`.

    Args:
        query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        scale: Whether to apply a scale to the score function.
    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    Raises:
        ValueError: If `key` and `query` depths do not match.
    """
    depth = query.get_shape()[-1]
    key_units = keys.get_shape()[-1]
    if depth != key_units:
        raise ValueError(
            "Incompatible or unknown inner dimensions between query and keys. "
            "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
            "Perhaps you need to set num_units to the keys' dimension (%s)?"
            % (query, depth, keys, key_units, key_units))
    dtype = query.dtype

    # Reshape from [batch_size, depth] to [batch_size, 1, depth]
    # for matmul.
    query = tf.expand_dims(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                keys is [batch_size, max_time, depth].
    # the inner product is asked to **transpose keys' inner shape** to get a
    # batched matmul on:
    #   [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of:
    #   [batch_size, 1, max_time].
    # we then squeeze out the center singleton dimension.
    score = tf.matmul(query, keys, transpose_b=True)
    score = tf.squeeze(score, [1])

    if scale:
        # Scalar used in weight scaling
        g = tf.get_variable(
            "attention_g", dtype=dtype,
            initializer=tf.ones_initializer, shape=())
        score = g * score
    return score


class LuongAttentionColocation(tf.contrib.seq2seq.LuongAttention):

    def __init__(self, *args, **kwargs):
        self._colocation = kwargs.pop('colocation')
        super(LuongAttentionColocation, self).__init__(*args, **kwargs)

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        with tf.variable_scope(None, "luong_attention", [query]):
            with nmt.colocation_cm(self._colocation, name="luong_attention"):
                score = _luong_score(query, self._keys, self._scale)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


def _bahdanau_score(processed_query, keys, normalize):
    """Implements Bahdanau-style (additive) scoring function.

    This attention has two forms.  The first is Bhandanau attention,
    as described in:
    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:
    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, set `normalize=True`.

    Args:
        processed_query: Tensor, shape `[batch_size, num_units]`
            to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        normalize: Whether to normalize the score function.
    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    v = tf.get_variable(
        "attention_v", [num_units], dtype=dtype)
    if normalize:
        # Scalar used in weight normalization
        g = tf.get_variable(
            "attention_g", dtype=dtype,
            initializer=tf.constant_initializer(
                math.sqrt((1. / num_units))),
            shape=())
        # Bias added prior to the nonlinearity
        b = tf.get_variable(
            "attention_b", [num_units], dtype=dtype,
            initializer=tf.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * tf.math.rsqrt(
            tf.math.reduce_sum(tf.math.square(v)))
        return tf.math.reduce_sum(
            normed_v * tf.math.tanh(keys + processed_query + b), [2])
    else:
        return tf.math.reduce_sum(v * tf.math.tanh(keys + processed_query), [2])


class BahdanauAttentionColocation(tf.contrib.seq2seq.BahdanauAttention):

    def __init__(self, *args, **kwargs):
        self._colocation = kwargs.pop('colocation')
        super(BahdanauAttentionColocation, self).__init__(*args, **kwargs)
        self.query_layer.build(self._num_units)

    def __call__(self, query, state):
        """Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        with tf.variable_scope(None, "bahdanau_attention", [query]):
            with nmt.colocation_cm(self._colocation, name="noop"):
                processed_query = (self.query_layer(
                    query) if self.query_layer else query)
                score = _bahdanau_score(
                    processed_query, self._keys, self._normalize)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


def create_attention_mechanism(
        attention_option, num_units, memory, colocation):
    """Create attention mechanism based on the attention_option."""
    # Mechanism
    if attention_option == "luong":
        attention_mechanism = LuongAttentionColocation(num_units, memory,
                                                       colocation=colocation)
    elif attention_option == "scaled_luong":
        attention_mechanism = LuongAttentionColocation(
            num_units,
            memory,
            scale=True,
            colocation=colocation)
    elif attention_option == "bahdanau":
        attention_mechanism = BahdanauAttentionColocation(
            num_units, memory, colocation=True)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = BahdanauAttentionColocation(
            num_units,
            memory,
            normalize=True,
            colocation=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


class AttentionNMT(nmt.NMT):
    """NMT with atttention model builder."""
    # pylint: disable=too-few-public-methods

    def __init__(self, hparams, colocation, unroll,
                 attention_architecture='standard',
                 dtype=tf.float32):
        super(AttentionNMT, self).__init__(hparams, colocation, unroll, dtype)
        self.attention_architecture = attention_architecture

        self.create_attention_fn = create_attention_mechanism

    def _build_decoder_cell(self, encoder_outputs, encoder_state):
        assert self.attention_architecture == 'standard'

        attention_mechanism = self.create_attention_fn(
            "bahdanau", self.hparams.num_units, memory=encoder_outputs,
            colocation=self.colocation)

        decoder_cell = self.create_rnn_cell(
            self.hparams.num_layers, self.hparams.num_residual_layers)

        decoder_cell = AttentionWrapper(
            decoder_cell,
            attention_mechanism,
            colocation=self.colocation)

        # pass hidden state
        batch_size = encoder_outputs.shape[0]
        initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)

        return decoder_cell, initial_state


def attention_nmt(src_input, tgt_input, tgt_output, colocation, unroll=True,
                  **kwargs):
    """Returns a NMT with attention network."""
    hparams = nmt.create_nmt_hparams(**kwargs)
    return AttentionNMT(hparams, colocation, unroll).build_graph(
        src_input, tgt_input, tgt_output)
