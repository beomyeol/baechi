"""Neural machine translation."""
from __future__ import absolute_import, division, print_function

import collections
import contextlib

import tensorflow as tf
from nmt import model_helper

_CELL_CLASS = {
    'lstm': tf.keras.layers.LSTMCell,
    'gru': tf.keras.layers.GRUCell,
}


def _get_rnn_cell(unit_type, rnn_units, num_layers):
    cell_class = _CELL_CLASS[unit_type]
    cells = [cell_class(rnn_units) for _ in range(num_layers)]
    return (cells[0] if len(cells) == 1
            else tf.keras.layers.StackedRNNCells(cells))


def keras_nmt(src_input, tgt_input, src_vocab_size, tgt_vocab_size,
              rnn_unit_type, rnn_units, num_layers, encoder_type,
              residual=False, unroll=True):
    """Returns a NMT Keras model."""
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # pylint: disable=unused-argument
    if residual:
        raise ValueError('residual connections are not supported.')

    encoder_inputs = tf.keras.layers.Input(tensor=src_input)
    encoder_embedding = tf.keras.layers.Embedding(src_vocab_size, rnn_units)
    x = encoder_embedding(encoder_inputs)
    if encoder_type == 'uni':
        cell = _get_rnn_cell(rnn_unit_type, rnn_units, num_layers)
        encoder = tf.keras.layers.RNN(cell, return_state=True, unroll=unroll)
    elif encoder_type == 'bi':
        cell = _get_rnn_cell(rnn_unit_type, rnn_units, num_layers//2)
        encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.RNN(cell, return_state=True, unroll=unroll))
    encoder_states = encoder(x)[1:]  # [encoder output, encoder state, ...]

    decoder_inputs = tf.keras.layers.Input(tensor=tgt_input)
    decoder_embedding = tf.keras.layers.Embedding(tgt_vocab_size, rnn_units)
    x = decoder_embedding(decoder_inputs)
    cell = _get_rnn_cell(rnn_unit_type, rnn_units, num_layers)
    decoder = tf.keras.layers.RNN(cell, return_sequences=True, unroll=unroll)
    decoder_outputs = decoder(x, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(tgt_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    return tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)


NMTHParams = collections.namedtuple(
    'NMTHParams',
    [
        'src_vocab_size',
        'tgt_vocab_size',
        'unit_type',
        'num_units',
        'num_layers',
        'encoder_type',
        'forget_bias',
        'num_residual_layers',
        'dropout',
        'num_gpus'
    ]
)


def create_nmt_hparams(src_vocab_size, tgt_vocab_size, rnn_unit_type,
                       rnn_units, num_layers, encoder_type, residual,
                       num_gpus=1):
    """Returns NMTHParams with the given arguments."""
    # pylint: disable=too-many-arguments
    return NMTHParams(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        unit_type=rnn_unit_type,
        num_units=rnn_units,
        num_layers=num_layers,
        encoder_type=encoder_type,
        forget_bias=1.0,
        num_residual_layers=(num_layers - 1 if residual else 0),
        dropout=0.0,
        num_gpus=num_gpus,
    )


def colocation_cm(colocation, name=None, op=None):
    """Gets a context manager to colocate ops."""
    if colocation:
        return tf.colocate_with(tf.no_op(name) if op is None else op)
    return contextlib.suppress()


class NMT():
    """NMT model builder."""

    def __init__(self, hparams, colocation, unroll, dtype=tf.float32):
        self.hparams = hparams
        self.colocation = colocation
        self.unroll = unroll
        self.dtype = dtype

    def _colocation_cm(self, name=None, op=None):
        return colocation_cm(self.colocation, name=name, op=op)

    @staticmethod
    def build_embedding(inputs, vocab_size, embedding_dim, colocation):
        """Returns embed inputs for the given inputs with the spec."""
        with colocation_cm(colocation, name='embedding_noop'):
            embedding = tf.get_variable(
                'embedding', shape=(vocab_size, embedding_dim))
            return tf.nn.embedding_lookup(embedding, inputs)

    @staticmethod
    def rnn(cell, inputs, unroll, dtype=None, initial_state=None):
        """Runs static or dynamic rnn."""
        if unroll:
            outputs, state = tf.nn.static_rnn(
                cell, inputs=inputs, dtype=dtype, initial_state=initial_state)
        else:
            outputs, state = tf.nn.dynamic_rnn(
                cell, inputs=inputs, dtype=dtype, initial_state=initial_state,
                swap_memory=True)
        return outputs, state

    @staticmethod
    def bi_rnn(fw_cell, bw_cell, inputs, unroll, dtype):
        """Runs static or dynamic bidirectional rnn."""
        if unroll:
            bi_outputs, fw_state, bw_state = tf.nn.static_bidirectional_rnn(
                fw_cell, bw_cell, inputs, dtype=dtype)
            bi_state = (fw_state, bw_state)
        else:
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, inputs, dtype=dtype)
            bi_outputs = tf.concat(bi_outputs, -1)
        return bi_outputs, bi_state

    def create_rnn_cell(self, num_layers, num_residual_layers, base_gpu=0):
        """Wrapper function for model_helper.create_rnn()."""
        return model_helper.create_rnn_cell(
            unit_type=self.hparams.unit_type,
            num_units=self.hparams.num_units,
            num_layers=num_layers,
            forget_bias=self.hparams.forget_bias,
            num_residual_layers=num_residual_layers,
            dropout=self.hparams.dropout,
            colocation=self.colocation,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            num_gpus=self.hparams.num_gpus,
            base_gpu=base_gpu,
            single_cell_fn=None)

    def _build_bidirectional_rnn(self, inputs, num_bi_layers,
                                 num_bi_residual_layers):
        fw_cell = self.create_rnn_cell(num_bi_layers, num_bi_residual_layers)
        bw_cell = self.create_rnn_cell(num_bi_layers, num_bi_residual_layers)
        encoder_outputs, bi_encoder_state = self.bi_rnn(
            fw_cell, bw_cell, inputs, self.unroll, dtype=self.dtype)
        return encoder_outputs, bi_encoder_state

    def _build_encoder(self, src_input):
        with tf.variable_scope('encoder'):
            encoder_emb_input = self.build_embedding(
                src_input,
                vocab_size=self.hparams.src_vocab_size,
                embedding_dim=self.hparams.num_units,
                colocation=self.colocation)

            if self.unroll:
                encoder_emb_input = tf.unstack(encoder_emb_input, axis=1)

            if self.hparams.encoder_type == 'uni':
                encoder_cell = self.create_rnn_cell(
                    self.hparams.num_layers, self.hparams.num_residual_layers)
                encoder_outputs, encoder_state = self.rnn(
                    encoder_cell, encoder_emb_input, self.unroll,
                    dtype=self.dtype)
            elif self.hparams.encoder_type == 'bi':
                num_bi_layers = int(self.hparams.num_layers / 2)
                num_bi_residual_layers = int(
                    self.hparams.num_residual_layers / 2)

                res = self._build_bidirectional_rnn(
                    encoder_emb_input, num_bi_layers, num_bi_residual_layers)
                encoder_outputs, bi_encoder_state = res

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(
                            bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(
                            bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError('Unknown encoder type: %s' %
                                 self.hparams.encoder_type)

            if self.unroll:
                device_str = model_helper.get_device_str(
                    self.hparams.num_layers - 1,
                    self.hparams.num_gpus)
                with tf.device(device_str):
                    encoder_outputs = tf.stack(encoder_outputs, axis=1)

        return encoder_outputs, encoder_state

    def _build_decoder_cell(self, encoder_outputs, encoder_state):
        # pylint: disable=unused-argument
        cell = self.create_rnn_cell(self.hparams.num_layers,
                                    self.hparams.num_residual_layers)
        return cell, encoder_state

    def _build_decoder(self, tgt_input, encoder_outputs, encoder_state):
        with tf.variable_scope('decoder'):
            decoder_emb_input = self.build_embedding(
                tgt_input,
                vocab_size=self.hparams.tgt_vocab_size,
                embedding_dim=self.hparams.num_units,
                colocation=self.colocation)

            if self.unroll:
                decoder_emb_input = tf.unstack(decoder_emb_input, axis=1)

            decoder_cell, initial_state = self._build_decoder_cell(
                encoder_outputs, encoder_state)

            decoder_outputs, _ = self.rnn(decoder_cell,
                                          decoder_emb_input,
                                          self.unroll,
                                          initial_state=initial_state)

            if self.unroll:
                device_str = model_helper.get_device_str(
                    self.hparams.num_layers - 1,
                    self.hparams.num_gpus)
                with tf.device(device_str):
                    decoder_outputs = tf.stack(decoder_outputs, axis=1)

            return decoder_outputs

    def build_graph(self, src_input, tgt_input, tgt_output):
        """Returns logits."""
        encoder_outputs, encoder_state = self._build_encoder(src_input)
        decoder_outputs = self._build_decoder(
            tgt_input, encoder_outputs, encoder_state)

        output_layer = tf.layers.Dense(self.hparams.tgt_vocab_size,
                                       use_bias=False,
                                       name='output_projection')
        output_layer.build(decoder_outputs.shape)

        num_layers = self.hparams.num_layers
        num_gpus = self.hparams.num_gpus

        device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
        with tf.device(model_helper.get_device_str(device_id, num_gpus)):
            with self._colocation_cm(name='output_projection_noop'):
                logits = output_layer(decoder_outputs)

            with tf.variable_scope('loss'):
                with self._colocation_cm(name='noop'):
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tgt_output, logits=logits, name='xentropy')
                    loss = tf.reduce_sum(losses) / tf.to_float(losses.shape[0])

        return logits, loss


def nmt(src_input, tgt_input, tgt_output, nmt_colocation, unroll=True, **kwargs):
    """Returns a NMT network."""
    hparams = create_nmt_hparams(**kwargs)
    return NMT(hparams, nmt_colocation, unroll).build_graph(
        src_input, tgt_input, tgt_output)


class SeparateCellNMT():
    """NMT graph builder where separate cells are used for each sequence."""

    def __init__(self, hparams, dtype=tf.float32):
        self.hparams = hparams
        self.dtype = dtype

    def create_rnn_cell(self, num_layers, num_residual_layers, base_gpu=0):
        """Wrapper function for model_helper.create_rnn()."""
        return model_helper.create_rnn_cell(
            unit_type=self.hparams.unit_type,
            num_units=self.hparams.num_units,
            num_layers=num_layers,
            forget_bias=self.hparams.forget_bias,
            num_residual_layers=num_residual_layers,
            dropout=self.hparams.dropout,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            num_gpus=self.hparams.num_gpus,
            base_gpu=base_gpu,
            single_cell_fn=None)

    def _build_encoder(self, src_input):
        with tf.variable_scope('encoder'):
            # unstack inputs
            src_input_steps = tf.unstack(src_input, axis=1)

            embedding = tf.get_variable(
                'embedding',
                shape=(self.hparams.src_vocab_size, self.hparams.num_units))

            assert self.hparams.encoder_type == 'uni'
            assert self.hparams.num_residual_layers == 0

            outputs = []
            state = None
            for step, src_input_step in enumerate(src_input_steps):
                with tf.variable_scope('step_%d' % step):
                    # TODO: colocate ops
                    src_input_step = tf.nn.embedding_lookup(
                        embedding, src_input_step)

                    # TODO: colocate ops
                    cell = self.create_rnn_cell(
                        self.hparams.num_layers,
                        self.hparams.num_residual_layers)

                    if state is None:
                        state = cell.get_initial_state(
                            inputs=src_input_step)

                    output, state = cell(src_input_step, state)
                    outputs.append(output)

        return outputs, state

    def _build_decoder(self, tgt_input, encoder_outputs, encoder_state):
        with tf.variable_scope('decoder'):
            # unstack inputs
            tgt_input_steps = tf.unstack(tgt_input, axis=1)

            embedding = tf.get_variable(
                'embedding',
                shape=(self.hparams.tgt_vocab_size, self.hparams.num_units))

            output_layer = tf.layers.Dense(self.hparams.tgt_vocab_size,
                                           name='output_projection')

            outputs = []
            state = encoder_state
            for step, tgt_input_step in enumerate(tgt_input_steps):
                with tf.variable_scope('step_%d' % step):
                    # TODO: colocate ops
                    tgt_input_step = tf.nn.embedding_lookup(
                        embedding, tgt_input_step)

                    # TODO: colocate ops
                    cell = self.create_rnn_cell(
                        self.hparams.num_layers,
                        self.hparams.num_residual_layers)

                    output, state = cell(tgt_input_step, state)

                    # TODO: colocate ops
                    output = output_layer(output)
                    outputs.append(output)

            return tf.stack(outputs, axis=1)

    def build_graph(self, src_input, tgt_input):
        """Returns logits."""
        encoder_outputs, encoder_state = self._build_encoder(src_input)
        decoder_outputs = self._build_decoder(
            tgt_input, encoder_outputs, encoder_state)
        return decoder_outputs


def separate_cell_nmt(src_input, tgt_input, unroll=True, **kwargs):
    """Returns a NMT network."""
    hparams = create_nmt_hparams(**kwargs)
    assert unroll
    return SeparateCellNMT(hparams).build_graph(src_input, tgt_input)
