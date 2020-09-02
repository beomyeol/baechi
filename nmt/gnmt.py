"""GNMT model."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from nmt import nmt, attention_nmt, model_helper
from utils import logger

_LOGGER = logger.get_logger(__file__)


class GNMT(attention_nmt.AttentionNMT):
    """GNMT model builder."""
    # pylint: disable=too-few-public-methods

    def _build_all_encoder_layers(self, bi_encoder_outputs, num_uni_layers,
                                  num_uni_residual_layers):
        uni_cell = self.create_rnn_cell(
            num_uni_layers, num_uni_residual_layers, base_gpu=1)
        encoder_outputs, encoder_state = self.rnn(
            uni_cell, bi_encoder_outputs, self.unroll, dtype=self.dtype)
        return encoder_outputs, encoder_state

    def _build_encoder(self, src_input):
        if self.hparams.encoder_type != 'gnmt':
            raise ValueError('GNMT only support gnmt encoder_type.')

        # Build GNMT encoder.
        with tf.variable_scope('encoder'):
            num_bi_layers = 1
            num_uni_layers = self.hparams.num_layers - num_bi_layers
            num_uni_residual_layers = (
                self.hparams.num_residual_layers - num_bi_layers)
            _LOGGER.info("Build a GNMT encoder")
            _LOGGER.info("  num_bi_layers = %d", num_bi_layers)
            _LOGGER.info("  num_uni_layers = %d", num_uni_layers)

            encoder_emb_input = self.build_embedding(
                src_input,
                vocab_size=self.hparams.src_vocab_size,
                embedding_dim=self.hparams.num_units,
                colocation=self.colocation)

            if self.unroll:
                encoder_emb_input = tf.unstack(encoder_emb_input, axis=1)

            _LOGGER.info("Build bidirectional layers.")
            res = self._build_bidirectional_rnn(
                encoder_emb_input,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=0,  # no residual connection
            )
            bi_encoder_outputs, bi_encoder_state = res

            _LOGGER.info("Build unidirectional layers.")
            encoder_outputs, encoder_state = self._build_all_encoder_layers(
                bi_encoder_outputs, num_uni_layers, num_uni_residual_layers)

            # Pass all encoder states to the decoder except
            # the first bi-directional layer
            encoder_state = (bi_encoder_state[1],) + (
                (encoder_state,) if num_uni_layers == 1 else encoder_state)

            if self.unroll:
                device_str = model_helper.get_device_str(
                    self.hparams.num_layers - 1,
                    self.hparams.num_gpus)
                with tf.device(device_str):
                    encoder_outputs = tf.stack(encoder_outputs, axis=1)

        return encoder_outputs, encoder_state

    def _build_decoder_cell(self, encoder_outputs, encoder_state):
        _LOGGER.info("Build a GNMT decoder cell")
        attention_option = "bahdanau"
        attention_mechanism = self.create_attention_fn(
            attention_option, self.hparams.num_units, memory=encoder_outputs,
            colocation=self.colocation)

        # pylint: disable=protected-access
        cell_list = model_helper._cell_list(
            unit_type=self.hparams.unit_type,
            num_units=self.hparams.num_units,
            num_layers=self.hparams.num_layers,
            num_residual_layers=self.hparams.num_residual_layers - 1,
            forget_bias=self.hparams.forget_bias,
            dropout=self.hparams.dropout,
            num_gpus=self.hparams.num_gpus,
            colocation=self.colocation,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            base_gpu=0,
            single_cell_fn=None,
            residual_fn=gnmt_residual_fn,
        )

        # Only wrap the bottom layer with the attention mechanism.
        attention_cell = cell_list.pop(0)

        attention_cell = attention_nmt.AttentionWrapper(
            attention_cell,
            attention_mechanism,
            attention_layer_size=None,  # don't use attention layer.
            output_attention=False,
            colocation=self.colocation,
            name="attention")

        if self.attention_architecture == "gnmt":
            cell = GNMTAttentionMultiCell(
                attention_cell, cell_list)
        elif self.attention_architecture == "gnmt_v2":
            cell = GNMTAttentionMultiCell(
                attention_cell, cell_list, use_new_attention=True)
        else:
            raise ValueError(
                "Unknown attention_architecture %s" % (
                    self.attention_architecture))

        # pass hidden state
        batch_size = encoder_outputs.shape[0]
        decoder_initial_state = tuple(
            zs.clone(cell_state=es)
            if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
            for zs, es in zip(
                cell.zero_state(batch_size, self.dtype), encoder_state))

        return cell, decoder_initial_state


def gnmt_residual_fn(inputs, outputs):
    """Residual function that handles different inputs and outputs inner dims.
    Args:
      inputs: cell inputs, this is actual inputs concatenated with
              the attention vector.
      outputs: cell outputs
    Returns:
      outputs + actual inputs
    """
    def split_input(inp, out):
        out_dim = out.get_shape().as_list()[-1]
        inp_dim = inp.get_shape().as_list()[-1]
        return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)
    actual_inputs, _ = tf.contrib.framework.nest.map_structure(
        split_input, inputs, outputs)

    def assert_shape_match(inp, out):
        inp.get_shape().assert_is_compatible_with(out.get_shape())
    tf.contrib.framework.nest.assert_same_structure(actual_inputs, outputs)
    tf.contrib.framework.nest.map_structure(
        assert_shape_match, actual_inputs, outputs)
    return tf.contrib.framework.nest.map_structure(
        lambda inp, out: inp + out, actual_inputs, outputs)


class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
    """A MultiCell with GNMT attention style."""

    def __init__(self, attention_cell, cells, use_new_attention=False):
        """Creates a GNMTAttentionMultiCell.
        Args:
          attention_cell: An instance of AttentionWrapper.
          cells: A list of RNNCell wrapped with AttentionInputWrapper.
          use_new_attention: Whether to use the attention generated from
                             current step bottom layer's output.
                             Default is False.
        """
        cells = [attention_cell] + cells
        self.use_new_attention = use_new_attention
        super(GNMTAttentionMultiCell, self).__init__(
            cells, state_is_tuple=True)

    def __call__(self, inputs, state, scope=None):
        """Run the cell with bottom layer's attention copied to all uppers."""
        if not tf.contrib.framework.nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s"
                % (len(self.state_size), state))

        with tf.variable_scope(scope or "multi_rnn_cell"):
            new_states = []

            with tf.variable_scope("cell_0_attention"):
                attention_cell = self._cells[0]
                attention_state = state[0]
                cur_inp, new_attention_state = attention_cell(
                    inputs, attention_state)
                new_states.append(new_attention_state)

            for i in range(1, len(self._cells)):
                with tf.variable_scope("cell_%d" % i):

                    cell = self._cells[i]
                    cur_state = state[i]

                    if self.use_new_attention:
                        cur_inp = tf.concat(
                            [cur_inp, new_attention_state.attention], -1)
                    else:
                        cur_inp = tf.concat(
                            [cur_inp, attention_state.attention], -1)

                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, tuple(new_states)


def gnmt(src_input, tgt_input, tgt_output, colocation, unroll=True, **kwargs):
    """Returns logits for GNMT."""
    hparams = nmt.create_nmt_hparams(**kwargs)
    return GNMT(hparams, colocation, unroll, 'gnmt').build_graph(
        src_input, tgt_input, tgt_output)


def gnmt_v2(src_input, tgt_input, tgt_output, colocation, unroll=True,
            **kwargs):
    """Returns logits for GNMT v2."""
    hparams = nmt.create_nmt_hparams(**kwargs)
    return GNMT(hparams, colocation, unroll, 'gnmt_v2').build_graph(
        src_input, tgt_input, tgt_output)
