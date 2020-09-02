"""NMT model helper functions."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from utils import logger

_LOGGER = logger.get_logger(__file__)


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


class ColocationWrapper(tf.nn.rnn_cell.RNNCell):
    """Colocates RNN cell operations."""

    def __init__(self, cell):
        super(ColocationWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        name_scope_str = type(self).__name__ + "ZeroState"
        with tf.name_scope(name_scope_str, values=[batch_size]):
            with tf.colocate_with(tf.no_op('noop')):
                return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        if self._cell.built:
            with tf.colocate_with(tf.no_op(type(self).__name__ + '_noop')):
                return self._cell(inputs, state, scope=scope)
        else:
            return self._cell(inputs, state, scope=scope)


def _single_cell(unit_type, num_units, forget_bias, dropout, colocation, mode,
                 residual_connection=False, device_str=None, residual_fn=None,
                 cudnn=False):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    # Cell Type
    log_str = " "
    if unit_type == "lstm":
        if cudnn:
            log_str += "  CudnnLSTM"
            single_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
                num_units)
        else:
            log_str += "  LSTM, forget_bias=%g" % forget_bias
            single_cell = tf.contrib.rnn.LSTMBlockCell(
                num_units,
                forget_bias=forget_bias)
    elif unit_type == "gru":
        if cudnn:
            log_str += " CudnnGRU"
            single_cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(
                num_units)
        else:
            log_str += " GRU"
            single_cell = tf.contrib.rnn.GRUBlockCellV2(num_units)
    elif unit_type == "layer_norm_lstm":
        log_str += "  Layer Normalized LSTM, forget_bias=%g" % forget_bias
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)
    elif unit_type == "nas":
        log_str += "  NASCell"
        single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))
        log_str += "  %s, dropout=%g " % (type(single_cell).__name__, dropout)

    # Residual
    if residual_connection:
        single_cell = tf.nn.rnn_cell.ResidualWrapper(
            single_cell, residual_fn=residual_fn)
        log_str += "  %s" % type(single_cell).__name__

    if colocation:
        single_cell = ColocationWrapper(single_cell)
        log_str += " %s" % (type(single_cell).__name__)

    # Device Wrapper
    if device_str:
        single_cell = tf.nn.rnn_cell.DeviceWrapper(single_cell, device_str)
        log_str += "  %s, device=%s" % (type(single_cell).__name__, device_str)

    _LOGGER.info(log_str)

    return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, colocation, mode, num_gpus, base_gpu=0,
               single_cell_fn=None, residual_fn=None):
    """Create a list of RNN cells."""
    if not single_cell_fn:
        single_cell_fn = _single_cell

    # Multi-GPU
    cell_list = []
    for i in range(num_layers):
        _LOGGER.info("cell %d", i)
        single_cell = single_cell_fn(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            colocation=colocation,
            mode=mode,
            residual_connection=(i >= num_layers - num_residual_layers),
            device_str=get_device_str(i + base_gpu, num_gpus),
            residual_fn=residual_fn
        )
        cell_list.append(single_cell)

    return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, colocation, mode, num_gpus,
                    base_gpu=0, single_cell_fn=None):
    """Create multi-layer RNN cell.

    Args:
      unit_type: string representing the unit type, i.e. "lstm".
      num_units: the depth of each unit.
      num_layers: number of cells.
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
        cells in the returned list will be wrapped with `ResidualWrapper`.
      forget_bias: the initial forget bias of the RNNCell(s).
      dropout: floating point value between 0.0 and 1.0:
        the probability of dropout.  this is ignored if `mode != TRAIN`.
      mode: either tf.contrib.learn.TRAIN/EVAL/INFER
      num_gpus: The number of gpus to use when performing round-robin
        placement of layers.
      base_gpu: The gpu device id to use for the first RNN cell in the
        returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
        as its device id.
      single_cell_fn: allow for adding customized cell.
        When not specified, we default to model_helper._single_cell
    Returns:
      An `RNNCell` instance.
    """
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           colocation=colocation,
                           mode=mode,
                           num_gpus=num_gpus,
                           base_gpu=base_gpu,
                           single_cell_fn=single_cell_fn)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.nn.rnn_cell.MultiRNNCell(cell_list)
