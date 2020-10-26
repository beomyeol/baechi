# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Runs training."""
from __future__ import absolute_import, division, print_function

import collections
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem

from image_classifier.networks import nets_factory
from nmt import model_factory
from placer import placer_lib, cost as cost_lib
from third_party.grappler import graph_placer as grappler_graph_placer
from utils import logger

tf.app.flags.DEFINE_boolean(
    'log_device_placement', False, 'Logging device placement.')

tf.app.flags.DEFINE_boolean(
    'colocate_grads_with_ops', False, 'Colocate gradient with ops.')

tf.app.flags.DEFINE_enum(
    'optimizer', 'sgd',
    ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'sgd', 'rmsprop'],
    'The name of the optimizer')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_string(
    'logdir', '', 'Path to log dir.')

tf.app.flags.DEFINE_string(
    'cost_path', '/tmp/cost.pkl', 'Path to the cost file.')

tf.app.flags.DEFINE_boolean(
    'costgen', False, 'Generate cost dict.')

tf.app.flags.DEFINE_boolean(
    'only_forward', False, 'Consider only forward ops.')

tf.app.flags.DEFINE_float('memory_fraction', 1.0, 'GPU memory fraction')

tf.app.flags.DEFINE_string(
    'comm_cost_coeffs', '0.0001754,134',
    'Comma-separated linear communication cost function coefficients')

tf.app.flags.DEFINE_float(
    'comm_cost_factor', 1.0, 'Communication cost function factor.')

tf.app.flags.DEFINE_float(
    'cost_factor', 1.0, 'Factor that applies to all costs')

###### Image classifier ######
tf.app.flags.DEFINE_enum(
    'data_format', 'NHWC', ['NHWC', 'NCHW'], 'Image data format')

##### NMT ######
tf.app.flags.DEFINE_integer('vocab_size', 5000, 'Vocabulary size.')
tf.app.flags.DEFINE_integer('max_seq_length', 30, 'Max. sequence length.')
tf.app.flags.DEFINE_integer('rnn_units', 1024, 'RNN units.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'RNN # layers.')
tf.app.flags.DEFINE_enum(
    'rnn_unit_type', 'lstm', ['lstm', 'gru'], 'RNN unit type.')
tf.app.flags.DEFINE_enum(
    'encoder_type', 'bi', ['bi', 'uni', 'gnmt'], 'Encoder type.')
tf.app.flags.DEFINE_boolean(
    'residual', False, 'Add residual connections to RNN.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus for NMT.')
tf.app.flags.DEFINE_boolean('disable_nmt_colocation', False,
                            'Disable the NMT ops colocation.')


##### Grappler ######
tf.app.flags.DEFINE_boolean('grappler', False, 'Use Grappler.')
tf.app.flags.DEFINE_integer(
    'grappler_time', 3600, 'Allotted time in seconds for Grappler.')


_LOGGER = logger.get_logger(__file__)


def _configure_optimizer(optimizer_name, learning_rate):
    """Configures the optimizer used for training.

    Args:
        learning_rate: A scalar or `Tensor` learning rate.

    Returns:
        An instance of an optimizer.

    Raises:
        ValueError: if optimizer_name is not recognized.
    """
    if optimizer_name == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif optimizer_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, name='Momentum')
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError(
            'Optimizer [%s] was not recognized' % optimizer_name)
    return optimizer


def _get_gpu_devices(sess_config):
    with tf.Session(config=sess_config) as sess:
        return [
            {"name": device.name,
             "memory_size": device.memory_limit_bytes,
             "type": device.device_type}
            for device in sess.list_devices()
            if device.device_type == 'GPU']


_NUM_CLASSES = {
    'cifarnet': 10,
    'inception_v3': 1000,
}

ModelSpec = collections.namedtuple('ModelSpec', ['cls', 'image_size'])


def build_image_classifier_model(inputs, model_name, data_format):
    """Builds a image classifier with the given specs."""
    # pylint: disable=too-many-locals
    _LOGGER.info('data format: %s', data_format)

    images, labels = inputs

    num_classes = _NUM_CLASSES[model_name]
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes)

    logits, _ = network_fn(images, data_format=data_format)

    with tf.variable_scope('loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_sum(losses) / tf.to_float(images.shape[0])

    return loss


def build_nmt_model(inputs, model_name, **kwargs):
    """Builds NMT with the given specs."""
    # pylint: disable=too-many-locals
    # log NMT spec.
    _LOGGER.info(', '.join(['{}={}'.format(*item) for item in kwargs.items()]))

    src_input, target_input, target_output = inputs

    vocab_size = kwargs.pop('vocab_size')

    # replicate vocab size
    kwargs['src_vocab_size'] = vocab_size
    kwargs['tgt_vocab_size'] = vocab_size

    model_fn = model_factory.get_model_fn(model_name, **kwargs)
    _, loss = model_fn(src_input, target_input, target_output)

    return loss


def build_model(inputs, model_name, data_format, **kwargs):
    """Builds a model with the given specs."""
    if model_name in _NUM_CLASSES:
        return build_image_classifier_model(inputs, model_name, data_format)

    return build_nmt_model(inputs, model_name, **kwargs)


def run_op(target_op, warmup_count=5, num_measurement=10,
           profile_every_n_steps=None, logdir=None, config=None):
    """Runs the given graph."""
    # pylint: disable=too-many-locals, too-many-arguments
    with tf.Session(config=config) as sess:
        if logdir:
            writer = tf.summary.FileWriter(logdir=logdir,
                                           graph=tf.get_default_graph())
        else:
            writer = None

        sess.run(tf.global_variables_initializer())

        warmup_start_time = time.time()

        for _ in range(warmup_count):
            sess.run(target_op)

        warmup_end_time = time.time()
        _LOGGER.info('Warmup time: %s',
                     str(warmup_end_time - warmup_start_time))

        runtimes = []
        run_metadata_list = []
        for step in range(1, num_measurement + 1):
            if profile_every_n_steps and step % profile_every_n_steps == 0:
                _LOGGER.info('Profiling step %d...', step)
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                sess.run(target_op,
                         options=run_options,
                         run_metadata=run_metadata)
                if writer:
                    writer.add_run_metadata(
                        run_metadata, 'step-{}'.format(step))
                    # pylint: disable=invalid-name
                    metadata_out_path = os.path.join(
                        logdir, 'run_metadata-{}.pbtxt'.format(step))
                    with open(metadata_out_path, 'wb') as f:
                        f.write(run_metadata.SerializeToString())

                run_metadata_list.append(run_metadata)
            else:
                start_time = time.time()
                sess.run(target_op)
                end_time = time.time()
                runtimes.append(end_time - start_time)

        _LOGGER.info('Profile run time: %s',
                     str(time.time() - warmup_end_time))

        avg_step_time = np.average(runtimes)

        _LOGGER.info('Graph execution stats. #samples=%d, median=%s, mean=%s',
                     len(runtimes),
                     np.median(runtimes),
                     np.average(runtimes))

        return avg_step_time, run_metadata_list


def get_costs(target_op, warmup_count=5, num_measurement=50,
              profile_every_n_steps=5, sess_config=None, logdir=None):
    """Generates costs with tf.Session."""
    # pylint: disable=too-many-arguments
    avg_step_time, run_metadata_list = run_op(
        target_op,
        warmup_count=warmup_count,
        num_measurement=num_measurement,
        profile_every_n_steps=profile_every_n_steps,
        logdir=logdir,
        config=sess_config)
    cost_dict = cost_lib.build_cost_dict(run_metadata_list)
    return avg_step_time, cost_dict


def generate_cost(target_op, cost_path, sess_config=None, logdir=None):
    """Generates cost data for the graph at the given path."""
    if not cost_path:
        raise ValueError('cost_path is required.')

    # copy graphdef since get_costs will create init_op.
    graphdef = tf.get_default_graph().as_graph_def()

    start_time = time.time()
    step_time, cost_dict = get_costs(
        target_op, sess_config=sess_config, logdir=logdir)

    _LOGGER.info('Original runtime: %f', step_time)

    cost_dir_path = os.path.dirname(cost_path)
    if cost_dir_path:
        os.makedirs(cost_dir_path, exist_ok=True)
    # pylint: disable=invalid-name
    with open(cost_path, 'wb') as f:
        _LOGGER.info('Saving to %s...', cost_path)
        cost_data = {'graphdef': graphdef,
                     'cost_dict': cost_dict}
        pickle.dump(cost_data, f)

    _LOGGER.info('Profile run costs: %s', str(time.time() - start_time))


def run_placement(target_op, cost_path, comm_cost_coeffs, cost_factor,
                  logdir=None, sess_config=None):
    """Runs the placement."""
    # pylint: disable=too-many-locals

    if not cost_path:
        raise ValueError('cost_path is required.')

    # pylint: disable=invalid-name
    with open(cost_path, 'rb') as f:
        cost_data = pickle.load(f)

    graph = tf.get_default_graph()

    assert cost_data['graphdef'] == graph.as_graph_def()

    devices = _get_gpu_devices(sess_config)

    cost_dict = cost_data['cost_dict']

    # adjust costs for sensitivity experiments.
    if cost_factor != 1.0:
        cost_dict, comm_cost_coeffs = cost_lib.adjust_costs(
            cost_factor, cost_dict, comm_cost_coeffs)

    start_time = time.time()
    placer = placer_lib.get_placer(
        graph,
        devices=devices,
        cost_dict=cost_dict,
        comm_cost_coeffs=comm_cost_coeffs)
    placer.run()
    _LOGGER.info('Entire placement time: %s', str(time.time() - start_time))


def _build_image_classifier_inputs(model_name, batch_size, data_format):
    num_classes = _NUM_CLASSES[model_name]
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=num_classes)

    if data_format == 'NHWC':
        input_shape = (batch_size,
                       network_fn.default_image_size,
                       network_fn.default_image_size,
                       3)
    else:
        input_shape = (batch_size,
                       3,
                       network_fn.default_image_size,
                       network_fn.default_image_size)

    images = np.ones(input_shape, dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int32)

    element = (images, labels)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def _build_nmt_inputs(batch_size, max_seq_length):
    input_shape = (batch_size, max_seq_length)

    src_input = np.ones(input_shape, dtype=np.int32)
    target_input = np.ones(input_shape, dtype=np.int32)
    target_output = np.ones(input_shape, dtype=np.int32)

    element = (src_input, target_input, target_output)

    with tf.variable_scope('dataset'):
        dataset = tf.data.Dataset.from_tensors(element).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


def build_inputs(model_name, batch_size, data_format, max_seq_length):
    """Generates dummy inputs."""
    if model_name in _NUM_CLASSES:
        return _build_image_classifier_inputs(
            model_name, batch_size, data_format)
    return _build_nmt_inputs(batch_size, max_seq_length)


def build_train_op(loss, optimizer_name, learning_rate,
                   colocate_grads_with_ops):
    """Builds a train op."""
    optimizer = _configure_optimizer(optimizer_name, learning_rate)
    grads_and_vars = optimizer.compute_gradients(
        loss, colocate_gradients_with_ops=colocate_grads_with_ops)
    global_step = tf.train.create_global_step()
    return optimizer.apply_gradients(grads_and_vars,
                                     global_step=global_step)


def run_grappler(target_op, allotted_time, logdir, sess_config):
    """Runs Grappler placement."""
    tf.logging.set_verbosity(tf.logging.INFO)

    # need to create a session here with memory fraction.
    # otherwise, memory fraction flag is not correctly set due to a session
    # created by cluster
    with tf.Session(config=sess_config):
        pass

    graph = tf.get_default_graph()

    cluster = gcluster.Cluster()
    metagraph = tf.train.export_meta_graph(graph=graph,
                                           clear_extraneous_savers=True)

    _LOGGER.info('Grappler allotted time: %d', allotted_time)

    placed_metagraph_list = grappler_graph_placer.PlaceGraph(
        metagraph,
        cluster=cluster,
        allotted_time=allotted_time,
        verbose=True,
        sess_config=sess_config,
        gpu_only=True)

    _LOGGER.info('# found metagraph: %d', len(placed_metagraph_list))

    if len(placed_metagraph_list) == 0:
        _LOGGER.info('No feasible placement is found.')
        return

    if logdir:
        metagraph_dir = os.path.join(logdir, 'metagraph')
        os.makedirs(metagraph_dir, exist_ok=True)
        for i, metagraph in enumerate(placed_metagraph_list):
            metagraph_path = os.path.join(
                metagraph_dir, 'metagraph-%d.pbtxt' % i)
            # pylint: disable=invalid-name
            with open(metagraph_path, 'wb') as f:
                f.write(metagraph.SerializeToString())

    # use the last element because it is the best placement that is found.
    placed_metagraph = placed_metagraph_list[-1]

    # assign device placement
    for node in placed_metagraph.graph_def.node:
        tf_op = graph.get_operation_by_name(node.name)
        # pylint: disable=protected-access
        tf_op._set_device(node.device)

    step_time = run_op(
        target_op, warmup_count=10, num_measurement=21,
        profile_every_n_steps=21, logdir=logdir,
        config=sess_config)[0]

    _LOGGER.info('Average runtime: {}'.format(step_time))


def parse_comm_cost_coeffs(coeffs_str, factor=1.0):
    comm_cost_coeffs = coeffs_str.split(',')
    assert len(comm_cost_coeffs) == 2

    comm_cost_coeffs[0] = float(comm_cost_coeffs[0])
    comm_cost_coeffs[1] = int(comm_cost_coeffs[1])

    if factor != 1.0:
        _LOGGER.info('Communication cost factor: %s', str(factor))
        comm_cost_coeffs = tuple(
            [value * factor for value in comm_cost_coeffs])

    return comm_cost_coeffs


def main(unparsed_args):
    """Main function."""
    if len(unparsed_args) > 1:
        raise RuntimeError('Unparsed args: {}'.format(unparsed_args[1:]))

    # pylint: disable=invalid-name
    FLAGS = tf.app.flags.FLAGS
    # pylint: enable=invalid-name

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement)
    if FLAGS.memory_fraction != 1.0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = \
            FLAGS.memory_fraction
    # disable TF optimizer
    sess_config.graph_options.optimizer_options.opt_level = -1
    _LOGGER.debug('Session config: %s', str(sess_config))

    inputs = build_inputs(
        model_name=FLAGS.model_name,
        batch_size=FLAGS.batch_size,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        max_seq_length=FLAGS.max_seq_length,
    )

    # build graph
    loss = build_model(
        inputs=inputs,
        model_name=FLAGS.model_name,
        # image classifier
        data_format=FLAGS.data_format,
        # NMT
        vocab_size=FLAGS.vocab_size,
        rnn_units=FLAGS.rnn_units,
        num_layers=FLAGS.num_layers,
        rnn_unit_type=FLAGS.rnn_unit_type,
        encoder_type=FLAGS.encoder_type,
        residual=FLAGS.residual,
        num_gpus=FLAGS.num_gpus,
        colocation=not FLAGS.disable_nmt_colocation)

    only_forward = FLAGS.only_forward
    _LOGGER.info('Only consider forward ops: %s', str(only_forward))
    colocate_grads_with_ops = FLAGS.colocate_grads_with_ops
    _LOGGER.info('Coloate grads with ops: %s' % str(colocate_grads_with_ops))

    comm_cost_coeffs = parse_comm_cost_coeffs(
        FLAGS.comm_cost_coeffs, FLAGS.comm_cost_factor)

    if only_forward:
        assert colocate_grads_with_ops

    # add to the train op collections to support important ops identification
    tf.add_to_collection(tf.GraphKeys.TRAIN_OP, loss)

    target_op = loss

    if FLAGS.costgen:
        if not only_forward:
            train_op = build_train_op(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op

        generate_cost(target_op,
                      cost_path=FLAGS.cost_path,
                      sess_config=sess_config,
                      logdir=FLAGS.logdir)
    else:
        if not only_forward:
            train_op = build_train_op(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op

        if FLAGS.grappler:
            run_grappler(
                target_op,
                allotted_time=FLAGS.grappler_time,
                logdir=FLAGS.logdir,
                sess_config=sess_config)
            return

        run_placement(
            target_op,
            cost_path=FLAGS.cost_path,
            comm_cost_coeffs=comm_cost_coeffs,
            cost_factor=FLAGS.cost_factor,
            logdir=FLAGS.logdir,
            sess_config=sess_config)

        if only_forward:
            # build train op
            train_op = build_train_op(
                loss,
                optimizer_name=FLAGS.optimizer,
                learning_rate=FLAGS.learning_rate,
                colocate_grads_with_ops=colocate_grads_with_ops)
            target_op = train_op

        step_time = run_op(
            target_op, warmup_count=10, num_measurement=51,
            profile_every_n_steps=51, logdir=FLAGS.logdir,
            config=sess_config)[0]

        _LOGGER.info('Average runtime: {}'.format(step_time))


if __name__ == "__main__":
    tf.app.run(main)
