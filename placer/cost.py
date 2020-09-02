"""Cost module."""
from __future__ import absolute_import, division, print_function

import collections
import re

import numpy as np

from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)

Cost = collections.namedtuple(
    'Cost',
    ['compute_cost', 'output_memory', 'temporary_memory', 'persistent_memory'])


def convert_op_perfs_to_cost_dict(op_perfs):
    """Converts the OpPerformanceData list into the cost dict."""
    cost_dict = {}
    for op_perf in op_perfs:
        op_memory = op_perf.op_memory
        cost_dict[op_perf.node] = Cost(
            # the unit of OpPerformace Data's compute cost is nanosecond
            compute_cost=int(op_perf.compute_cost // 1e3),
            output_memory=list(op_memory.output_memory),
            temporary_memory=op_memory.temp_memory,
            persistent_memory=op_memory.persistent_memory)
    assert len(cost_dict) == len(op_perfs), 'Duplicates exist'
    return cost_dict


def convert_cost_graph_to_cost_dict(cost_graph):
    """Converts CostGraphDef into the cost dict. """
    cost_dict = {}
    for node in cost_graph.node:
        if node.name == '_SOURCE' or node.name == '_SINK':
            continue
        assert node.name not in cost_dict
        cost_dict[node.name] = Cost(
            compute_cost=node.compute_cost,
            output_memory=[
                output_info.size for output_info in node.output_info],
            temporary_memory=node.temporary_memory_size,
            persistent_memory=node.persistent_memory_size)
    return cost_dict


def convert_cost_graphs_to_cost_dict(cost_graphs):
    """Converts a list of CostGraphDef into the cost dict. """
    aggr_cost_dict = {}
    for cost_graph in cost_graphs:
        for name, cost in convert_cost_graph_to_cost_dict(cost_graph).items():
            if name in aggr_cost_dict:
                aggr_cost = aggr_cost_dict[name]
                if cost.output_memory != aggr_cost.output_memory:
                    _LOGGER.warn(
                        'different output memory. name=%s, prev=%s, new=%s',
                        name,
                        str(cost.output_memory),
                        str(aggr_cost.output_memory))
                if cost.temporary_memory != aggr_cost.temporary_memory:
                    _LOGGER.warn(
                        'different temp memory. name=%s, prev=%d, new=%d',
                        name,
                        cost.temporary_memory,
                        aggr_cost.temporary_memory)
                assert cost.persistent_memory == aggr_cost.persistent_memory
                aggr_cost.compute_cost.append(cost.compute_cost)
            else:
                aggr_cost_dict[name] = Cost(
                    compute_cost=[cost.compute_cost],
                    output_memory=cost.output_memory,
                    temporary_memory=cost.temporary_memory,
                    persistent_memory=cost.persistent_memory)

    return {name: Cost(compute_cost=np.median(cost.compute_cost),
                       output_memory=cost.output_memory,
                       temporary_memory=cost.temporary_memory,
                       persistent_memory=cost.persistent_memory)
            for name, cost in aggr_cost_dict.items()}


def build_cost_dict(run_metadata_list):
    """Build a cost dict from a list of tf.RunMetadata protobuf."""
    aggr_cost_dict = {}
    for run_metadata in run_metadata_list:
        cost_dict = build_cost_dict_from_step_stats(run_metadata.step_stats)
        for name, cost in cost_dict.items():
            if name in aggr_cost_dict:
                aggr_cost = aggr_cost_dict[name]
                aggr_cost['compute_cost'].append(cost.compute_cost)
                # use max output memory
                aggr_cost['output_memory'] = [
                    max(prev, new) for prev, new
                    in zip(aggr_cost['output_memory'], cost.output_memory)]
                aggr_cost['temporary_memory'] = max(
                    aggr_cost['temporary_memory'], cost.temporary_memory)
                aggr_cost['persistent_memory'] = max(
                    aggr_cost['persistent_memory'], cost.persistent_memory)
            else:
                aggr_cost_dict[name] = {
                    'compute_cost': [cost.compute_cost],
                    'output_memory': cost.output_memory,
                    'temporary_memory': cost.temporary_memory,
                    'persistent_memory':  cost.persistent_memory,
                }

    for raw_cost in aggr_cost_dict.values():
        raw_cost['compute_cost'] = int(np.median(raw_cost['compute_cost']))

    return {
        name: Cost(**raw_cost) for name, raw_cost in aggr_cost_dict.items()}


def adjust_costs(cost_factor, cost_dict, comm_cost_coeffs):
    _LOGGER.info('Adjusting costs. factor=%f', cost_factor)

    new_cost_dict = {
        name: Cost(
            compute_cost=int(cost.compute_cost * cost_factor),
            output_memory=cost.output_memory,
            temporary_memory=cost.temporary_memory,
            persistent_memory=cost.persistent_memory)
        for name, cost in cost_dict.items()}

    new_comm_cost_coeffs = [coeff * cost_factor for coeff in comm_cost_coeffs]
    new_comm_cost_coeffs[1] = int(new_comm_cost_coeffs[1])

    return new_cost_dict, new_comm_cost_coeffs


def build_cost_dict_from_step_stats(step_stats):
    """Build a cost dict from tf.StepStats protobuf."""
    # Reference: tensflow's BuildCostModel() in step_stats_collector.cc.

    # skip operator that ends /_\d+
    # these operators are duplicates that acts like the original ops
    #   due to the communication.
    def _is_skip_node(name):
        if name == '_SOURCE' or re.match(r".+/_\d+$", name):
            _LOGGER.debug("Skipping node name=%s" % name)
            return True
        return False

    raw_cost_dict = {}

    def _add_to_cost_dict(op_name, key, value, update_existing=False):
        if op_name in raw_cost_dict:
            stat_dict = raw_cost_dict[op_name]
            if key in stat_dict:
                assert update_existing
                stat_dict[key] += value
            else:
                stat_dict[key] = value
        else:
            raw_cost_dict[op_name] = {key: value}

    def _calculate_memory_size(tensor_desc):
        return tensor_desc.allocation_description.allocated_bytes

    # parse hardware stats
    hardware_stats = None
    gpu_dev_stats_list = []
    for dev_stats in step_stats.dev_stats:
        device = dev_stats.device
        last_device_string = device.split('/')[-1]
        # expected string example:
        #   "/device:GPU:0/stream:all" (hardware stats)
        #   "/job:localhost/replica:0/task:0/device:GPU:0" (normal stats)
        if last_device_string == 'stream:all':
            hardware_stats = dev_stats
        elif 'device:' in last_device_string:
            gpu_dev_stats_list.append(dev_stats)
        else:
            _LOGGER.debug('Skipping stats for device: %s', device)

    if hardware_stats is not None:
        for node_stats in hardware_stats.node_stats:
            # stream:all has only compute cost information
            name = node_stats.node_name.split(":")[0]
            if _is_skip_node(name):
                continue
            _add_to_cost_dict(
                name, 'compute_cost', node_stats.op_end_rel_micros,
                update_existing=True)

    for dev_stats in gpu_dev_stats_list:
        for node_stats in dev_stats.node_stats:
            name = node_stats.node_name
            if _is_skip_node(name):
                continue

            # compute cost
            # if hardware stats is available, skip compute cost
            if name not in raw_cost_dict:
                _add_to_cost_dict(
                    name, 'compute_cost', node_stats.op_end_rel_micros)

            # output memory
            output_memory = [None] * len(node_stats.output)
            for output in node_stats.output:
                output_memory[output.slot] = \
                    _calculate_memory_size(output.tensor_description)
            _add_to_cost_dict(name, 'output_memory', output_memory)

            # memory stats
            memory_stats = node_stats.memory_stats
            _add_to_cost_dict(name, 'temporary_memory',
                              memory_stats.temp_memory_size)
            _add_to_cost_dict(name, 'persistent_memory',
                              memory_stats.persistent_memory_size)

    # convert raw cost dict to immutable cost dict
    return {name: Cost(**raw_cost) for name, raw_cost in raw_cost_dict.items()
            if len(raw_cost) == 4}
