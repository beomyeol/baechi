# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Placer utility module."""
# pylint: disable=invalid-name
from __future__ import absolute_import, division, print_function

import collections
import concurrent.futures
import functools
import operator
import sys
import os
import itertools

import matplotlib.pyplot as plt
import networkx as nx

from tensorflow.core.framework import attr_value_pb2
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


def humanize_num_bytes(num_bytes):
    """Returns a number of bytes string."""
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while num_bytes >= 1024 and i < len(suffixes)-1:
        num_bytes /= 1024.
        i += 1
    number_str = ('%.2f' % num_bytes).rstrip('0').rstrip('.')
    return '%s%s' % (number_str, suffixes[i])


def print_colocation_group(op_graph, print_cb=print):
    colocation_groups = {}

    for _, op_data in op_graph.nodes.items():
        op_name = op_data['name']
        group_name = op_data['colocation_group']
        if group_name in colocation_groups:
            colocation_groups[group_name].append(op_name)
        else:
            colocation_groups[group_name] = [op_name]

    for group_name in sorted(colocation_groups):
        print_cb('{}: {}'.format(group_name, colocation_groups[group_name]))


class ColocationGroupMap():
    """Data structure that maps group to a set of groups that are co-located."""

    def __init__(self):
        self._map = {}

    def colocate(self, group1, group2):
        prev_group1 = self._map.get(group1, set([group1]))
        prev_group2 = self._map.get(group2, set([group2]))
        new_group = prev_group1 | prev_group2

        # update existing groups
        for group in new_group:
            self._map[group] = new_group

    def __getitem__(self, op):
        return self._map[op]

    def __len__(self):
        return len(self._map)

    def items(self):
        return self._map.items()


def create_colocation_group_to_ops_map(op_graph):
    """Generate a dict that maps a colocation group to its op id list."""
    retval = {}

    for op_id, op_data in op_graph.nodes().items():
        # assume there is only one group
        group = op_data['colocation_group']
        if group in retval:
            retval[group].append(op_id)
        else:
            retval[group] = [op_id]

    return retval


def create_colocation_group_infos(op_graph):
    """Generate a dict that maps a colocation group to its information.

    dict value is another dict that has following entries.
        "ops": a list of op data that are in the colocation group.
        "temp_memory_max": max temp memory of ops in the group.
        "output_memory_max": max output memory of ops in the group.
        "persistent_memory_sum": accumulated persistent memory of ops
                                 in the colocation group.
    """
    colocation_group_infos = {
        group_name: {"ops": [op_graph.nodes[op_id] for op_id in op_ids]}
        for group_name, op_ids
        in create_colocation_group_to_ops_map(op_graph).items()}

    # calculate memory requirement for each group
    for _, group_info in colocation_group_infos.items():
        temp_memory_max = 0
        output_memory_max = 0
        persistent_memory_sum = 0
        output_memory_sum = 0

        for op in group_info["ops"]:
            temp_memory_max = max(temp_memory_max, op["temporary_memory"])
            current_out_memory_sum = sum(op["output_memory"])
            output_memory_max = max(output_memory_max, current_out_memory_sum)
            persistent_memory_sum += op["persistent_memory"]
            output_memory_sum += current_out_memory_sum

        group_info["temp_memory_max"] = temp_memory_max
        group_info["output_memory_max"] = output_memory_max
        group_info["persistent_memory_sum"] = persistent_memory_sum
        group_info["output_memory_sum"] = output_memory_sum

    return colocation_group_infos


def get_ready_ts(op_graph, op_id):
    """Returns a ready timestamp for the given operator."""
    op_data = op_graph.nodes[op_id]
    ready_ts = 0
    in_edges = list(op_graph.in_edges(op_id, data=True))
    for in_edge in in_edges:
        from_op_id, _, edge_data = in_edge
        from_op = op_graph.nodes[from_op_id]
        if from_op["p"] == op_data['p']:
            new_ready_ts = from_op["end_ts"]
        else:
            new_ready_ts = max(
                [tensor['recv_end_ts'] for tensor in edge_data['tensor']])
        ready_ts = max(ready_ts, new_ready_ts)
    return ready_ts


def save_placement_graph(op_graph,
                         filename,
                         figsize=None,
                         font_size=12,
                         with_colocation_group=False):
    """Save placement graph to a file."""
    fig = plt.figure(figsize=figsize)
    # pos = nx.drawing.nx_agraph.graphviz_layout(op_graph)
    # pos = nx.drawing.layout.spring_layout(op_graph)
    pos = nx.drawing.layout.circular_layout(op_graph)
    # pos = nx.drawing.layout.shell_layout(op_graph)

    labels = {}
    for op_id, data in op_graph.nodes(True):
        label = ('\n'.join([data['name'], data['colocation_group']])
                 if with_colocation_group else data['name'])
        labels[op_id] = label

    nx.draw_networkx_labels(op_graph, pos, labels=labels, font_size=font_size)
    nx.draw_networkx_nodes(op_graph, pos, node_color='b')
    nx.draw_networkx_edges(op_graph, pos)
    fig.savefig(filename)


class TimestampOpTuple():
    """A tuple of timestamp and operator."""

    def __init__(self, ts, op):
        self.ts = ts
        self.op = op

    @property
    def op_id(self):
        """Returns the operator id."""
        return self.op['id']

    def __lt__(self, other):
        return ((self.ts, self.op['topo_order']) <
                (other.ts, other.op['topo_order']))

    def __eq__(self, other):
        return self.ts == other.ts and self.op == other.op

    def __repr__(self):
        return 'TimestampOpTuple(ts={}, op={})'.format(self.ts, self.op)


class SortedTimestampOps():
    """Sorted list of timestamp and operator tuples."""

    def __init__(self, *args):
        self._list = list(*args)
        self._sorted = False

    def add_op(self, ts, op):
        """Adds a new op with the given timestamp."""
        self._list.append(TimestampOpTuple(ts, op))
        self._sorted = False

    def __len__(self):
        return len(self._list)

    def remove_op(self, op):
        """Removes the given operator from the list.

        Returns:
            True if the given operator is removed. False, otherwise.
        """
        op_id = op['id']
        for i, ts_op in enumerate(self._list):
            if ts_op.op_id == op_id:
                del self._list[i]
                return True
        return False

    def _sort(self):
        if not self._sorted:
            self._list = sorted(self._list)
            self._sorted = True

    def __getitem__(self, index):
        self._sort()
        return self._list[index]

    def pop(self, index=0):
        """Pops the item at the given index."""
        self._sort()
        return self._list.pop(index)

    def __iter__(self):
        self._sort()
        return self._list.__iter__()


def find_index_of_ts_op_tuple(ts_op_tuples, op_id):
    """Finds the index of the given operator id at the ts and op tuple list."""
    for i, ts_op_tuple in enumerate(ts_op_tuples):
        if ts_op_tuple.op_id == op_id:
            return i
    return -1


def transfer_colocation_group(from_graph, to_graph):
    """Transfers the colocation information in tf.Graph."""
    for tf_op in to_graph.get_operations():
        try:
            from_graph_op = from_graph.get_operation_by_name(tf_op.name)
            tf_op._set_attr(
                "_class",
                attr_value_pb2.AttrValue(
                    list=attr_value_pb2.AttrValue.ListValue(
                        s=from_graph_op.get_attr("_class"))))
        except (KeyError, ValueError):
            _LOGGER.debug('Skipping op=%s', tf_op.name)


def transfer_collections(from_graph, to_graph):
    """Transfers the collection information in tf.Graph."""
    for key in from_graph.get_all_collection_keys():
        for value in from_graph.get_collection(key):
            to_graph.add_to_collection(key, value)


def transfer_colocation_group_graphdef(from_graphdef, to_graphdef):
    """Transfers the colocation information in tf.GraphDef."""
    colocation_groups = {node.name: node.attr['_class']
                         for node in from_graphdef.node
                         if '_class' in node.attr}
    for node in to_graphdef.node:
        if node.name in colocation_groups:
            node.attr['_class'].CopyFrom(colocation_groups[node.name])


def clear_graphdef_device(graphdef):
    """Clears the device fields from the given tf.GraphDef."""
    for node in graphdef.node:
        node.ClearField('device')
    return graphdef


def prune_non_important_ops(op_graph, important_ops):
    """Prunes non important ops.

    Returns:
        a tuple of new placement graph and index.
    """
    op_ids_to_remove = []
    for op_id, op_data in op_graph.nodes.items():
        op_name = op_data['name']
        if op_name not in important_ops:
            op_ids_to_remove.append(op_id)
    _LOGGER.info('# ops to prune: %d', len(op_ids_to_remove))

    op_graph = op_graph.copy()
    op_graph.remove_nodes_from(op_ids_to_remove)
    assert nx.algorithms.components.is_weakly_connected(op_graph)

    important_op_graph = nx.DiGraph()
    important_op_index = {}  # key: op name, value: op id

    # update node and edge ids
    for op_id, (_, op_data) in enumerate(op_graph.nodes.items()):
        op_data['id'] = op_id
        important_op_graph.add_node(op_id, **op_data)
        important_op_index[op_data['name']] = op_id

    edge_id = 0
    for from_op_id, to_op_id, edge_data in op_graph.edges(data=True):
        edge_data['id'] = edge_id
        from_op_name = op_graph.nodes[from_op_id]['name']
        to_op_name = op_graph.nodes[to_op_id]['name']
        important_op_graph.add_edge(
            important_op_index[from_op_name],
            important_op_index[to_op_name],
            **edge_data)
        edge_id += 1
    assert edge_id == op_graph.number_of_edges()

    return important_op_graph, important_op_index


def get_estimated_memory(tensor):
    """Returns estimated memory bytes from tf.Tensor()."""
    if not tensor.shape:
        # unknown shape. TODO: fix
        shape_list = [1]
    else:
        shape_list = tensor.shape.as_list()
        if len(shape_list) == 0:
            shape_list = [1]
        # replace None to 1. TODO: fix.
        shape_list = [shape if shape else 1 for shape in shape_list]
    return tensor.dtype.size * functools.reduce(operator.mul, shape_list)


_PERSISTENT_OPS = ["Variable", "VariableV2", "AutoReloadVariable",
                   "VarHandleOp", "ReadVariableOp"]


def is_persistent_op(tf_op):
    """Returns whether the given tf.Operation is a persistent op."""
    op_type = tf_op.type
    return op_type in _PERSISTENT_OPS


def log_stats(op_graph, device_graph):
    """Logs stats for each device based on the placement results."""
    num_ops = {}
    computation_time = {}
    temp_memory = {}
    persistent_memory = {}
    groups = {}

    for device_id in device_graph:
        num_ops[device_id] = 0
        computation_time[device_id] = 0
        temp_memory[device_id] = 0
        persistent_memory[device_id] = 0
        groups[device_id] = set()

    for _, op_data in op_graph.nodes.items():
        device_id = op_data["p"]
        num_ops[device_id] += 1
        groups[device_id].add(op_data["colocation_group"])
        computation_time[device_id] += op_data["weight"]
        temp_memory[device_id] += op_data["temporary_memory"]
        persistent_memory[device_id] += op_data["persistent_memory"]

    # log stats
    for device_id, device_node in device_graph.nodes.items():
        out_str = "device: %d" % device_id
        out_str += ", # ops: %d" % num_ops[device_id]
        if groups[device_id]:
            out_str += ", # groups: %d" % len(groups[device_id])
        out_str += ", computation time: %d us" % computation_time[device_id]
        out_str += ", temp memory: {}".format(
            humanize_num_bytes(temp_memory[device_id]))
        out_str += ", persistent memory: {}".format(
            humanize_num_bytes(persistent_memory[device_id]))
        if 'peak_memory' in device_node:
            out_str += ", peak memory: {}".format(
                humanize_num_bytes(device_node['peak_memory']))
        _LOGGER.info(out_str)


def assign_topo_order(op_graph):
    """Assigns topological sort order to the nodes in the graph."""
    ready_op_ids = collections.deque()
    for op_id, op_data in op_graph.nodes.items():
        op_data['ready_count'] = 0
        if op_graph.in_degree(op_id) == 0:
            ready_op_ids.append(op_id)

    topo_order = 0
    while ready_op_ids:
        current_op_id = ready_op_ids.popleft()
        current_op = op_graph.nodes[current_op_id]
        if 'topo_order' in current_op:
            raise RuntimeError('Cycle exists')
        current_op['topo_order'] = topo_order
        topo_order += 1

        for _, next_op_id in op_graph.out_edges(current_op_id):
            next_op = op_graph.nodes[next_op_id]
            next_op['ready_count'] += 1
            if next_op['ready_count'] == op_graph.in_degree(next_op_id):
                ready_op_ids.append(next_op_id)

    assert topo_order == op_graph.number_of_nodes(), \
        'topo_order=%d, #nodes=%d' % (topo_order, op_graph.number_of_nodes())


def transfer_placement(from_op_graph, to_op_graph):
    """Transfers the device placement between placement graphs."""
    for op_id, p in from_op_graph.nodes.data('p'):
        to_op_graph.nodes[op_id]['p'] = p


def generate_op_run_log(op_data):
    """Returns operator execution log."""
    return 'device id={} op={}, ready={}, start={}, end={}\n'.format(
        op_data['p'], op_data['name'], op_data['ready_ts'],
        op_data['start_ts'], op_data['end_ts'])


def generate_memcpy_log(
        tensor_data, from_op, to_op, from_device_id, to_device_id, cached):
    """Generates a memcpy log."""
    # pylint: disable=too-many-arguments
    return ('memcpy tensor={}, device {}->{}, cached={}, from={}, to={}, '
            'send_start={}, recv_start={}, recv_end={}, cost={}\n'.format(
                tensor_data['name'], from_device_id, to_device_id, cached,
                from_op['name'], to_op['name'], tensor_data['send_start_ts'],
                tensor_data['recv_start_ts'], tensor_data['recv_end_ts'],
                tensor_data['weight']))


def process_finished_op(op_graph, op_id):
    """Processes the finished op and returns next ops to be ready to execute.
    """
    ready_ops = []
    for _, next_op_id in op_graph.out_edges(op_id):
        next_op = op_graph.nodes[next_op_id]
        next_op['ready_count'] += 1
        in_degree = op_graph.in_degree(next_op_id)
        if next_op['ready_count'] == in_degree:
            # ready to run
            ready_ops.append(next_op)
    return ready_ops


class ReadyOpManager():
    """Ready operator manager."""

    def __init__(self, op_graph, devices, log_file=None, parallel_exec=False):
        self._op_graph = op_graph
        self._devices = devices
        self._log_file = log_file
        self._ready_ops = []

        if parallel_exec:
            max_workers = os.cpu_count()
            _LOGGER.info(
                'Using parallel execution. # worker threads: %d', max_workers)
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers)
            self._chunk_size = 50
        else:
            self.executor = None

    def _estimate_tensor_transfer_end_ts(
            self, tensor_data_list, send_device, recv_device,
            send_op, recv_op, dry_run):
        """Calculates the estimated tensor transfer end timestamp."""
        # pylint: disable=too-many-locals,too-many-arguments
        end_ts = -1
        tensors_to_send = []

        for tensor_data in tensor_data_list:
            cached_tensor = recv_device.get_cached_tensor(tensor_data['name'])
            if cached_tensor is None:
                tensors_to_send.append(tensor_data)
            else:
                if not dry_run:
                    # update tensor transfer info
                    tensor_data.update(cached_tensor)
                    if self._log_file:
                        self._log_file.write(generate_memcpy_log(
                            tensor_data, send_op, recv_op,
                            send_device.id, recv_device.id, True))
                end_ts = max(end_ts, cached_tensor['recv_end_ts'])

        if len(tensors_to_send) > 0:
            if dry_run:
                # generate merged tensor
                # cannot send them one by one because channel time
                # will not be updated.
                tensor_names = []
                tensor_weights = []
                for tensor in tensors_to_send:
                    tensor_names.append(tensor['name'])
                    tensor_weights.append(tensor['weight'])
                merged_tensor_data = {
                    'name': ','.join(tensor_names),
                    'weight': sum(tensor_weights),
                }
                send_start_ts, _ = send_device.send_tensor(
                    merged_tensor_data, send_op["end_ts"], dry_run=True)
                _, recv_end_ts = recv_device.recv_tensor(
                    merged_tensor_data, dry_run=True,
                    send_start_ts=send_start_ts)
                end_ts = max(end_ts, recv_end_ts)
            else:
                # send tensors one by one
                for tensor_data in tensors_to_send:
                    send_start_ts, _ = send_device.send_tensor(
                        tensor_data, send_op["end_ts"], dry_run=False)
                    _, recv_end_ts = recv_device.recv_tensor(
                        tensor_data, dry_run=False,
                        send_start_ts=send_start_ts)
                    if self._log_file:
                        self._log_file.write(generate_memcpy_log(
                            tensor_data, send_op, recv_op,
                            send_device.id, recv_device.id, False))
                    end_ts = max(end_ts, recv_end_ts)

            # mark that there is a data transfer between devices.
            recv_op['data_transfer'][recv_device.id].add(send_device.id)

        assert end_ts > 0
        return end_ts

    def get_ready_ts(self, op_data, device, dry_run=False):
        """Returns a ready timestamp for the operator on the device."""
        if not device.is_placeable(op_data):
            return None

        if dry_run and device.id in op_data['ready_tss']:
            return op_data['ready_tss'][device.id]

        # store a list of devices that send tensors to the device for
        #   calculating ready_ts on this device.
        # this information is used for invalidating ready ops.
        op_data['data_transfer'][device.id] = set()

        ready_ts = 0
        in_edges = list(self._op_graph.in_edges(op_data['id'], data=True))
        for in_edge in in_edges:
            from_op_id, _, edge_data = in_edge
            from_op = self._op_graph.nodes[from_op_id]
            from_op_device = self._devices[from_op['p']]
            if from_op_device.id == device.id:
                new_ready_ts = from_op["end_ts"]
            else:
                new_ready_ts = self._estimate_tensor_transfer_end_ts(
                    edge_data['tensor'],
                    send_device=from_op_device, recv_device=device,
                    send_op=from_op, recv_op=op_data, dry_run=dry_run)
            ready_ts = max(ready_ts, new_ready_ts)

        op_data['ready_tss'][device.id] = ready_ts
        return ready_ts

    def get_schedule_ts(self, op_data, device):
        """Returns a schedule-able timestamp for the operator on the device."""
        if 'p' in op_data and op_data['p'] != device.id:
            # get schedule-able ts only for the assigned device.
            return None
        ready_ts = self.get_ready_ts(op_data, device, dry_run=True)
        if ready_ts is not None:
            return max(ready_ts, device.next_available_ts)
        return None

    def get_schedule_tss(self, op_data):
        """Return a schedule-able ts dict for the operator on all devices."""
        retval = {}
        for device in self._devices.values():
            schedule_ts = self.get_schedule_ts(op_data, device)
            if schedule_ts is not None:
                retval[device.id] = schedule_ts
        return retval

    def _populate_schedule_tss(self, ready_op):
        schedule_tss = self.get_schedule_tss(ready_op)
        ready_op['schedule_tss'] = schedule_tss
        ready_op['urgent_ts'] = (max(schedule_tss.values())
                                 if schedule_tss else sys.maxsize)

    def _divide_into_chunks(self, iterable):
        iterator = iter(iterable)
        chunk = list(itertools.islice(iterator, self._chunk_size))
        while chunk:
            yield chunk
            chunk = list(itertools.islice(iterator, self._chunk_size))

    @staticmethod
    def _process_chunk(fn, chunk):
        for item in chunk:
            fn(item)

    def populate_schedule_tss(self):
        """Populates schedule_tss and urgent_ts of all operators."""
        if self.executor is None:
            for ready_op in self._ready_ops:
                self._populate_schedule_tss(ready_op)
        else:
            futures = [
                self.executor.submit(
                    self._process_chunk, self._populate_schedule_tss, chunk)
                for chunk in self._divide_into_chunks(self._ready_ops)]
            concurrent.futures.wait(futures)

    @staticmethod
    def _initialize_op(op_data):
        op_data['ready_tss'] = {}
        op_data['data_transfer'] = {}

    def add(self, op_data):
        """Adds the operator into the ready operator list."""
        self._initialize_op(op_data)
        return self._ready_ops.append(op_data)

    @staticmethod
    def _invalidate_ready_ts(ready_op, changed_devices):
        for dev_id, send_dev_ids in ready_op['data_transfer'].items():
            if len(send_dev_ids) == 0:
                # no data transfer. Need not invalidate ready_ts
                continue
            need_update = (dev_id in changed_devices
                           or len(send_dev_ids & changed_devices) > 0)
            if need_update:
                # remove ready_ts for this operator on the device.
                # ready_ts will be recalculated.
                ready_op['ready_tss'].pop(dev_id, None)

    def remove(self, op_data):
        """Removes the operator from the ready operator list."""
        # if this op has data transfer, other ops that uses the same channel
        #   need to update their ready tss
        changed_devices = set()
        target_device_id = op_data['p']
        changed_devices |= op_data['data_transfer'][target_device_id]
        changed_devices.add(target_device_id)

        if self.executor is None:
            for ready_op in self._ready_ops:
                self._invalidate_ready_ts(ready_op, changed_devices)
        else:
            futures = [
                self.executor.submit(
                    self._process_chunk,
                    functools.partial(self._invalidate_ready_ts,
                                      changed_devices=changed_devices),
                    chunk)
                for chunk in self._divide_into_chunks(self._ready_ops)]
            concurrent.futures.wait(futures)

        return self._ready_ops.remove(op_data)

    def extend(self, iterable):
        """Extends the ready operator list with the given iterable object."""
        for op_data in iterable:
            self._initialize_op(op_data)
        return self._ready_ops.extend(iterable)

    def __len__(self):
        return len(self._ready_ops)

    def __iter__(self):
        return iter(self._ready_ops)


def prune_dangling_ops(op_graph):
    """Prunes the dangling operators from the graph."""
    new_op_graph = nx.DiGraph()
    new_op_index = {}

    new_op_id = 0
    for op_id, op_data in op_graph.nodes.items():
        if op_graph.in_degree(op_id) == 0 and op_graph.out_degree(op_id) == 0:
            _LOGGER.info('skipping dangling operator: %s', op_data['name'])
            continue
        op_data['id'] = new_op_id
        new_op_graph.add_node(new_op_id, **op_data)
        new_op_index[op_data['name']] = new_op_id
        new_op_id += 1

    edge_id = 0
    for from_op_id, to_op_id, edge_data in op_graph.edges(data=True):
        edge_data['id'] = edge_id
        from_op_name = op_graph.nodes[from_op_id]['name']
        to_op_name = op_graph.nodes[to_op_id]['name']
        new_op_graph.add_edge(
            new_op_index[from_op_name],
            new_op_index[to_op_name],
            **edge_data)
        edge_id += 1
    assert edge_id == op_graph.number_of_edges()

    return new_op_graph, new_op_index
