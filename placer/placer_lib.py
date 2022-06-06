# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Placer module."""
# pylint: disable=too-many-lines,invalid-name
from __future__ import absolute_import, division, print_function

import copy
import functools
import json
import operator
import random
import time
from collections import deque

import networkx as nx
from future.utils import bytes_to_native_str
import numpy as np

import tensorflow as tf
from placer import adjuster as adjuster_lib
from placer import grouper as grouper_lib
from placer import placer_utils
from placer.m_etf import m_etf
from placer.m_sct import m_sct
from placer.m_topo import m_topo
from placer.virtual_scheduler import VirtualScheduler
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem
from utils import logger

FLAGS = tf.app.flags.FLAGS
_LOGGER = logger.get_logger(__file__, level=logger.INFO)


tf.app.flags.DEFINE_enum("placement_method", "m_sct",
                         ["m_sct", "m_sct_reserve", "m_topo",
                          "m_topo_nonuniform", "m_etf", "m_etf_reserve"],
                         "Placement method for placer.")

tf.app.flags.DEFINE_enum("placer_type", "default",
                         ["default", "colocation", "fusion"],
                         "Placer type.")

tf.app.flags.DEFINE_boolean(
    "only_important_ops", False, "Use only important ops for the placement.")

tf.app.flags.DEFINE_float(
    "placer_memory_fraction", 1.0,
    "Device memory fraction that is used by the placer.")

##### Placement graph building #####
tf.app.flags.DEFINE_string(
    "stats_path", "./stats.log",
    "Profiling result path to use for placement.")
tf.app.flags.DEFINE_string(
    "device_info_path", "./device_info.json",
    "Path to the JSON file where device information is stored.")

##### Logging ######
tf.app.flags.DEFINE_boolean(
    "log_placer_device_placement", False,
    "Log the device placement in the placement graph.")
tf.app.flags.DEFINE_boolean(
    "log_colocation_graph", False,
    "Log a graph consisting of colocation groups")

##### Colocation group placer #####
tf.app.flags.DEFINE_boolean(
    "resolve_cycle", False,
    "Resolve a cycle if exists by removing a single edge.")
tf.app.flags.DEFINE_boolean(
    "consider_all_edges_in_grouping", False,
    "Using all edges in generating all groups")

##### Fused op placer #####
tf.app.flags.DEFINE_boolean(
    "fusion_check_disjoint_paths", False,
    "Enable disjoint path check to find more fusion chances")

tf.app.flags.DEFINE_boolean(
    "fusion_allow_cycle", False, "Allow cycles in operator fusion.")

##### SCT flags #####
tf.app.flags.DEFINE_float(
    "sct_threshold", 0.1,
    "Threshold to transform relaxed SCT solutions to integers.")


def get_comm_cost(tensor_size, coef, intercept):
    """Returns communication cost (in microsecs) for the given tensor size."""
    return int(round(coef * tensor_size + intercept))


def get_raw_colocation_group(tf_op):
    """Returns a raw string-typed co-location group of the given TF op."""
    return [bytes_to_native_str(colocation_group)
            for colocation_group in tf_op.colocation_groups()]


def read_stats():
    """Reads stats from --stats_path."""
    assert tf.gfile.Exists(FLAGS.stats_path), \
        "Not exist profiling stats path: " + FLAGS.stats_path

    stats = {}
    with open(FLAGS.stats_path, "r") as f:
        for line in f:
            name, data = line.split(": ", 1)
            stats[name] = json.loads(data)

    assert stats, "No stats"
    return stats


class PlacementGraphGenerator():
    """Placement graph generator."""
    # pylint: disable=too-few-public-methods

    def __init__(self, tf_graph, stats, comm_cost_fn, only_important_ops,
                 is_cost_dict):
        # pylint: disable=too-many-arguments
        """
        Args:
            tf_graph: TensorFlow graph.
            stats: stats dict that maps op name to cost dict or stats dict.
            comm_cost_fn: a function that returns comm. cost for tensor size.
            only_important_ops: flags to use only important ops.
            is_cost_dict: flags for stats type.
        """
        self._tf_graph = tf_graph
        self._stats = stats
        self._comm_cost_fn = comm_cost_fn
        self._only_important_ops = only_important_ops
        self._is_cost_dict = is_cost_dict

    def _add_nodes(self, op_graph):
        """Adds operator nodes to the given op graph.

        Returns:
            a dict that maps from a operator name to an index in the graph.
        """
        # pylint: disable=too-many-locals

        # the weight of each node is the computation time
        op_index = {}  # key: op name, value: index in op_graph
        tf_only_ops = set()
        num_fixed_durations = 0
        num_fixed_output_memory = 0
        stats = dict(self._stats)  # generate a copy

        for i, tf_op in enumerate(self._tf_graph.get_operations()):
            if tf_op.name in stats:
                data = stats[tf_op.name]
                if self._is_cost_dict:
                    duration = data.compute_cost
                    output_memory = list(data.output_memory)
                    temporary_memory = data.temporary_memory
                    persistent_memory = data.persistent_memory
                else:
                    duration = data["dur"]
                    output_memory = [output["requested_bytes"]
                                     for output in data["outputs"]]
                    # TODO: set temporary memory if possible
                    temporary_memory = 0
                    # TODO: fix this. all output memory is not persistent
                    persistent_memory = sum(output_memory)

                # each operator should have computer cost > 0
                if duration <= 0:
                    _LOGGER.debug("Change duration of op=%s from %d to 1",
                                  tf_op.name, duration)
                    duration = 1
                    num_fixed_durations += 1

                # output memory should be zero or a positive value
                for port, out_memory in enumerate(output_memory):
                    if out_memory <= 0:
                        _LOGGER.debug(
                            "Change output memory of %s:%d from %d to 0",
                            tf_op.name,
                            port,
                            output_memory[port])
                        output_memory[port] = 0
                        num_fixed_output_memory += 1

                if len(tf_op.outputs) != len(output_memory):
                    raise RuntimeError(
                        "Different # outputs for {}. tf#={} stats#={}".format(
                            tf_op.name, len(tf_op.outputs),
                            len(output_memory)))

                del stats[tf_op.name]
            else:
                _LOGGER.debug("op: %s does not have stats", tf_op.name)
                tf_only_ops.add(tf_op.name)
                # TODO: is this okay?
                duration = 1
                output_memory = [placer_utils.get_estimated_memory(output)
                                 for output in tf_op.outputs]
                temporary_memory = persistent_memory = 0
                if placer_utils.is_persistent_op(tf_op):
                    persistent_memory = sum(output_memory)

            op_graph.add_node(i,
                              weight=duration,
                              name=tf_op.name,
                              id=i,
                              temporary_memory=temporary_memory,
                              persistent_memory=persistent_memory,
                              output_memory=output_memory,
                              colocation_group=get_raw_colocation_group(tf_op))

            _LOGGER.debug("New node added. %d[%s] colocation_group=%s",
                          i, tf_op.name, op_graph.nodes[i]["colocation_group"])

            op_index[tf_op.name] = i

        _LOGGER.info("# ops in the trace are not used: %s", len(stats))
        _LOGGER.info("# ops only in TF: %d", len(tf_only_ops))

        _LOGGER.debug("Ops in the trace are not used: %s", stats.keys())
        _LOGGER.debug("Ops only in TF: %s", tf_only_ops)

        _LOGGER.info("# fixed operator durations: %d", num_fixed_durations)
        _LOGGER.info("# fixed output memory: %d", num_fixed_output_memory)

        return op_index

    def _add_edges(self, op_graph, op_index):
        """Adds edges between operators to the graph."""
        i = 0
        for tf_op in self._tf_graph.get_operations():
            op_id = op_index[tf_op.name]
            for input_tensor in tf_op.inputs:
                input_op_name, index = input_tensor.name.split(":")
                index = int(index)

                input_op_id = op_index[input_op_name]
                memory = op_graph.nodes[input_op_id]["output_memory"][index]

                # the weight is the communication time.
                weight = self._comm_cost_fn(memory)

                edge_data = {
                    "id": i,
                    "weight": weight,
                    "tensor": [{
                        "name": input_tensor.name,
                        "weight": weight,
                        "num_bytes": memory,
                    }],
                }

                if op_graph.has_edge(input_op_id, op_id):
                    # merge edge data only if a new tensor is transferred.
                    prev_edge_data = op_graph[input_op_id][op_id]
                    exists = False
                    for prev_edge_tensor in prev_edge_data['tensor']:
                        if prev_edge_tensor['name'] == input_tensor.name:
                            exists = True
                            break
                    if not exists:
                        prev_edge_data['weight'] += edge_data['weight']
                        prev_edge_data['tensor'] += edge_data['tensor']
                else:
                    op_graph.add_edge(input_op_id, op_id, **edge_data)

                    _LOGGER.debug("New edge added. %d[%s] -> %d[%s] %s",
                                  input_op_id,
                                  input_tensor.name,
                                  op_id,
                                  tf_op.name,
                                  edge_data)
                    i += 1

            # use 4 bytes for control edges
            weight = self._comm_cost_fn(4)
            for control_input in tf_op.control_inputs:
                control_input_op_id = op_index[control_input.name]
                edge_data = {
                    "id": i,
                    "weight": weight,
                    "tensor": [{
                        # control tensor starts with ^
                        "name": "^%s" % control_input.name,
                        "weight": weight,
                        "num_bytes": 4,
                    }],
                }

                assert not op_graph.has_edge(control_input_op_id, op_id)

                op_graph.add_edge(control_input_op_id, op_id, **edge_data)
                _LOGGER.debug("New edge added. %d[%s] -> %d[%s] %s",
                              control_input_op_id,
                              control_input.name,
                              op_id,
                              tf_op.name,
                              edge_data)
                i += 1

        assert i == op_graph.number_of_edges()

    def run(self):
        """Generates a placement graph.

        Returns:
            a tuple of placement graph and a dictionary that maps an operator
            name to a node index in the placement graph.
        """
        op_graph = nx.DiGraph()
        op_index = self._add_nodes(op_graph)
        self._add_edges(op_graph, op_index)

        op_graph, op_index = placer_utils.prune_dangling_ops(op_graph)

        if self._only_important_ops:
            metagraph = tf.train.export_meta_graph(
                graph=self._tf_graph,
                clear_extraneous_savers=True)
            item = gitem.Item(metagraph, ignore_colocation=False)
            important_ops = item.IdentifyImportantOps()
            _LOGGER.info("Use only important ops. # important ops=%d",
                         len(important_ops))
            op_graph, op_index = placer_utils.prune_non_important_ops(
                op_graph, important_ops)

        # assign topo order
        placer_utils.assign_topo_order(op_graph)

        return op_graph, op_index


def create_placement_graph(tf_graph, stats, comm_cost_fn, only_important_ops,
                           is_cost_dict=False):
    """Create a placement graph for the given TF graph.

    Args:
        tf_graph: TensorFlow graph.
        stats: stats dictionary that maps op name to cost dict or stats dict.
        comm_cost_fn: a function that returns comm. cost for tensor size.
        only_important_ops: flags to use only important ops.
        is_cost_dict: flags for stats type.

    Returns:
        a tuple of placement graph and a dictionary that maps an operator name
        to a node index in the placement graph.
    """
    generator = PlacementGraphGenerator(
        tf_graph, stats, comm_cost_fn, only_important_ops, is_cost_dict)
    return generator.run()


def create_device_graph(devices):
    """Creates a placement graph for devices and network.

    Args:
        devices: device information list
    """
    graph = nx.Graph()

    for device_id, device_info in devices.items():
        graph.add_node(device_id,
                       id=device_id,
                       name=device_info["name"],
                       size=0,
                       memory_limit=device_info["memory_size"])

    for i in graph.nodes:
        for j in graph.nodes:
            # TODO: should this be added?
            if i == j:
                graph.add_edge(i, i, weight=0)
            else:
                graph.add_edge(i, j, weight=1)
    return graph


def get_local_devices(cluster=None):
    """Returns a list of available local devices."""
    if cluster:
        devices = cluster.ListDevices()
    else:
        cluster = gcluster.Cluster()
        devices = cluster.ListDevices()
        cluster.Shutdown()

    return [{"name": named_device.name,
             "memory_size": named_device.properties.memory_size,
             "type": named_device.properties.type}
            for named_device in devices]


def is_xla_devices(named_device):
    """Returns true if the given named device is a XLA device.

    Args:
        named_device: device_properties_pb2.NamedDevice
    """
    return "XLA_" in named_device.name


class Placer():
    """Abstract Placer."""

    def run(self):
        """Runs placement."""
        # pylint: disable=no-self-use
        return NotImplementedError()


class DefaultPlacer(Placer):
    """Default placer."""
    # pylint: disable=too-many-instance-attributes,too-many-arguments

    def __init__(self, tf_graph, devices, method, comm_cost_fn,
                 only_important_ops, memory_fraction=1.0,
                 log_device_placement=None, cost_factor=1.0,
                 cost_dict=None, grouper=None, adjustment_method=None,
                 adjustment_with_memory_limit=None, sct_threshold=None,
                 **kwargs):
        # pylint: disable=too-many-locals, unused-argument
        self.tf_graph = tf_graph
        self.devices = devices
        self.comm_cost_fn = comm_cost_fn
        self.cost_dict = dict(cost_dict)
        self.grouper = grouper
        self._memory_fraction = memory_fraction
        self._log_device_placement = log_device_placement

        _LOGGER.info("Placement method: %s" % method)
        self.placement_fn = self._initialize_placement_fn(
            method, sct_threshold=sct_threshold)

        # build placement graph
        if self.cost_dict is None:
            stats = read_stats()
            self.op_graph, self.op_index = create_placement_graph(
                self.tf_graph, stats, self.comm_cost_fn, only_important_ops,
                is_cost_dict=False)
        else:
            self.op_graph, self.op_index = create_placement_graph(
                self.tf_graph, self.cost_dict, self.comm_cost_fn,
                only_important_ops, is_cost_dict=True)

        # adjust cost
        if cost_factor != 1.0:
            diff = abs(cost_factor - 1.0)
            min_factor = 1.0 - diff
            max_factor = 1.0 + diff
            _LOGGER.info('Adjusting costs. factor range [%.2f..%.2f]', min_factor, max_factor)
            for _, op_data in self.op_graph.nodes.items():
                op_data["weight"] *= np.random.uniform(min_factor, max_factor)

            for u, v, edge_data in self.op_graph.edges.data():
                factor = np.random.uniform(min_factor, max_factor)
                edge_data["weight"] *= factor

                for tensor in edge_data["tensor"]:
                    tensor["weight"] *= factor

        # run grouper
        grouper = grouper_lib.get_grouper(self.grouper)
        grouper(self.op_graph)

        # build device graph
        self.device_graph = create_device_graph(self.devices)

        # adjuster
        self._adjuster = adjuster_lib.get_adjuster(
            adjustment_method, self.op_graph, self.device_graph,
            adjustment_with_memory_limit=adjustment_with_memory_limit)
        _LOGGER.info("Adjusting the placement: %s",
                     self._adjuster.__class__.__name__)
        if method.endswith('_colocation'):
            assert isinstance(self._adjuster, adjuster_lib.NoopAdjuster), \
                'Adjustment is unnecessary for the placement with colocation.'

        required_memory = self._calculate_memory_requirement(self.op_graph)
        required_temp_memory = required_memory[0]
        required_persistent_memory = required_memory[1]
        required_output_memory = required_memory[2]
        _LOGGER.info(
            "Total required memory: temp=%s, persistent=%s, output=%s",
            placer_utils.humanize_num_bytes(required_temp_memory),
            placer_utils.humanize_num_bytes(required_persistent_memory),
            placer_utils.humanize_num_bytes(required_output_memory))

        if memory_fraction != 1.0:
            _LOGGER.info('Memory Fraction for the placement: %s',
                         str(memory_fraction))
            for device_id, device_node in self.device_graph.nodes.items():
                device_node['memory_limit'] = int(
                    memory_fraction * device_node['memory_limit'])
                _LOGGER.info('device:%d memory=%d',
                             device_id, device_node['memory_limit'])

    @staticmethod
    def _initialize_placement_fn(method, sct_threshold):
        if method == "m_topo":
            placement_fn = functools.partial(m_topo, colocation=True)
        elif method == "m_topo_nonuniform":
            placement_fn = functools.partial(
                m_topo, colocation=True, uniform=False)
        elif method == "m_sct":
            placement_fn = functools.partial(
                m_sct, threshold=sct_threshold, colocation=True)
        elif method == "m_sct_reserve":
            placement_fn = functools.partial(
                m_sct, threshold=sct_threshold, colocation=True,
                reserve_memory=True)
        elif method == "m_etf":
            placement_fn = functools.partial(m_etf, colocation=True)
        elif method == "m_etf_reserve":
            placement_fn = functools.partial(
                m_etf, colocation=True, reserve_memory=True)
        else:
            raise ValueError("Unsupported placement method: %s" % method)
        return placement_fn

    @staticmethod
    def _calculate_memory_requirement(op_graph):
        """Returns the memory sum of the ops in the placement graph."""
        temp_memory_list = [op_data["temporary_memory"]
                            for _, op_data in op_graph.nodes.items()]
        persistent_memory_list = [op_data["persistent_memory"]
                                  for _, op_data in op_graph.nodes.items()]
        output_memory_list = [sum(op_data["output_memory"])
                              for _, op_data in op_graph.nodes.items()]
        return (functools.reduce(operator.add, temp_memory_list),
                functools.reduce(operator.add, persistent_memory_list),
                functools.reduce(operator.add, output_memory_list))

    def place_ops(self):
        """Places tf ops in the graph by following the placement decision."""
        for tf_op in self.tf_graph.get_operations():
            # tf_op may not be in the op_index if it is not an important op
            if tf_op.name in self.op_index:
                op_id = self.op_index[tf_op.name]
                device_id = self.op_graph.nodes[op_id]["p"]
                # pylint: disable=protected-access
                tf_op._set_device(self.devices[device_id]["name"])

    def run_adjuster(self):
        """Adjusts placement results to hold the colocation rule."""
        self.op_graph = self._adjuster.run()

    def run_placement_algorithm(self):
        """Runs the placement algorithms over operators."""
        _LOGGER.info("Operator placement")
        start_time = time.time()
        self.op_graph = self.placement_fn(self.op_graph, self.device_graph)
        elapsed_time = time.time() - start_time
        _LOGGER.info('placement_time=%s, #ops=%d',
                     str(elapsed_time), self.op_graph.number_of_nodes())

    def run(self):
        """Runs placement algorithm and places ops by following the results."""
        self.run_placement_algorithm()
        self.run_adjuster()

        # simulate execution
        scheduler = VirtualScheduler(self.op_graph, self.device_graph)
        scheduler.initialize()
        makespan = scheduler.run()
        _LOGGER.info("Estimated makespan: %f sec", makespan / 1e6)

        # print stats
        placer_utils.log_stats(self.op_graph, self.device_graph)

        if self._log_device_placement:
            for _, data in self.op_graph.nodes().items():
                _LOGGER.info("[placement] %s: %d", data["name"], data["p"])

        self.place_ops()

        return self.tf_graph


class ColocationGraphPlacer(DefaultPlacer):
    """Placer that runs on the colocation-based merged placement graph.

    This creates a new placement graph where ops are merged based on colocation
    groups.
    """

    def __init__(self, *args, **kwargs):
        super(ColocationGraphPlacer, self).__init__(*args, **kwargs)

        # generate a colocation graph
        colocation_op_graph, colocation_op_index = \
            self._create_colocation_graph(self.op_graph)
        self._find_cycle_and_save_figures(colocation_op_graph)
        self.colocation_op_graph = colocation_op_graph
        self.colocation_op_index = colocation_op_index

        if FLAGS.log_colocation_graph:
            _LOGGER.info("Colocation group graph")
            for node_id in nx.topological_sort(colocation_op_graph):
                inputs = [colocation_op_graph.nodes[from_op_id]["name"]
                          for from_op_id, _
                          in colocation_op_graph.in_edges(node_id)]
                node_data = colocation_op_graph.nodes[node_id]

                _LOGGER.info(
                    "name=%s, weight=%d, memory=%d, ops=%s, inputs=%s",
                    node_data["name"],
                    node_data["weight"],
                    node_data["memory"],
                    node_data["op_names"],
                    inputs)

    @staticmethod
    def _create_colocation_graph(op_graph):
        """Creates a new simulation graph by considering the colocation rule.

        This will create a new logical large op node for colocated op nodes
        by simply gathering stats of them.
        """
        def _should_ignore(op1_node, op2_node):
            # Not adding an edge that connects to the optimizer through
            # control dependency to break the cycle.
            # TODO: Can the name rule "control_dependency" -> "Apply*"
            # be applied to any graph?
            op1_name = op1_node["name"]
            op2_name = op2_node["name"]
            if "control_dependency" in op1_name and "Apply" in op2_name:
                # Ignore this edge to break the cycle.
                _LOGGER.info("Ignore an edge from %s[%s] to %s[%s].",
                             op1_name,
                             op1_node["colocation_group"],
                             op2_name,
                             op2_node["colocation_group"])
                return True
            else:
                return False

        new_graph = nx.DiGraph()
        new_index = {}

        i = 0
        for op_id in op_graph:
            op_node = op_graph.nodes[op_id]
            colocation_group = op_node["colocation_group"]
            if colocation_group in new_index:
                target_node = new_graph.nodes[new_index[colocation_group]]
                # Update the existing node
                # sum up the computation time by assuming that a device runs
                # operators sequentially.
                target_node["weight"] += op_node["weight"]
                target_node["memory"] += op_node["memory"]
                target_node["op_names"].append(op_node["name"])
                _LOGGER.debug("Node updated. %s", str(target_node))
            else:
                new_graph.add_node(i,
                                   name=colocation_group,
                                   weight=op_node["weight"],
                                   memory=op_node["memory"],
                                   op_names=[op_node["name"]],
                                   id=i)
                new_index[colocation_group] = i
                _LOGGER.debug("New node added. %d: %s",
                              i, str(new_graph.nodes[i]))
                i += 1

        # add edges
        i = 0
        for op1_id, op2_id in op_graph.edges:
            edge = op_graph[op1_id][op2_id]
            op1_node = op_graph.nodes[op1_id]
            op2_node = op_graph.nodes[op2_id]
            group1 = op1_node["colocation_group"]
            group2 = op2_node["colocation_group"]
            group1_id = new_index[group1]
            group2_id = new_index[group2]
            if group1_id == group2_id:
                # Ignore edges within the same colocation group
                continue

            if new_graph.has_edge(group1_id, group2_id):
                # Update the existing edge
                target_edge = new_graph[group1_id][group2_id]
                target_edge["weight"] += edge["weight"]
                _LOGGER.debug("Edge updated. %d[%s] -> %d[%s] %s",
                              group1_id,
                              group1,
                              group2_id,
                              group2,
                              str(target_edge))
            else:
                if not FLAGS.consider_all_edges_in_grouping:
                    if _should_ignore(op1_node, op2_node):
                        continue

                new_graph.add_edge(group1_id, group2_id,
                                   weight=edge["weight"], id=i)
                _LOGGER.debug("New edge added. %d[%s] -> %d[%s] %s",
                              group1_id,
                              group1,
                              group2_id,
                              group2,
                              str(new_graph[group1_id][group2_id]))
                i += 1

        return new_graph, new_index

    def _find_cycle_and_save_figures(self, colocation_op_graph):
        """Finds a cycle in the colocation op graph.

        If a cycle exists, save the cycle in the graph and
        also corresponding ops as figures. Then, raise ValueError.
        Otherwise, just return.
        """
        try:
            nodes_in_cycle = [u for u, _ in nx.find_cycle(colocation_op_graph)]
            _LOGGER.info("Cycle: %s",
                         str([colocation_op_graph.nodes[u]["name"]
                              for u in nodes_in_cycle]))

            if FLAGS.resolve_cycle:
                op1_id = nodes_in_cycle[0]
                op2_id = nodes_in_cycle[1]
                _LOGGER.info(
                    "Removing %s -> %s to remove a cycle",
                    colocation_op_graph.nodes[op1_id]["name"],
                    colocation_op_graph.nodes[op2_id]["name"])
                colocation_op_graph.remove_edge(op1_id, op2_id)
                self._find_cycle_and_save_figures(colocation_op_graph)
            else:
                placer_utils.save_placement_graph(
                    colocation_op_graph.subgraph(nodes_in_cycle),
                    "cycle_graph.eps",
                    figsize=(20, 20),
                    font_size=10)

                op_ids_in_cycle = set()
                for colocation_op_id in nodes_in_cycle:
                    colocation_op = colocation_op_graph.nodes[colocation_op_id]
                    for op_name in colocation_op["op_names"]:
                        op_ids_in_cycle.add(self.op_index[op_name])
                placer_utils.save_placement_graph(
                    self.op_graph.subgraph(op_ids_in_cycle),
                    "cycle_op_graph.eps",
                    figsize=(20, 20),
                    font_size=7,
                    with_colocation_group=True)
                raise ValueError("Cycle exists in the placement graph")

        except nx.NetworkXNoCycle:
            _LOGGER.info("No cycle exists")

    def run_placement_algorithm(self):
        """Runs the placement algorithm over colocation groups."""
        _LOGGER.info("Colocation group level placement")
        self.colocation_op_graph = self.placement_fn(self.colocation_op_graph,
                                                     self.device_graph)
        # assign ops to devices based on the colocation group placement results
        for _, op_data in self.op_graph.nodes.items():
            group = op_data["colocation_group"]
            group_index = self.colocation_op_index[group]
            group_op_data = self.colocation_op_graph.nodes[group_index]

            op_data["p"] = group_op_data["p"]


class FusedOpPlacer(DefaultPlacer):
    """Placer that runs on the placement graph consisting of fused ops.

    This creates a new graph by merging ops that are in the same group and
    directly connected in the placement graph
    """

    def __init__(self, *args, **kwargs):
        super(FusedOpPlacer, self).__init__(*args, **kwargs)
        self.fused_op_graph = self._generate_fused_op_graph(
            self.op_graph,
            kwargs['fusion_check_disjoint_paths'],
            kwargs['fusion_allow_cycle'])

    @staticmethod
    def _add_fused_edge(fused_op_graph, from_op_id, to_op_id, edge_data):
        """Adds an edge to the fused op graph.

        Returns:
            True if a new edge is added. False, otherwise.
        """
        if fused_op_graph.has_edge(from_op_id, to_op_id):
            # update existing edge
            prev_edge = fused_op_graph[from_op_id][to_op_id]
            prev_edge_tensors = [prev_edge_tensor['name'] for prev_edge_tensor
                                 in prev_edge['tensor']]
            tensors_to_add = [
                tensor_data for tensor_data in edge_data['tensor']
                if tensor_data['name'] not in prev_edge_tensors]
            for tensor_data in tensors_to_add:
                prev_edge['weight'] += tensor_data['weight']
                prev_edge['tensor'].append(tensor_data)
            return False

        fused_op_graph.add_edge(from_op_id, to_op_id, **edge_data)
        return True

    @staticmethod
    def _assign_new_ids(fused_op_graph):
        """Returns a new graph that of which nodes have unique sequential ids.
        """
        new_fused_op_graph = nx.DiGraph()
        fused_id_map = {}  # maps ids in op_graph to new ids in fused_op_graph
        num_fused_ops = 0

        for new_id, (old_id, data) in enumerate(fused_op_graph.nodes.items()):
            # update id information
            data["old_id"] = old_id
            data["id"] = new_id
            new_fused_op_graph.add_node(new_id, **data)
            fused_id_map[old_id] = new_id

            # log fused op information
            if "fused_ops" in data:
                num_fused_ops += len(data["fused_ops"])
                fused_ops_list = [
                    "%s[%d]" % (fused_op_data["name"], fused_op_data["id"])
                    for fused_op_data in data["fused_ops"]]
                _LOGGER.debug("[FusedOp] %s[%d], fused_ops=%s",
                              data["name"], old_id, str(fused_ops_list))

        for new_id, (u, v, data) in enumerate(fused_op_graph.edges(data=True)):
            data["old_id"] = data["id"]
            data["id"] = new_id
            new_fused_op_graph.add_edge(
                fused_id_map[u], fused_id_map[v], **data)

        _LOGGER.info("# fused ops: %d", num_fused_ops)

        return new_fused_op_graph

    @staticmethod
    def _generate_fused_op_graph(
            op_graph, fusion_check_disjoint_paths, allow_cycle=False):
        """Generates a fused op graph.

        This first identifies ops that can be fused.
        When ops are in the same colocation group and they are directly
        connected, we fuse two ops into a single op.
        """
        # pylint: disable=too-many-locals,too-many-branches
        _LOGGER.info("Allow cycle in operator fusion: %s", str(allow_cycle))

        fused_op_graph = copy.deepcopy(op_graph)

        group_to_ops = placer_utils.create_colocation_group_to_ops_map(
            fused_op_graph)

        for ops in group_to_ops.values():
            internal_edges = deque(
                [(u, v) for u, v in fused_op_graph.edges(ops)
                 if u in ops and v in ops])

            while len(internal_edges) > 0:
                op1, op2 = internal_edges.popleft()

                # internal edge might be connected to a fused op
                if not fused_op_graph.has_edge(op1, op2):
                    continue

                # check whether there is another path from op1 to op2
                if not allow_cycle and (fused_op_graph.out_degree(op1) > 1
                                        and fused_op_graph.in_degree(op2) > 1):
                    if fusion_check_disjoint_paths:
                        # CAVEATS: finding disjoint paths may take long time
                        paths = list(
                            nx.node_disjoint_paths(fused_op_graph, op1, op2))
                        if len(paths) > 1:
                            # ops cannot be fused since it will create a cycle
                            continue
                    else:
                        # skip this fusion due to potential cycle generation
                        continue

                # fuse op2 into op1
                op1_data = fused_op_graph.nodes[op1]
                op2_data = fused_op_graph.nodes[op2]

                _LOGGER.debug("%s[%d] is fused into %s[%d]",
                              op2_data["name"], op2, op1_data["name"], op1)

                op1_data["weight"] += op2_data["weight"]
                op1_data["persistent_memory"] += op2_data["persistent_memory"]
                # use max since each op runs at a time, not simultaneously
                op1_data["temporary_memory"] = max(
                    op1_data["temporary_memory"], op2_data["temporary_memory"])

                # add op2's edges to op1
                for in_edge in fused_op_graph.in_edges(op2, data=True):
                    from_op, _, edge_data = in_edge
                    if from_op == op1:
                        continue
                    if FusedOpPlacer._add_fused_edge(
                            fused_op_graph, from_op, op1, edge_data):
                        # if the new edge is a candidate,
                        # check it by adding it to internal_edges
                        if from_op in ops:
                            internal_edges.append((from_op, op1))

                for out_edge in fused_op_graph.out_edges(op2, data=True):
                    _, to_op, edge_data = out_edge
                    if to_op == op1:
                        continue
                    if FusedOpPlacer._add_fused_edge(
                            fused_op_graph, op1, to_op, edge_data):
                        # if the new edge is a candidate,
                        # check it by adding it to internal_edges
                        if to_op in ops:
                            internal_edges.append((op1, to_op))

                # op2 might be a fused op. merge information
                new_fused_ops = op1_data.get("fused_ops", [])
                new_fused_ops += op2_data.pop("fused_ops", [])
                new_fused_ops += [op2_data]
                op1_data["fused_ops"] = new_fused_ops

                fused_op_graph.remove_node(op2)
                ops.remove(op2)

                # update output memory
                # CAVEATS: output port number is no longer valid.
                output_tensors = {}
                for out_edge in fused_op_graph.out_edges(op1, data=True):
                    out_edge_data = out_edge[-1]
                    for tensor_data in out_edge_data['tensor']:
                        output_tensors[tensor_data['name']] = \
                            tensor_data['num_bytes']
                op1_data['output_memory'] = list(output_tensors.values())

        # need to assign new node ids and edge ids to be compatible with m_sct.
        # m_sct expects op_graph that has consecutive node ids and edge ids.
        return FusedOpPlacer._assign_new_ids(fused_op_graph)

    def run_placement_algorithm(self):
        """Runs the placement algorithm over the graph with fused ops."""
        _LOGGER.info("Fused op placement")

        start_time = time.time()
        fused_op_graph = self.placement_fn(self.fused_op_graph,
                                           self.device_graph)
        elapsed_time = time.time() - start_time
        _LOGGER.info('placement_time=%s, #ops=%d',
                     str(elapsed_time), self.fused_op_graph.number_of_nodes())

        # assign device placement results to op_graph
        for _, op_data in fused_op_graph.nodes.items():
            # fused op has a new id, so use old_id to get the id at op_graph
            self.op_graph.nodes[op_data["old_id"]]["p"] = op_data["p"]
            for fused_op_data in op_data.get("fused_ops", []):
                # ids in fused_ops are op_graph's ids
                fused_op_id = fused_op_data["id"]
                self.op_graph.nodes[fused_op_id]["p"] = op_data["p"]


def get_placer(tf_graph, devices, comm_cost_coeffs, placement_method=None,
               placer_type=None, cost_dict=None, grouper=None,
               adjustment_method=None, adjustment_with_memory_limit=None,
               only_important_ops=None, memory_fraction=None,
               log_device_placement=None, fusion_check_disjoint_paths=None,
               fusion_allow_cycle=None, sct_threshold=None,
               cost_factor=None):
    """Get placer."""
    placement_method = placement_method or FLAGS.placement_method

    comm_cost_coef, comm_cost_intercept = comm_cost_coeffs
    _LOGGER.info("Communication cost function: {} * x + {}".format(
        comm_cost_coef, comm_cost_intercept))
    comm_cost_fn = functools.partial(get_comm_cost,
                                     coef=comm_cost_coef,
                                     intercept=comm_cost_intercept)

    placer_type = placer_type or FLAGS.placer_type
    if placer_type == "default":
        placer_cls = DefaultPlacer
    elif placer_type == "colocation":
        placer_cls = ColocationGraphPlacer
    elif placer_type == "fusion":
        placer_cls = FusedOpPlacer

    # extract ids from names and generate a dict
    def get_id(device_name):
        return int(device_name[device_name.rfind(":") + 1:])
    for device in devices:
        device["id"] = get_id(device["name"])
    devices = {device["id"]: device for device
               in sorted(devices, key=lambda device: device["id"])}

    _LOGGER.info("Devices: %s", str(devices))

    if only_important_ops is None:
        only_important_ops = FLAGS.only_important_ops

    if log_device_placement is None:
        log_device_placement = FLAGS.log_placer_device_placement

    memory_fraction = memory_fraction or FLAGS.placer_memory_fraction

    if fusion_check_disjoint_paths is None:
        fusion_check_disjoint_paths = FLAGS.fusion_check_disjoint_paths

    sct_threshold = sct_threshold or FLAGS.sct_threshold

    if fusion_allow_cycle is None:
        fusion_allow_cycle = FLAGS.fusion_allow_cycle

    return placer_cls(
        tf_graph, devices, placement_method, comm_cost_fn, only_important_ops,
        cost_dict=cost_dict, grouper=grouper,
        adjustment_method=adjustment_method,
        adjustment_with_memory_limit=adjustment_with_memory_limit,
        memory_fraction=memory_fraction,
        log_device_placement=log_device_placement,
        fusion_check_disjoint_paths=fusion_check_disjoint_paths,
        fusion_allow_cycle=fusion_allow_cycle,
        sct_threshold=sct_threshold,
        cost_factor=cost_factor)


def run_random_placement(tf_graph, devices, ignore_colocation=True):
    """Places the operators in tf.Graph over the devices randomly."""
    _LOGGER.info('Run the random placement. #devices=%d, ignore_colocation=%s',
                 len(devices), str(ignore_colocation))
    stats = {}
    if ignore_colocation:
        for tf_op in tf_graph.get_operations():
            device_name = devices[random.randrange(len(devices))]['name']
            stats[device_name] = stats.get(device_name, 0) + 1
            # pylint: disable=protected-access
            tf_op._set_device(device_name)
    else:
        op_graph, _ = create_placement_graph(
            tf_graph, stats={}, comm_cost_fn=lambda x: x,
            only_important_ops=False, is_cost_dict=True)
        tf_grouper = grouper_lib.TFColocationGrouper()
        op_graph = tf_grouper(op_graph)
        group_to_ops_map = placer_utils.create_colocation_group_to_ops_map(
            op_graph)
        for op_ids in group_to_ops_map.values():
            device_name = devices[random.randrange(len(devices))]['name']
            for op_id in op_ids:
                op_name = op_graph.nodes[op_id]['name']
                tf_op = tf_graph.get_operation_by_name(op_name)
                # pylint: disable=protected-access
                tf_op._set_device(device_name)
            stats[device_name] = stats.get(device_name, 0) + len(op_ids)

    _LOGGER.info('Random placement result.')
    for device_name, num_ops in stats.items():
        _LOGGER.info('device=%s, # ops=%d', device_name, num_ops)
