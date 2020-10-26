# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Topological sort based placement."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from placer import placer_utils as utils
from placer import device as device_wrapper
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class Topo():
    """Topological sort based placement.

    Places operator until the device get full."""

    def __init__(self, op_graph, device_graph):
        self.op_graph = op_graph
        self.device_graph = device_graph
        self._device_wrapper_cls = device_wrapper.DeviceWrapper
        self._devices = {
            device_id: self._device_wrapper_cls(
                device_id, self.device_graph, self.op_graph, False)
            for device_id in self.device_graph}
        topo_order_id_tuples = sorted([
            (topo_order, op_id) for op_id, topo_order
            in self.op_graph.nodes(data='topo_order')
        ])
        self._sorted_op_ids = [op_id for _, op_id in topo_order_id_tuples]

    def initialize(self):
        """Initializes."""
        for _, op_data in self.op_graph.nodes.items():
            op_data['memory'] = self._device_wrapper_cls.get_op_memory(op_data)
            op_data['executed_out_count'] = 0

    def is_feasible(self, op_data, device):
        """Returns whether the operator placement on the device is feasible."""
        return device.is_placeable(op_data)

    def _place_ops(self):
        current_op_index = 0
        for _, device in sorted(self._devices.items()):
            while current_op_index < len(self._sorted_op_ids):
                op_id = self._sorted_op_ids[current_op_index]
                op_data = self.op_graph.nodes[op_id]
                if not self.is_feasible(op_data, device):
                    # no more operator can be placed on this device
                    break
                device.place_op(op_id)
                current_op_index += 1

        if current_op_index != len(self._sorted_op_ids):
            raise RuntimeError(
                '{} operators cannot be placed on devices.'.format(
                    len(self._sorted_op_ids) - current_op_index))

    def run(self):
        """Places operators on devices based on the m_topo algorithm."""
        self._place_ops()
        _LOGGER.info('Topo placement stats')
        utils.log_stats(self.op_graph, self.device_graph)


def _calculate_max_memory_per_device(op_graph, device_graph):
    # assumes 'memory' exists in op_data
    required_memory = 0
    max_op_memory = 0
    for _, op_data in op_graph.nodes.items():
        op_memory = op_data['memory']
        max_op_memory = max(max_op_memory, op_memory)
        required_memory += op_memory

    _LOGGER.info('required memory=%s, max op memory=%s',
                 utils.humanize_num_bytes(required_memory),
                 utils.humanize_num_bytes(max_op_memory))

    # device_memory_sum = sum([device_node['memory_limit'] for _, device_node
                             # in device_graph.nodes.items()])
    # if required_memory > device_memory_sum:
        # raise RuntimeError(
            # 'Not enough memory. required={}, available={}'.format(
                # utils.humanize_num_bytes(required_memory),
                # utils.humanize_num_bytes(device_memory_sum)))

    max_memory_per_device = required_memory // device_graph.number_of_nodes()
    max_memory_per_device += max_op_memory
    _LOGGER.info('Max memory per device: %s',
                 utils.humanize_num_bytes(max_memory_per_device))
    # assumes that each device has memory capacity larger than
    # max_memory_per_device above...
    # for device_id, device_memory in device_graph.nodes.data('memory_limit'):
    #     if device_memory < max_memory_per_device:
    #         raise RuntimeError('Not enough memory on device {}. '
    #                            'required={}, available={}'.format(
    #                                device_id, max_memory_per_device,
    #                                device_memory))

    return max_memory_per_device


class TopoUniform(Topo):
    """Topological sort placement that places ops over devices uniformly."""

    def __init__(self, op_graph, device_graph):
        super(TopoUniform, self).__init__(op_graph, device_graph)
        self._max_memory_per_device = None

    def initialize(self):
        super(TopoUniform, self).initialize()
        self._max_memory_per_device = _calculate_max_memory_per_device(
            self.op_graph, self.device_graph)

    def is_feasible(self, op_data, device):
        return (device.used_memory + op_data['memory'] <=
                self._max_memory_per_device)


class TopoWithColocation(Topo):
    """Topological sort based placement with colocation group constraints."""

    def __init__(self, op_graph, device_graph):
        super(TopoWithColocation, self).__init__(op_graph, device_graph)
        self.colocation_group_infos = {}

    def initialize(self):
        super(TopoWithColocation, self).initialize()
        self.colocation_group_infos = utils.create_colocation_group_infos(
            self.op_graph)

    def is_group_feasible(self, group_info, device):
        """Returns True if the group can be placed on the device."""
        group_memory = (group_info['temp_memory_max'] +
                        group_info['persistent_memory_sum'])
        return device.available_memory >= group_memory

    def _place_ops(self):
        current_op_index = 0

        for device_id, device in sorted(self._devices.items()):
            while current_op_index < len(self._sorted_op_ids):
                op_id = self._sorted_op_ids[current_op_index]
                op_data = self.op_graph.nodes[op_id]
                if 'p' not in op_data:
                    group = op_data['colocation_group']
                    group_info = self.colocation_group_infos[group]
                    if not self.is_group_feasible(group_info, device):
                        # no more operator can be placed on this device
                        # TODO: address memory fragmentation issue.
                        #       this device may contain more operators.
                        break

                    for colocated_op in group_info['ops']:
                        assert 'p' not in colocated_op, \
                            'Assigned device exists'
                        device.place_op(colocated_op['id'])

                current_op_index += 1

        if current_op_index != len(self._sorted_op_ids):
            raise RuntimeError(
                '{} operators cannot be placed on devices.'.format(
                    len(self._sorted_op_ids) - current_op_index))


class TopoUniformWithColocation(TopoWithColocation):
    """Topological sort based placement with colocation group constraints."""

    def __init__(self, op_graph, device_graph):
        super(TopoUniformWithColocation, self).__init__(op_graph, device_graph)
        self._max_memory_per_device = None

    def initialize(self):
        super(TopoUniformWithColocation, self).initialize()
        self._max_memory_per_device = _calculate_max_memory_per_device(
            self.op_graph, self.device_graph)

    def is_group_feasible(self, group_info, device):
        """Returns True if the group can be placed on the device."""
        group_memory = (group_info['temp_memory_max'] +
                        group_info['persistent_memory_sum'])
        return device.used_memory + group_memory < self._max_memory_per_device


def m_topo(op_graph, device_graph, colocation=False, uniform=True):
    """Places operators on devices evenly by using the topological sort.

    Args:
        op_graph: simulation graph
        device_graph: device graph
        colocation: flag whether the colocation rule is taken into account
        uniform: flag whether # ops per device are uniformly distributed
                 over devices
    """

    if colocation:
        topo_cls = TopoUniformWithColocation if uniform else TopoWithColocation
    else:
        topo_cls = TopoUniform if uniform else Topo

    topo = topo_cls(copy.deepcopy(op_graph), copy.deepcopy(device_graph))
    topo.initialize()
    topo.run()

    utils.transfer_placement(topo.op_graph, op_graph)

    return op_graph
