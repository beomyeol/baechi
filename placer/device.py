# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Device Wrapper."""
from __future__ import absolute_import, division, print_function

from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class DeviceWrapper():
    """Wrapper class for a node in the device graph."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, device_id, device_graph, op_graph, memory_check=True):
        self._id = device_id
        self._device_graph = device_graph
        self._node = device_graph.nodes[device_id]
        self._op_graph = op_graph
        self._memory_check = memory_check

        self._next_available_ts = 0
        self._channel_available_ts = 0
        self._cached_tensors = {}

    @staticmethod
    def type():
        """Wrapper type."""
        return "normal"

    @property
    def id(self):
        """Device id."""
        # pylint: disable=invalid-name
        return self._id

    def __getitem__(self, key):
        return self._node[key]

    @property
    def used_memory(self):
        """Currently used memory."""
        return self._node["size"]

    @property
    def next_available_ts(self):
        """Device's next available timestamp."""
        return self._next_available_ts

    @property
    def memory_limit(self):
        """Memory limit."""
        return self._node["memory_limit"]

    @property
    def available_memory(self):
        """Available memory on this device."""
        return self.memory_limit - self.used_memory

    @property
    def channel_available_ts(self):
        """Next channel available ts."""
        return self._channel_available_ts

    def get_cached_tensor(self, tensor_name):
        """Returns whether the given tensor is cached at this device."""
        return self._cached_tensors.get(tensor_name, None)

    def send_tensor(self, tensor_data, from_op_end_ts, dry_run=False):
        """Sends the tensor associated with the given edge."""
        send_start_ts = max(from_op_end_ts, self.channel_available_ts)
        comm_cost = tensor_data['weight']
        send_end_ts = send_start_ts + comm_cost
        if not dry_run:
            self._channel_available_ts = send_end_ts
            tensor_data['send_start_ts'] = send_start_ts
            tensor_data['send_end_ts'] = send_end_ts
        return send_start_ts, send_end_ts

    def recv_tensor(self, tensor_data, dry_run=False, send_start_ts=None):
        """Receives the tensor associated with the given edge."""
        tensor_name = tensor_data['name']
        assert tensor_name not in self._cached_tensors, \
            'tensor {} was already received at device {}'.format(
                tensor_name, self.id)

        comm_cost = tensor_data['weight']
        recv_start_ts = max(send_start_ts or tensor_data['send_start_ts'],
                            self.channel_available_ts)
        recv_end_ts = recv_start_ts + comm_cost
        if not dry_run:
            self._channel_available_ts = recv_end_ts
            tensor_data['recv_start_ts'] = recv_start_ts
            tensor_data['recv_end_ts'] = recv_end_ts
            self._cached_tensors[tensor_name] = {
                'send_start_ts': tensor_data['send_start_ts'],
                'send_end_ts': tensor_data['send_end_ts'],
                'recv_start_ts': tensor_data['recv_start_ts'],
                'recv_end_ts': tensor_data['recv_end_ts']}
        return recv_start_ts, recv_end_ts

    @staticmethod
    def get_op_memory(op_data):
        """Returns operator's required memory."""
        return op_data["temporary_memory"] + sum(op_data["output_memory"])

    def is_placeable(self, op_data):
        """Returns whether the given operator can be placed on this device."""
        return self.get_op_memory(op_data) <= self.available_memory

    def place_op(self, op_id):
        """Place the given op on this device, but not execute it yet."""
        op_data = self._op_graph.nodes[op_id]
        op_memory = self.get_op_memory(op_data)

        assert "p" not in op_data, \
            "Operator id={} was already placed. prev={}, new={}".format(
                op_id, op_data["p"], self.id)
        if self._memory_check:
            assert self.is_placeable(op_data), \
                "Cannot place op {} on dev {}. required={}, avail={}".format(
                    op_id, self.id, op_memory, self.available_memory)

        op_data["p"] = self.id
        self._node["size"] += op_memory

    def run_op(self, op_id):
        """Run the given op on this device.

        Update op information and device information for this execution.
        """
        op_data = self._op_graph.nodes[op_id]

        assert 'start_ts' not in op_data, \
            "Op id=%d was executed before" % op_id
        assert op_data["p"] == self.id, \
            "Op id={}, Assigned dev={}, this dev={}".format(
                op_id, op_data["p"], self.id)

        # update stats
        op_data["start_ts"] = max(self.next_available_ts, op_data["ready_ts"])
        op_data["end_ts"] = op_data["start_ts"] + op_data["weight"]
        self._next_available_ts = op_data["end_ts"]

        _LOGGER.debug(
            "Op name=%s runs on device %d. start_ts=%.2f, end_ts=%.2f"
            + (", ops=%s" % op_data["op_names"]
               if "op_names" in op_data else ""),
            op_data['name'],
            self.id,
            op_data['start_ts'],
            op_data["end_ts"])


class DeviceWrapperAllocator(DeviceWrapper):
    """Wrapper class for a node in the device graph with memory allocator."""

    def __init__(self, device_id, device_graph, op_graph, memory_check=True):
        super(DeviceWrapperAllocator, self).__init__(
            device_id, device_graph, op_graph, memory_check)
        self._node["peak_memory"] = 0

    @staticmethod
    def type():
        return "allocator"

    @property
    def peak_memory(self):
        """Peak memory."""
        return self._node["peak_memory"]

    def allocate_memory(self, num_bytes):
        """Allocates the given bytes memory."""
        if self._memory_check:
            assert self.available_memory >= num_bytes

        self._node["size"] += num_bytes
        if self._node["size"] > self.peak_memory:
            self._node["peak_memory"] = self._node["size"]

    def deallocate_memory(self, num_bytes):
        """Deallocates the given bytes number."""
        self._node["size"] -= num_bytes

    def place_op(self, op_id):
        """Place the given op on this device, but not execute it yet."""
        op_data = self._op_graph.nodes[op_id]

        assert "p" not in op_data, \
            "Operator id={} was already placed. prev={}, new={}".format(
                op_id, op_data["p"], self.id)
        if self._memory_check:
            assert self.is_placeable(op_data), \
                "Cannot place op {} on dev {}. required={}, avail={}".format(
                    op_id, self.id, self.get_op_memory(op_data),
                    self.available_memory)

        op_data["p"] = self.id

        # not update memory yet

    def run_op(self, op_id):
        """Run the given op on this device.

        Update op information and device information for this execution.
        """
        op_data = self._op_graph.nodes[op_id]

        self.allocate_memory(sum(op_data["output_memory"]))
        self.allocate_memory(op_data["temporary_memory"])

        super(DeviceWrapperAllocator, self).run_op(op_id)

        self.deallocate_memory(op_data["temporary_memory"])

        # deallocate input op's output memory when all its successor ops
        # finished.
        for input_op_id, _ in self._op_graph.in_edges(op_id):
            input_op = self._op_graph.nodes[input_op_id]
            executed_out_count = input_op.get("executed_out_count", 0) + 1
            input_op["executed_out_count"] = executed_out_count
            if executed_out_count == self._op_graph.out_degree(input_op_id):
                # deallocate input op's output memory
                input_op_device = self._device_graph.nodes[input_op["p"]]
                input_op_device["size"] -= sum(input_op["output_memory"])
                input_op_device["size"] += input_op["persistent_memory"]
