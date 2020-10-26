# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Memory-constrainted earliest time first placement."""
from __future__ import absolute_import, division, print_function

import copy
from collections import namedtuple
from enum import Enum

from placer import device as device_wrapper
from placer import placer_utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class DeviceState(Enum):
    """Device state."""
    FREE = 0
    BUSY = 1
    AWAKE = 2  # used in SCT


class ETFDevice(device_wrapper.DeviceWrapper):
    """SCT Device wrapper."""

    def __init__(self, device_id, device_graph, op_graph, reserve_memory):
        super(ETFDevice, self).__init__(device_id, device_graph, op_graph)
        self._current_ts = 0
        self._state = DeviceState.FREE
        self._node['peak_memory'] = 0
        self._last_op = None
        # group_name -> reserved_memory (available, max)
        self._reserved_memory_map = {} if reserve_memory else None

    @property
    def peak_memory(self):
        """Peak memory."""
        return self._node["peak_memory"]

    def advance(self, timestamp):
        """Advances the device time to the given timestamp."""
        self._current_ts = timestamp
        new_ready_ops = []
        if self._current_ts == self.next_available_ts:
            if self._last_op is not None:
                new_ready_ops = placer_utils.process_finished_op(
                    self._op_graph, self._last_op['id'])
        if self.next_available_ts < self._current_ts:
            self._next_available_ts = self._current_ts
        return new_ready_ops

    def get_schedule_ts(self, op_data):
        """Returns schedule-able timestamp for the operator."""
        schedule_tss = op_data['schedule_tss']
        if self.id in schedule_tss:
            return schedule_tss[self.id]
        return None

    def _get_earlist_ready_op(self, ready_op_manager):
        # assumes that 'schedule_tss' is updated at op_data
        ready_ts_ops = placer_utils.SortedTimestampOps()
        for ready_op in ready_op_manager:
            schedule_ts = self.get_schedule_ts(ready_op)
            if schedule_ts is not None:
                ready_ts_ops.add_op(schedule_ts, ready_op)

        return ready_ts_ops[0].op if len(ready_ts_ops) > 0 else None

    def _get_op_on_free(self, ready_op_manager):
        ready_op = self._get_earlist_ready_op(ready_op_manager)
        if ready_op and self.get_schedule_ts(ready_op) <= self._current_ts:
            return ready_op
        return None

    def get_op_to_schedule(self, ready_op_manager):
        """Returns an operator to schedule if available.

        Returns:
            a operator that is scheduled.
        """
        # assumes schedule_tss in op_data is updated.
        if self._current_ts < self.next_available_ts:
            self._state = DeviceState.BUSY
            return None

        # device is free
        self._state = DeviceState.FREE
        return self._get_op_on_free(ready_op_manager)

    def get_next_ts(self, ready_op_manager):
        """Returns the timestamp when this device can have any action."""
        # schedule_tss should be populated.
        next_ts = None
        if self._state == DeviceState.BUSY:
            next_ts = self.next_available_ts
        elif self._state == DeviceState.FREE:
            earliest_ready_op = self._get_earlist_ready_op(ready_op_manager)
            if earliest_ready_op is not None:
                next_ts = earliest_ready_op['schedule_tss'][self.id]
        else:
            raise ValueError('Unknown state: {}'.format(self._state))
        if next_ts is not None and next_ts <= self._current_ts:
            raise ValueError('Timestamp should move forward')
        return next_ts

    @staticmethod
    def get_op_memory(op_data):
        return op_data["temporary_memory"] + sum(op_data["output_memory"])

    def is_placeable(self, op_data):
        available_memory = self.available_memory
        if self._reserved_memory_map is not None:
            # consider reserved memory
            available_memory += self._reserved_memory_map.get(
                op_data['colocation_group'], (0, 0))[0]
        return self.get_op_memory(op_data) <= available_memory

    def _allocate_memory_raw(self, num_bytes):
        if self._memory_check:
            assert self.available_memory >= num_bytes

        self._node["size"] += num_bytes
        if self._node["size"] > self.peak_memory:
            self._node["peak_memory"] = self._node["size"]

    def allocate_memory(self, num_bytes, op_data):
        """Allocates the given bytes memory by the operator."""
        if num_bytes == 0:
            return

        group_name = op_data['colocation_group']
        _LOGGER.debug('allocate memory. group=%s, num_bytes=%d, used=%d',
                      group_name, num_bytes, self.used_memory)
        if self._reserved_memory_map is not None:
            # memory should have been reserved for the group.
            avail, limit = self._reserved_memory_map[group_name]

            remained_bytes = avail - num_bytes
            if remained_bytes >= 0:
                # update reserved memory
                self._reserved_memory_map[group_name] = (remained_bytes, limit)
                _LOGGER.debug(
                    'use reserved memory group=%s, avail=%d, limit=%d',
                    group_name, remained_bytes, limit)
                return
            # no more reserved bytes. need to allocate.
            num_bytes = -remained_bytes
            assert num_bytes >= 0
            self._reserved_memory_map[group_name] = (0, limit)

        _LOGGER.debug('allocate raw memory. group=%s, num_bytes=%d',
                      group_name, num_bytes)
        self._allocate_memory_raw(num_bytes)

    def _deallocate_memory_raw(self, num_bytes):
        self._node["size"] -= num_bytes
        assert self._node["size"] >= 0

    def deallocate_memory(self, num_bytes, op_data):
        """Deallocates the given bytes number by the operator."""
        if num_bytes == 0:
            return

        group_name = op_data['colocation_group']
        _LOGGER.debug('deallocate memory. group=%s, num_bytes=%d, used=%d',
                      group_name, num_bytes, self.used_memory)

        if (self._reserved_memory_map is not None
                and group_name in self._reserved_memory_map):
            # restore reserved bytes.
            avail, limit = self._reserved_memory_map[group_name]
            avail = avail + num_bytes
            if avail <= limit:
                self._reserved_memory_map[group_name] = (avail, limit)
                return
            num_bytes = avail - limit
            assert num_bytes >= 0
            self._reserved_memory_map[group_name] = (limit, limit)

        _LOGGER.debug('deallocate raw memory. group=%s, num_bytes=%d',
                      group_name, num_bytes)
        self._deallocate_memory_raw(num_bytes)

    def reserve_memory(self, num_bytes, group_name):
        """Reserves memory for the given group."""
        assert group_name not in self._reserved_memory_map
        self._reserved_memory_map[group_name] = (num_bytes, num_bytes)
        _LOGGER.debug('Reserved memory. group=%s, num_bytes=%d, used=%d',
                      group_name, num_bytes, self.used_memory)
        self._allocate_memory_raw(num_bytes)

    def deallocate_reserved_memory(self, group_name):
        """Deallocates reserved memory for the group."""
        available_bytes, _ = self._reserved_memory_map[group_name]
        _LOGGER.debug(
            'deallocate reserved memory. group=%s, num_bytes=%d, used=%d',
            group_name, available_bytes, self.used_memory)
        self._deallocate_memory_raw(available_bytes)
        del self._reserved_memory_map[group_name]

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

        # allocate persistent memory
        self.allocate_memory(op_data["persistent_memory"], op_data)

    def deallocate_predecessor_memory(self, op_id, devices):
        """Deallocates input op's output memory on finishing its successor ops.

        This should be called after run_op().
        """
        for input_op_id, _ in self._op_graph.in_edges(op_id):
            input_op = self._op_graph.nodes[input_op_id]
            executed_out_count = input_op.get("executed_out_count", 0) + 1
            input_op["executed_out_count"] = executed_out_count
            if executed_out_count == self._op_graph.out_degree(input_op_id):
                # deallocate input op's output memory
                devices[input_op["p"]].deallocate_memory(
                    sum(input_op["output_memory"]), input_op)

    def run_op(self, op_id):
        """Runs the given op on this device.

        Update op information and device information for this execution.
        """
        op_data = self._op_graph.nodes[op_id]

        # allocate output_memory
        self.allocate_memory(sum(op_data["output_memory"]), op_data)
        self.allocate_memory(op_data["temporary_memory"], op_data)
        super(ETFDevice, self).run_op(op_id)
        self.deallocate_memory(op_data["temporary_memory"], op_data)
        self._last_op = self._op_graph.nodes[op_id]


ScheduleOpMetadata = namedtuple(
    'ScheduleOpMetadata', ['ts_op_tuple', 'device'])


class ETF():
    """Memory-constrainted earliest time first placement."""

    def __init__(self, op_graph, device_graph, colocation, reserve_memory,
                 log_file=None):
        # pylint: disable=too-many-arguments
        self.op_graph = op_graph
        self._device_graph = device_graph
        self._log_file = log_file
        self._colocation_group_infos = (
            placer_utils.create_colocation_group_infos(self.op_graph)
            if colocation else None)
        self.reserve_memory = reserve_memory

        # initialized in self.initialize()
        self._num_scheduled_ops = None
        self._devices = {}  # device id -> device wrapper
        self._ready_op_manager = None

    def initialize(self):
        """Initializes."""
        self._num_scheduled_ops = 0
        self._devices = {
            device_id: ETFDevice(device_id, self._device_graph,
                                 self.op_graph, self.reserve_memory)
            for device_id in self._device_graph.nodes}
        self._ready_op_manager = placer_utils.ReadyOpManager(
            self.op_graph, self._devices, self._log_file)

        for op_id, op_data in self.op_graph.nodes.items():
            op_data['ready_count'] = 0
            if self.op_graph.in_degree(op_id) == 0:
                self._ready_op_manager.add(op_data)

    def is_colocation_considered(self):
        """Returns whether the colocation is considered in the placement."""
        return self._colocation_group_infos is not None

    def _process_scheduled_op(self, scheduled_op):
        self._ready_op_manager.remove(scheduled_op)
        self._num_scheduled_ops += 1

        if self.is_colocation_considered() and self.reserve_memory:
            group_name = scheduled_op['colocation_group']
            group_info = self._colocation_group_infos[group_name]
            group_info['reference_count'] -= 1
            if group_info['reference_count'] == 0:
                # all ops in the group finished. deallocate reserved memory.
                device = self._devices[scheduled_op['p']]
                device.deallocate_reserved_memory(group_name)

    def _get_required_memory_for_group(self, group_info):
        if self.reserve_memory:
            return sum([group_info["temp_memory_max"],
                        group_info["persistent_memory_sum"],
                        group_info["output_memory_max"]])

        return group_info["persistent_memory_sum"]

    def _is_placeable_group(self, group_name, device_id):
        """Returns whether the ops in the group can be placed on the device."""
        group_info = self._colocation_group_infos[group_name]
        required_memory = self._get_required_memory_for_group(group_info)
        return required_memory <= self._devices[device_id].available_memory

    def _get_next_op_metadata(self):
        self._ready_op_manager.populate_schedule_tss()

        if self.is_colocation_considered():
            # filter the cases where the operators in a group cannot be placed
            # on a device.
            for ready_op in self._ready_op_manager:
                if 'p' in ready_op:
                    # this operator is already placed on the device.
                    # memory check was done before.
                    continue
                schedule_tss = ready_op['schedule_tss']
                infeasible_device_ids = [
                    device_id for device_id in schedule_tss
                    if not self._is_placeable_group(
                        ready_op['colocation_group'], device_id)]
                for device_id in infeasible_device_ids:
                    del schedule_tss[device_id]

        next_op_metadata = None
        for device in self._devices.values():
            next_op = device.get_op_to_schedule(self._ready_op_manager)
            if next_op is None:
                continue

            op_metadata = ScheduleOpMetadata(
                ts_op_tuple=placer_utils.TimestampOpTuple(
                    device.get_schedule_ts(next_op), next_op),
                device=device)

            if next_op_metadata is None:
                next_op_metadata = op_metadata
            else:
                if op_metadata.ts_op_tuple < next_op_metadata.ts_op_tuple:
                    next_op_metadata = op_metadata

        return next_op_metadata

    def _run_schedule_step(self, timestamp):
        """Runs a single schedule step at the given timestamp.

        Returns:
            next schedule timestamp.
        """
        for device in self._devices.values():
            self._ready_op_manager.extend(device.advance(timestamp))

        while True:
            op_metadata = self._get_next_op_metadata()
            if op_metadata is None:
                # no device has an operator to run at this time.
                break

            op_data = op_metadata.ts_op_tuple.op
            op_data['ready_ts'] = self._ready_op_manager.get_ready_ts(
                op_data, op_metadata.device, dry_run=False)
            op_id = op_data['id']
            if 'p' not in op_data:
                if self.is_colocation_considered():
                    # place all ops in the same group.
                    # Assumes the device has enough memory to assign all ops
                    # (this is checked at _get_next_op_metadata()).
                    group_name = op_data['colocation_group']
                    group_info = self._colocation_group_infos[group_name]
                    if self.reserve_memory:
                        op_metadata.device.reserve_memory(
                            self._get_required_memory_for_group(group_info),
                            group_name)
                        # reference counter for deallocate reserved memory
                        group_info['reference_count'] = len(group_info['ops'])
                    for colocated_op in group_info['ops']:
                        op_metadata.device.place_op(colocated_op['id'])
                else:
                    op_metadata.device.place_op(op_id)

            assert op_data['p'] == op_metadata.device.id
            op_metadata.device.run_op(op_id)
            op_metadata.device.deallocate_predecessor_memory(
                op_id, self._devices)
            if self._log_file:
                self._log_file.write(placer_utils.generate_op_run_log(op_data))
            self._process_scheduled_op(op_data)

        min_next_ts = None
        for device in self._devices.values():
            next_ts = device.get_next_ts(self._ready_op_manager)
            if next_ts is not None:
                min_next_ts = min(min_next_ts or next_ts, next_ts)

        return min_next_ts

    def run(self):
        """Runs the placement."""
        # start to schedule and place ops
        self.initialize()

        current_ts = 0  # current timestamp
        while True:
            next_ts = self._run_schedule_step(current_ts)
            if next_ts is None:
                break
            current_ts = next_ts

        assert self._num_scheduled_ops == self.op_graph.number_of_nodes(), \
            "# scheduled ops={}, # ops={}".format(
                self._num_scheduled_ops, self.op_graph.number_of_nodes())

        return current_ts


def m_etf(op_graph, device_graph, colocation=False, reserve_memory=False):
    """Places operators over the devices by using ETF."""
    etf = ETF(copy.deepcopy(op_graph), copy.deepcopy(device_graph),
              colocation, reserve_memory)
    runtime = etf.run()
    _LOGGER.info('ETF estimated runtime: %f', runtime / 1e6)
    placer_utils.transfer_placement(etf.op_graph, op_graph)
    return op_graph
