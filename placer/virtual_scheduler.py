# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Virtual Scheduler."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from placer.device import DeviceWrapperAllocator
from placer import placer_utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class VirtualScheduler():
    """Virtual Scheduler."""

    def __init__(self, op_graph, device_graph, log_execution=False):
        self.op_graph = op_graph
        self.device_graph = device_graph
        self._log_execution = log_execution
        self._devices = {
            device_id: DeviceWrapperAllocator(
                device_id, self.device_graph, self.op_graph, False)
            for device_id in self.device_graph.nodes}
        self._ready_op_queue = placer_utils.SortedTimestampOps()

    def _get_ready_ts(self, op_data):
        ready_ts = placer_utils.get_ready_ts(self.op_graph, op_data['id'])
        op_data['ready_ts'] = ready_ts
        return ready_ts

    def initialize(self):
        """Initializes."""
        for op_id, op_data in self.op_graph.nodes().items():
            op_data['ready_count'] = 0
            op_data['executed_out_count'] = 0
            if self.op_graph.in_degree(op_id) == 0:
                self._ready_op_queue.add_op(self._get_ready_ts(op_data),
                                            op_data)

    def run(self):
        """Runs execution simulation."""
        # pylint: disable=too-many-locals
        makespan = 0

        if self._log_execution:
            out_f = open('virtual_scheduler.log', 'w')

        while self._ready_op_queue:
            op_data = self._ready_op_queue.pop().op
            op_id = op_data['id']
            device_id = op_data['p']
            device = self._devices[device_id]

            device.run_op(op_id)

            if self._log_execution:
                out_f.write(placer_utils.generate_op_run_log(op_data))

            makespan = max(makespan, device.next_available_ts)

            # send tensors to the next op devices
            out_edges = self.op_graph.out_edges(op_id, data=True)
            for _, next_op_id, edge_data in out_edges:
                next_op = self.op_graph.nodes[next_op_id]
                next_op_device_id = next_op['p']
                if device_id != next_op_device_id:
                    next_op_device = self._devices[next_op_device_id]
                    for tensor_data in edge_data['tensor']:
                        cached_tensor = next_op_device.get_cached_tensor(
                            tensor_data['name'])
                        if cached_tensor is None:
                            # there is no cached tensor at the receiver
                            device.send_tensor(tensor_data, op_data['end_ts'])
                            next_op_device.recv_tensor(tensor_data)
                        else:
                            # update tensor transfer info
                            tensor_data.update(cached_tensor)

                        if self._log_execution:
                            out_f.write(placer_utils.generate_memcpy_log(
                                tensor_data, op_data, next_op, device_id,
                                next_op_device_id, cached_tensor is not None))


            ready_ops = placer_utils.process_finished_op(self.op_graph, op_id)
            for ready_op in ready_ops:
                self._ready_op_queue.add_op(self._get_ready_ts(ready_op),
                                            ready_op)

        if self._log_execution:
            out_f.close()

        return makespan
