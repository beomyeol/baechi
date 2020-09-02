# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import operator
import tensorflow as tf

from placer import placer_utils as utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class Adjuster():
    """Abstract adjuster that adjusts placement for colocation groups."""

    def __init__(self, op_graph, device_graph, device_memory_limit,
                 colocation_group_infos=None):
        self._op_graph = op_graph
        self._device_graph = device_graph
        self._device_memory_limit = device_memory_limit
        if colocation_group_infos is None:
            colocation_group_infos = utils.create_colocation_group_infos(
                self._op_graph)
        self._colocation_group_infos = colocation_group_infos

    def get_op(self, op_id):
        """Returns an operator with the given id."""
        return self._op_graph.nodes[op_id]

    @staticmethod
    def _get_op_memory(op_data):
        return op_data["temporary_memory"] + op_data["persistent_memory"]

    @abstractmethod
    def pick_device(self, ops):
        """Returns a device id to assign among ops in the same group.

        Args:
            ops: a list of op dicts, values of op_graph.
        """
        raise NotImplementedError()

    def is_feasible(self, ops, device_id):
        """Returns whether ops can be placed to the given device."""
        required_memory = sum([self._get_op_memory(op)
                               for op in ops if op["p"] != device_id])
        device = self._device_graph.nodes[device_id]
        return device["size"] + required_memory <= device["memory_limit"]

    def run(self):
        """Runs the adjustment."""
        _LOGGER.info('adjusting with device memory limit: %s',
                     str(self._device_memory_limit))

        adjusted_op_count = 0

        for group_info in self._colocation_group_infos.values():
            device_id = self.pick_device(group_info["ops"])
            # update assigned devices for ops
            for op in group_info["ops"]:
                if op["p"] != device_id:
                    adjusted_op_count += 1
                    _LOGGER.debug(
                        "Device placement is adjusted. op=%s, prev=%d, new=%d",
                        op["name"],
                        op["p"],
                        device_id)
                    if self._device_memory_limit:
                        op_memory = self._get_op_memory(op)
                        prev_device = self._device_graph.nodes[op["p"]]
                        prev_device["size"] -= op_memory
                        new_device = self._device_graph.nodes[device_id]
                        new_device["size"] += op_memory
                    op["p"] = device_id

        _LOGGER.info("# ops whose placements were adjusted: %d",
                     adjusted_op_count)

        return self._op_graph


class NoopAdjuster(Adjuster):
    """Adjuster that does not do anything.

    This may cause an error at the TensorFlow runtime due to dissatisfaction
    of the colocation rule.
    """

    def pick_device(self, ops):
        raise RuntimeError('Should not be called.')

    def run(self):
        return self._op_graph


class RandomAdjuster(Adjuster):
    """Adjuster that uses the device of random op in each group for all ops.
    """

    def pick_device(self, ops):
        if self._device_memory_limit:
            for op in ops:
                if self.is_feasible(ops, op["p"]):
                    return op["p"]
            raise RuntimeError(
                "Cannot adjust the placement: Not enough memory.")

        return ops[0]["p"]


class MemoryAdjuster(Adjuster):
    """Adjuster that picks the device based on the memory requirement.

    This uses the device where the largest memory required op is located
    for all ops in each group.
    """

    def pick_device(self, ops):
        # find out the op that requires the largest memory memory
        if self._device_memory_limit:
            sorted_ops = sorted(ops, key=self._get_op_memory, reverse=True)
            for op in sorted_ops:
                if self.is_feasible(ops, op["p"]):
                    return op["p"]
            raise RuntimeError(
                "Cannot adjust the placement: Not enough memory.")

        op_with_largest_memory = max(ops, key=self._get_op_memory)
        return op_with_largest_memory["p"]


class TimeAdjuster(Adjuster):
    """Adjuster that picks the device based on the computation time.

    This uses the device where the largest memory required op is located
    for all ops in each group.
    """

    def pick_device(self, ops):
        # find out the op that requires the largest memory memory
        if self._device_memory_limit:
            sorted_ops = sorted(
                ops, key=operator.itemgetter("weight"), reverse=True)
            for op in sorted_ops:
                if self.is_feasible(ops, op["p"]):
                    return op["p"]
            raise RuntimeError(
                "Cannot adjust the placement: Not enough memory.")

        op_with_largest_memory = max(ops, key=operator.itemgetter("weight"))
        return op_with_largest_memory["p"]


class FrequentAdjuster(Adjuster):
    """Adjuster that picks the most frequent device among assigned devices."""

    def pick_device(self, ops):
        device_ids = [op["p"] for op in ops]
        if self._device_memory_limit:
            sorted_device_ids = sorted(
                set(device_ids), key=device_ids.count, reverse=True)
            for device_id in sorted_device_ids:
                if self.is_feasible(ops, device_id):
                    return device_id
            raise RuntimeError(
                "Cannot adjust the placement: Not enough memory.")

        return max(set(device_ids), key=device_ids.count)


class MemorySumAdjuster(Adjuster):
    """Adjuster that picks the device on which ops has the largest memory."""

    def pick_device(self, ops):
        # calculate memory usages per device
        memory_per_device = {}
        for op in ops:
            device_id = op["p"]
            memory_per_device[device_id] = (
                memory_per_device.get(device_id, 0) + self._get_op_memory(op))

        if self._device_memory_limit:
            sorted_device_memory = sorted(
                memory_per_device.items(), key=operator.itemgetter(1),
                reverse=True)
            for device_id, _ in sorted_device_memory:
                if self.is_feasible(ops, device_id):
                    return device_id
            raise RuntimeError(
                "Cannot adjust the placement: Not enough memory.")

        # pick the device with the largest memory usages
        return max(memory_per_device.items(), key=operator.itemgetter(1))[0]


class TimeSumAdjuster(Adjuster):
    """Adjuster that picks the dev on which ops have the largest comp time."""

    def pick_device(self, ops):
        # calculate the sum of computation times per device
        time_per_device = {}
        for op in ops:
            device_id = op["p"]
            time_per_device[device_id] = (
                time_per_device.get(device_id, 0) + op["weight"])

        if self._device_memory_limit:
            sorted_device_times = sorted(
                time_per_device.items(), key=operator.itemgetter(1),
                reverse=True)
            for device_id, _ in sorted_device_times:
                if self.is_feasible(ops, device_id):
                    return device_id
            raise RuntimeError(
                "Cannot adjust the placement: Not enough memory.")

        # pick the device with the largest computation time
        return max(time_per_device.items(), key=operator.itemgetter(1))[0]


_ADJUSTER_CLASSES = {
    "noop": NoopAdjuster,
    "random": RandomAdjuster,
    "memory": MemoryAdjuster,
    "time": TimeAdjuster,
    "frequent": FrequentAdjuster,
    "memory_sum": MemorySumAdjuster,
    "time_sum": TimeSumAdjuster,
}


tf.app.flags.DEFINE_enum(
    "adjustment_method", "noop", list(_ADJUSTER_CLASSES.keys()),
    "Method to adjust placement for colocation groups")
tf.app.flags.DEFINE_boolean(
    'adjustment_with_memory_limit', False,
    'In adjusting the placement, the device memory is considered.')

FLAGS = tf.app.flags.FLAGS


def get_adjuster(adjustment_method, op_graph, device_graph,
                 adjustment_with_memory_limit=None,
                 colocation_group_infos=None):
    """Returns an adjuster instance according to the flags."""
    adjustment_method = adjustment_method or FLAGS.adjustment_method
    if adjustment_with_memory_limit is None:
        adjustment_with_memory_limit = FLAGS.adjustment_with_memory_limit
    adjuster_cls = _ADJUSTER_CLASSES[adjustment_method]
    return adjuster_cls(op_graph, device_graph, adjustment_with_memory_limit,
                        colocation_group_infos)
