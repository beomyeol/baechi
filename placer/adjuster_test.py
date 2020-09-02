"""Adjuster unitests."""
# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=invalid-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import networkx as nx

from placer import adjuster as adjuster_lib


def run_adjuster(method, op_graph, device_graph, device_memory_limit=False):
    adjuster = adjuster_lib.get_adjuster(
        method, op_graph, device_graph, device_memory_limit)
    return adjuster.run()


def get_test_graph():
    """Returns a test graph."""
    op_graph = nx.DiGraph()
    op_graph.add_node(1,
                      name='op1', id=1,
                      colocation_group='test_group', output_memory=[60],
                      weight=60, persistent_memory=0, temporary_memory=60,
                      p=1)
    op_graph.add_node(2,
                      name='op2', id=2,
                      colocation_group='test_group', output_memory=[1],
                      weight=100, persistent_memory=0, temporary_memory=1,
                      p=2)
    op_graph.add_node(3,
                      name='op3', id=3,
                      colocation_group='test_group', output_memory=[100],
                      weight=1, persistent_memory=0, temporary_memory=100,
                      p=3)
    op_graph.add_node(4,
                      name='op4', id=4,
                      colocation_group='test_group', output_memory=[70],
                      weight=70, persistent_memory=0, temporary_memory=70,
                      p=1)
    op_graph.add_node(5,
                      name='op5', id=5,
                      colocation_group='test_group', output_memory=[10],
                      weight=5, persistent_memory=0, temporary_memory=10,
                      p=2)
    op_graph.add_node(6,
                      name='op6', id=6,
                      colocation_group='test_group', output_memory=[10],
                      weight=10, persistent_memory=0, temporary_memory=10,
                      p=2)
    return op_graph


def get_device_graph():
    """Returns a test device graph."""
    device_graph = nx.DiGraph()
    device_graph.add_node(1,
                          size=130,  # op1, op4 (compute cost sum: 130)
                          memory_limit=260)
    device_graph.add_node(2,
                          size=21,  # op2, op5, op6 (compute cost sum: 115)
                          memory_limit=260)
    device_graph.add_node(3,
                          size=100,  # op3 (compute cost sum: 100)
                          memory_limit=260)
    return device_graph


class MemoryAdjusterTest(unittest.TestCase):

    def test_small(self):
        op_graph = get_test_graph()
        op_graph = run_adjuster(
            "memory",
            op_graph,
            device_graph=None,
            device_memory_limit=False)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 3)

    def test_small_with_memory_limit_fit(self):
        op_graph = get_test_graph()
        device_graph = get_device_graph()
        op_graph = run_adjuster(
            "memory",
            op_graph,
            device_graph=device_graph,
            device_memory_limit=True)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 3)

        self.assertEqual(device_graph.nodes[1]['size'], 0)
        self.assertEqual(device_graph.nodes[2]['size'], 0)
        self.assertEqual(device_graph.nodes[3]['size'], 251)

    def test_small_with_memory_limit_not_fit(self):
        op_graph = get_test_graph()
        device_graph = get_device_graph()
        device_graph.nodes[3]['memory_limit'] = 250
        op_graph = run_adjuster(
            "memory",
            op_graph,
            device_graph=device_graph,
            device_memory_limit=True)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 1)

        self.assertEqual(device_graph.nodes[1]['size'], 251)
        self.assertEqual(device_graph.nodes[2]['size'], 0)
        self.assertEqual(device_graph.nodes[3]['size'], 0)


class TimeAdjusterTest(unittest.TestCase):

    def test_small(self):
        op_graph = get_test_graph()
        op_graph = run_adjuster(
            "time",
            op_graph,
            device_graph=None,
            device_memory_limit=False)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 2)

    def test_small_with_memory_limit_fit(self):
        op_graph = get_test_graph()
        device_graph = get_device_graph()
        op_graph = run_adjuster(
            "time",
            op_graph,
            device_graph=device_graph,
            device_memory_limit=True)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 2)

        self.assertEqual(device_graph.nodes[1]['size'], 0)
        self.assertEqual(device_graph.nodes[2]['size'], 251)
        self.assertEqual(device_graph.nodes[3]['size'], 0)

    def test_small_with_memory_limit_not_fit(self):
        op_graph = get_test_graph()
        device_graph = get_device_graph()
        device_graph.nodes[2]['memory_limit'] = 250
        op_graph = run_adjuster(
            "time",
            op_graph,
            device_graph=device_graph,
            device_memory_limit=True)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 1)

        self.assertEqual(device_graph.nodes[1]['size'], 251)
        self.assertEqual(device_graph.nodes[2]['size'], 0)
        self.assertEqual(device_graph.nodes[3]['size'], 0)


class FrequentAdjusterTest(unittest.TestCase):

    def test_small(self):
        op_graph = get_test_graph()
        op_graph = run_adjuster(
            "frequent",
            op_graph,
            device_graph=None,
            device_memory_limit=False)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 2)


class MemorySumAdjusterTest(unittest.TestCase):

    def test_small(self):
        op_graph = get_test_graph()
        op_graph = run_adjuster(
            "memory_sum",
            op_graph,
            device_graph=None,
            device_memory_limit=False)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 1)

    def test_small_with_memory_limit_fit(self):
        op_graph = get_test_graph()
        device_graph = get_device_graph()
        op_graph = run_adjuster(
            "memory_sum",
            op_graph,
            device_graph=device_graph,
            device_memory_limit=True)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 1)

        self.assertEqual(device_graph.nodes[1]['size'], 251)
        self.assertEqual(device_graph.nodes[2]['size'], 0)
        self.assertEqual(device_graph.nodes[3]['size'], 0)

    def test_small_with_memory_limit_not_fit(self):
        op_graph = get_test_graph()
        device_graph = get_device_graph()
        device_graph.nodes[1]['memory_limit'] = 250
        op_graph = run_adjuster(
            "memory_sum",
            op_graph,
            device_graph=device_graph,
            device_memory_limit=True)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 3)

        self.assertEqual(device_graph.nodes[1]['size'], 0)
        self.assertEqual(device_graph.nodes[2]['size'], 0)
        self.assertEqual(device_graph.nodes[3]['size'], 251)


class TimeSumAdjusterTest(unittest.TestCase):

    def test_small(self):
        op_graph = get_test_graph()
        op_graph = run_adjuster(
            "time_sum",
            op_graph,
            device_graph=None,
            device_memory_limit=False)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 1)

    def test_small_with_memory_limit_fit(self):
        op_graph = get_test_graph()
        device_graph = get_device_graph()
        op_graph = run_adjuster(
            "time_sum",
            op_graph,
            device_graph=device_graph,
            device_memory_limit=True)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 1)

        self.assertEqual(device_graph.nodes[1]['size'], 251)
        self.assertEqual(device_graph.nodes[2]['size'], 0)
        self.assertEqual(device_graph.nodes[3]['size'], 0)

    def test_small_with_memory_limit_not_fit(self):
        op_graph = get_test_graph()
        device_graph = get_device_graph()
        device_graph.nodes[1]['memory_limit'] = 250
        op_graph = run_adjuster(
            "time_sum",
            op_graph,
            device_graph=device_graph,
            device_memory_limit=True)

        for _, op in op_graph.nodes().items():
            self.assertEqual(op['p'], 2)

        self.assertEqual(device_graph.nodes[1]['size'], 0)
        self.assertEqual(device_graph.nodes[2]['size'], 251)
        self.assertEqual(device_graph.nodes[3]['size'], 0)


if __name__ == "__main__":
    unittest.main()
