# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Unittests for groupers"""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=protected-access

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import networkx as nx

from placer import grouper


class GrouperTest(unittest.TestCase):

    def test_process_colocation_group_simple(self):
        op_graph = nx.DiGraph()
        op_graph.add_node(1,
                          name="op1", id=1,
                          colocation_group=["group1"])
        op_graph.add_node(2,
                          name="op2", id=2,
                          colocation_group=["group2"])
        op_graph.add_node(3,
                          name="op3", id=3,
                          colocation_group=["group3"])
        op_graph.add_node(4,
                          name="op4", id=4,
                          colocation_group=["group1", "group3"])

        grouper.process_colocation_group(op_graph)

        self.assertEqual(op_graph.nodes[1]["colocation_group"], "group1")
        self.assertEqual(op_graph.nodes[2]["colocation_group"], "group2")
        self.assertEqual(op_graph.nodes[3]["colocation_group"], "group1")
        self.assertEqual(op_graph.nodes[4]["colocation_group"], "group1")


CoplacementGrouper = grouper.CoplacementGrouper


class CoplacementGrouperTest(unittest.TestCase):

    def test_simple_step_1(self):
        op_graph = nx.DiGraph()
        # op0 -> op1
        op0 = {"id": 0, "name": "op0", "colocation_group": "op0"}
        op1 = {"id": 1, "name": "op1", "colocation_group": "op1"}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_edge(0, 1, is_control=False)

        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertTrue(is_changed)
        self.assertEqual(op_graph.nodes[0]["colocation_group"],
                         op_graph.nodes[1]["colocation_group"])
        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertFalse(is_changed)

    def test_simple_step_2(self):
        op_graph = nx.DiGraph()
        # op0 -> op2
        # op1 /
        op0 = {"id": 0, "name": "op0", "colocation_group": "op0"}
        op1 = {"id": 1, "name": "op1", "colocation_group": "op1"}
        op2 = {"id": 2, "name": "op2", "colocation_group": "op2"}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(0, 2, is_control=False)
        op_graph.add_edge(1, 2, is_control=False)

        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertTrue(is_changed)
        self.assertEqual(op_graph.nodes[0]["colocation_group"],
                         op_graph.nodes[1]["colocation_group"])
        self.assertEqual(op_graph.nodes[0]["colocation_group"],
                         op_graph.nodes[2]["colocation_group"])
        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertFalse(is_changed)

    def test_simple_step_3(self):
        op_graph = nx.DiGraph()
        # op0 -> op1 -> op2
        op0 = {"id": 0, "name": "op0", "colocation_group": "op0"}
        op1 = {"id": 1, "name": "op1", "colocation_group": "op1"}
        op2 = {"id": 2, "name": "op2", "colocation_group": "op2"}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(0, 1, is_control=False)
        op_graph.add_edge(1, 2, is_control=False)

        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertTrue(is_changed)
        self.assertEqual(op_graph.nodes[0]["colocation_group"],
                         op_graph.nodes[1]["colocation_group"])
        self.assertEqual(op_graph.nodes[0]["colocation_group"],
                         op_graph.nodes[2]["colocation_group"])
        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertFalse(is_changed)

    def test_simple_step_4(self):
        op_graph = nx.DiGraph()
        # op0 -> op1
        #     \-> op2
        op0 = {"id": 0, "name": "op0", "colocation_group": "op0"}
        op1 = {"id": 1, "name": "op1", "colocation_group": "op1"}
        op2 = {"id": 2, "name": "op2", "colocation_group": "op2"}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(0, 1, is_control=False)
        op_graph.add_edge(0, 2, is_control=False)

        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertFalse(is_changed)
        self.assertEqual(op_graph.nodes[0]["colocation_group"], "op0")
        self.assertEqual(op_graph.nodes[1]["colocation_group"], "op1")
        self.assertEqual(op_graph.nodes[2]["colocation_group"], "op2")

    def test_simple_step_5(self):
        op_graph = nx.DiGraph()
        # op0 -> op1 -> op2
        #     \-> op3
        op0 = {"id": 0, "name": "op0", "colocation_group": "op0"}
        op1 = {"id": 1, "name": "op1", "colocation_group": "op1"}
        op2 = {"id": 2, "name": "op2", "colocation_group": "op2"}
        op3 = {"id": 3, "name": "op3", "colocation_group": "op3"}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_node(3, **op3)
        op_graph.add_edge(0, 1, is_control=False)
        op_graph.add_edge(1, 2, is_control=False)
        op_graph.add_edge(0, 3, is_control=False)

        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertTrue(is_changed)
        self.assertEqual(op_graph.nodes[0]["colocation_group"], "op0")
        self.assertEqual(op_graph.nodes[1]["colocation_group"], "op1")
        self.assertEqual(op_graph.nodes[2]["colocation_group"], "op1")
        self.assertEqual(op_graph.nodes[3]["colocation_group"], "op3")
        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertFalse(is_changed)

    def test_simple(self):
        op_graph = nx.DiGraph()
        # op0 -> op1 ----> op2
        #   \-> op3   op4->/
        op0 = {"id": 0, "name": "op0", "colocation_group": ["op0"]}
        op1 = {"id": 1, "name": "op1", "colocation_group": ["op1"]}
        op2 = {"id": 2, "name": "op2", "colocation_group": ["op2"]}
        op3 = {"id": 3, "name": "op3", "colocation_group": ["op3"]}
        op4 = {"id": 4, "name": "op4", "colocation_group": ["op4"]}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_node(3, **op3)
        op_graph.add_node(4, **op4)
        op_graph.add_edge(0, 1, is_control=False)
        op_graph.add_edge(1, 2, is_control=False)
        op_graph.add_edge(0, 3, is_control=False)
        op_graph.add_edge(4, 2, is_control=False)

        grouper_instance = CoplacementGrouper()
        grouper_instance(op_graph)

        self.assertEqual(op_graph.nodes[0]["colocation_group"], "op0")
        self.assertEqual(op_graph.nodes[1]["colocation_group"], "op1")
        self.assertEqual(op_graph.nodes[2]["colocation_group"], "op1")
        self.assertEqual(op_graph.nodes[3]["colocation_group"], "op3")
        self.assertEqual(op_graph.nodes[4]["colocation_group"], "op1")

    def test_control_edge(self):
        op_graph = nx.DiGraph()
        # op0 -> op1
        op0 = {"id": 0, "name": "op0", "colocation_group": "op0"}
        op1 = {"id": 1, "name": "op1", "colocation_group": "op1"}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_edge(0, 1, is_control=True)

        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=True)
        self.assertFalse(is_changed)
        self.assertEqual(op_graph.nodes[0]["colocation_group"], "op0")
        self.assertEqual(op_graph.nodes[1]["colocation_group"], "op1")

        is_changed = CoplacementGrouper._run_colocation_step(
            op_graph, ignore_control_edges=False)
        self.assertTrue(is_changed)
        self.assertEqual(op_graph.nodes[0]["colocation_group"], "op0")
        self.assertEqual(op_graph.nodes[1]["colocation_group"], "op0")


if __name__ == "__main__":
    unittest.main()
