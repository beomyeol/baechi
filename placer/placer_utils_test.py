# pylint: disable=missing-function-docstring,missing-class-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import networkx as nx

from placer.placer_utils import get_ready_time
from placer.placer_utils import TimestampOpTuple
from placer.placer_utils import SortedTimestampOps
from placer.placer_utils import prune_non_important_ops


class PlacerUtilsTest(unittest.TestCase):

    def test_get_ready_time(self):
        # op_graph
        # op1 (end_time=5,p=0) -(w=2)-> op3
        #                           /
        #                         / (w=5)
        # op2 (end_time=10,p=1) /
        op_graph = nx.DiGraph()
        op_graph.add_node(1, id=1, name='op1', end_time=5, p=0)
        op_graph.add_node(2, id=2, name='op2', end_time=10, p=1)
        op_graph.add_node(3, id=3, name='op3')
        op_graph.add_edge(1, 3, weight=2)
        op_graph.add_edge(2, 3, weight=5)

        self.assertEqual(
            get_ready_time(op_graph, target_op_id=3, device_id=0), 15)

        self.assertEqual(
            get_ready_time(op_graph, target_op_id=3, device_id=1), 10)


class TimestampOpTupleTest(unittest.TestCase):

    def test_lt(self):
        tuple1 = TimestampOpTuple(0, {'id': 1, 'name': 'op1', 'memory': 5})
        tuple2 = TimestampOpTuple(0, {'id': 2, 'name': 'op2', 'memory': 5})
        self.assertLess(tuple1, tuple2)


class SortedTimestampOpsTest(unittest.TestCase):

    def test_add_remove_op(self):
        sorted_ts_ops = SortedTimestampOps()
        op1 = {'id': 1, 'name': 'op1'}
        op2 = {'id': 2, 'name': 'op2'}
        op3 = {'id': 3, 'name': 'op3'}

        sorted_ts_ops.add_op(1, op2)
        sorted_ts_ops.add_op(0, op1)
        sorted_ts_ops.add_op(2, op3)

        self.assertEqual(len(sorted_ts_ops), 3)
        sorted_ts_ops.remove_op(op2)
        self.assertEqual(len(sorted_ts_ops), 2)

        ts_op_tuple = sorted_ts_ops[0]
        self.assertEqual(ts_op_tuple.ts, 0)
        self.assertEqual(ts_op_tuple.op['name'], 'op1')

        ts_op_tuple = sorted_ts_ops.pop(0)
        self.assertEqual(ts_op_tuple.ts, 0)
        self.assertEqual(ts_op_tuple.op['name'], 'op1')

        ts_op_tuple = sorted_ts_ops.pop(0)
        self.assertEqual(ts_op_tuple.ts, 2)
        self.assertEqual(ts_op_tuple.op['name'], 'op3')


class PruneNonImportantOpsTest(unittest.TestCase):

    def test_simple(self):
        op_graph = nx.DiGraph()
        # op1 -> op2 -> op3
        #        / \
        # op4- >/   \-> op5
        op_graph.add_node(1, id=1, name='op1', weight=1, memory=1)
        op_graph.add_node(2, id=2, name='op2', weight=2, memory=2)
        op_graph.add_node(3, id=3, name='op3', weight=3, memory=3)
        op_graph.add_node(4, id=4, name='op4', weight=4, memory=4)
        op_graph.add_node(5, id=5, name='op5', weight=5, memory=5)
        op_graph.add_edge(1, 2, id=1, weight=1)
        op_graph.add_edge(2, 3, id=2, weight=2)
        op_graph.add_edge(4, 2, id=3, weight=3)
        op_graph.add_edge(2, 5, id=4, weight=4)

        important_ops = ['op2', 'op3', 'op4']
        important_op_graph, important_op_index = prune_non_important_ops(
            op_graph, important_ops)

        self.assertListEqual(list(important_op_index.keys()), important_ops)
        self.assertListEqual(list(important_op_index.values()), [0, 1, 2])

        self.assertEqual(important_op_graph.number_of_nodes(), 3)
        self.assertListEqual(list(important_op_graph.nodes()), [0, 1, 2])
        op2_id = important_op_index['op2']
        op2 = important_op_graph.nodes[op2_id]
        self.assertDictEqual(
            op2, {'id': op2_id, 'name': 'op2', 'weight': 2, 'memory': 2})
        op3_id = important_op_index['op3']
        op3 = important_op_graph.nodes[op3_id]
        self.assertDictEqual(
            op3, {'id': op3_id, 'name': 'op3', 'weight': 3, 'memory': 3})
        op4_id = important_op_index['op4']
        op4 = important_op_graph.nodes[op4_id]
        self.assertDictEqual(
            op4, {'id': op4_id, 'name': 'op4', 'weight': 4, 'memory': 4})
        self.assertEqual(important_op_graph.number_of_edges(), 2)
        edge1 = important_op_graph[op4_id][op2_id]
        edge1_id = edge1['id']
        self.assertIn(edge1_id, [0, 1])
        self.assertEqual(edge1['weight'], 3)
        edge2 = important_op_graph[op2_id][op3_id]
        edge2_id = edge2['id']
        self.assertEqual(edge2['weight'], 2)
        self.assertEqual(edge2_id, 0 if edge1_id == 1 else 0)


if __name__ == "__main__":
    unittest.main()
