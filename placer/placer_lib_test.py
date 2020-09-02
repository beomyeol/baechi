"""Unit tests for placer_lib module."""
# pylint: disable=missing-function-docstring

import unittest
import networkx as nx

from placer import placer_lib


class FusedOpPlacerTest(unittest.TestCase):
    """FusedOpPlacer test."""

    def test_generate_fused_op_graph1(self):
        op_graph = nx.DiGraph()
        # op0 -> op1 -> op2
        op0 = {'id': 0, 'name': 'op0', 'weight': 2, 'temporary_memory': 0,
               'persistent_memory': 5, 'colocation_group': 'group0',
               'output_memory': [1]}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3, 'temporary_memory': 2,
               'persistent_memory': 0, 'colocation_group': 'group0',
               'output_memory': [2]}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3, 'temporary_memory': 4,
               'persistent_memory': 0, 'colocation_group': 'group0',
               'output_memory': []}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(
            0, 1, id=0, weight=1,
            tensor=[{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        op_graph.add_edge(
            1, 2, id=1, weight=2,
            tensor=[{'name': 'op1:0', 'weight': 2, 'num_bytes': 2}])

        # pylint: disable=protected-access
        fused_op_graph = placer_lib.FusedOpPlacer._generate_fused_op_graph(
            op_graph, False)

        # all ops are fused
        self.assertEqual(fused_op_graph.number_of_nodes(), 1)
        fused_op = fused_op_graph.nodes[0]
        self.assertEqual(fused_op['weight'], 8)
        self.assertEqual(fused_op['temporary_memory'], 4)
        self.assertEqual(fused_op['persistent_memory'], 5)
        self.assertEqual(len(fused_op['output_memory']), 0)
        fused_op_names = [op['name'] for op in fused_op['fused_ops']]
        self.assertIn('op1', fused_op_names)
        self.assertIn('op2', fused_op_names)

    def test_generate_fused_op_graph2(self):
        op_graph = nx.DiGraph()
        # op0 -> op1 -> op2
        op0 = {'id': 0, 'name': 'op0', 'weight': 2, 'temporary_memory': 0,
               'persistent_memory': 5, 'colocation_group': 'group0',
               'output_memory': [1]}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3, 'temporary_memory': 2,
               'persistent_memory': 0, 'colocation_group': 'group0',
               'output_memory': [2]}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3, 'temporary_memory': 4,
               'persistent_memory': 0, 'colocation_group': 'group1',
               'output_memory': []}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(
            0, 1, id=0, weight=1,
            tensor=[{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        op_graph.add_edge(
            1, 2, id=1, weight=2,
            tensor=[{'name': 'op1:0', 'weight': 2, 'num_bytes': 2}])

        # pylint: disable=protected-access
        fused_op_graph = placer_lib.FusedOpPlacer._generate_fused_op_graph(
            op_graph, False)

        # op0 and op1 are fused
        self.assertEqual(fused_op_graph.number_of_nodes(), 2)
        fused_op0 = fused_op_graph.nodes[0]
        self.assertEqual(fused_op0['weight'], 5)
        self.assertEqual(fused_op0['temporary_memory'], 2)
        self.assertEqual(fused_op0['persistent_memory'], 5)
        self.assertListEqual(fused_op0['output_memory'], [2])
        fused_op1 = fused_op_graph.nodes[1]
        self.assertEqual(fused_op1['weight'], 3)
        self.assertEqual(fused_op1['temporary_memory'], 4)
        self.assertEqual(fused_op1['persistent_memory'], 0)
        self.assertListEqual(fused_op1['output_memory'], [])
        self.assertEqual(fused_op_graph.number_of_edges(), 1)
        edge = fused_op_graph[0][1]
        self.assertEqual(edge['id'], 0)
        self.assertEqual(edge['weight'], 2)
        self.assertListEqual(edge['tensor'],
                             [{'name': 'op1:0', 'weight': 2, 'num_bytes': 2}])

    def test_generate_fused_op_graph3(self):
        op_graph = nx.DiGraph()
        # op0 -> op2
        # op1 /
        op0 = {'id': 0, 'name': 'op0', 'weight': 2, 'temporary_memory': 0,
               'persistent_memory': 5, 'colocation_group': 'group0',
               'output_memory': [1]}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3, 'temporary_memory': 0,
               'persistent_memory': 3, 'colocation_group': 'group1',
               'output_memory': [2]}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3, 'temporary_memory': 4,
               'persistent_memory': 2, 'colocation_group': 'group0',
               'output_memory': []}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(
            0, 2, id=0, weight=1,
            tensor=[{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        op_graph.add_edge(
            1, 2, id=1, weight=2,
            tensor=[{'name': 'op1:0', 'weight': 2, 'num_bytes': 2}])

        # pylint: disable=protected-access
        fused_op_graph = placer_lib.FusedOpPlacer._generate_fused_op_graph(
            op_graph, False)

        # op0 and op2 are fused.
        self.assertEqual(fused_op_graph.number_of_nodes(), 2)
        fused_op0 = fused_op_graph.nodes[0]
        self.assertEqual(fused_op0['weight'], 5)
        self.assertEqual(fused_op0['temporary_memory'], 4)
        self.assertEqual(fused_op0['persistent_memory'], 7)
        self.assertListEqual(fused_op0['output_memory'], [])
        fused_op1 = fused_op_graph.nodes[1]
        self.assertEqual(fused_op1['weight'], 3)
        self.assertEqual(fused_op1['temporary_memory'], 0)
        self.assertEqual(fused_op1['persistent_memory'], 3)
        self.assertListEqual(fused_op1['output_memory'], [2])
        self.assertEqual(fused_op_graph.number_of_edges(), 1)
        edge = fused_op_graph[1][0]
        self.assertEqual(edge['id'], 0)
        self.assertEqual(edge['weight'], 2)
        self.assertListEqual(edge['tensor'],
                             [{'name': 'op1:0', 'weight': 2, 'num_bytes': 2}])

    def test_generate_fused_op_graph4(self):
        op_graph = nx.DiGraph()
        # op0 -> op2
        # op1 /
        op0 = {'id': 0, 'name': 'op0', 'weight': 2, 'temporary_memory': 0,
               'persistent_memory': 5, 'colocation_group': 'group0',
               'output_memory': [1]}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3, 'temporary_memory': 0,
               'persistent_memory': 3, 'colocation_group': 'group0',
               'output_memory': [2]}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3, 'temporary_memory': 4,
               'persistent_memory': 2, 'colocation_group': 'group0',
               'output_memory': []}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(
            0, 2, id=0, weight=1,
            tensor=[{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        op_graph.add_edge(
            1, 2, id=1, weight=2,
            tensor=[{'name': 'op1:0', 'weight': 2, 'num_bytes': 2}])

        # pylint: disable=protected-access
        fused_op_graph = placer_lib.FusedOpPlacer._generate_fused_op_graph(
            op_graph, False)

        # all ops are fused
        self.assertEqual(fused_op_graph.number_of_nodes(), 1)
        fused_op = fused_op_graph.nodes[0]
        self.assertEqual(fused_op['weight'], 8)
        self.assertEqual(fused_op['temporary_memory'], 4)
        self.assertEqual(fused_op['persistent_memory'], 10)
        self.assertListEqual(fused_op['output_memory'], [])

    def test_generate_fused_op_graph5(self):
        op_graph = nx.DiGraph()
        # op0 -> op2 -> op3 -> op4
        #     op1 /
        op0 = {'id': 0, 'name': 'op0', 'weight': 2, 'temporary_memory': 0,
               'persistent_memory': 5, 'colocation_group': 'group0',
               'output_memory': [1]}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3, 'temporary_memory': 0,
               'persistent_memory': 3, 'colocation_group': 'group0',
               'output_memory': [2]}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3, 'temporary_memory': 4,
               'persistent_memory': 0, 'colocation_group': 'group1',
               'output_memory': [3]}
        op3 = {'id': 3, 'name': 'op4', 'weight': 1, 'temporary_memory': 7,
               'persistent_memory': 3, 'colocation_group': 'group1',
               'output_memory': [4]}
        op4 = {'id': 4, 'name': 'op4', 'weight': 5, 'temporary_memory': 2,
               'persistent_memory': 0, 'colocation_group': 'group0',
               'output_memory': [0]}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_node(3, **op3)
        op_graph.add_node(4, **op4)
        op_graph.add_edge(
            0, 2, id=0, weight=1,
            tensor=[{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        op_graph.add_edge(
            1, 2, id=1, weight=2,
            tensor=[{'name': 'op1:0', 'weight': 2, 'num_bytes': 2}])
        op_graph.add_edge(
            2, 3, id=2, weight=3,
            tensor=[{'name': 'op2:0', 'weight': 3, 'num_bytes': 3}])
        op_graph.add_edge(
            3, 4, id=3, weight=4,
            tensor=[{'name': 'op3:0', 'weight': 4, 'num_bytes': 4}])

        # pylint: disable=protected-access
        fused_op_graph = placer_lib.FusedOpPlacer._generate_fused_op_graph(
            op_graph, False)

        # op2 and op3 are fused.
        self.assertEqual(fused_op_graph.number_of_nodes(), 4)
        self.assertEqual(fused_op_graph.nodes[0], {**op0, 'old_id': 0})
        self.assertEqual(fused_op_graph.nodes[1], {**op1, 'old_id': 1})
        fused_op = fused_op_graph.nodes[2]
        self.assertEqual(fused_op['weight'], 4)
        self.assertEqual(fused_op['temporary_memory'], 7)
        self.assertEqual(fused_op['persistent_memory'], 3)
        self.assertListEqual(fused_op['output_memory'], [4])
        expected_dict = {**op4, 'old_id': 4}
        expected_dict['id'] = 3
        self.assertEqual(fused_op_graph.nodes[3], expected_dict)
        self.assertEqual(fused_op_graph.number_of_edges(), 3)
        edge_ids = set()
        edge_ids.add(fused_op_graph[0][2]['id'])
        self.assertEqual(fused_op_graph[0][2]['weight'], 1)
        self.assertListEqual(
            fused_op_graph[0][2]['tensor'],
            [{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        edge_ids.add(fused_op_graph[1][2]['id'])
        self.assertEqual(fused_op_graph[1][2]['weight'], 2)
        self.assertListEqual(
            fused_op_graph[1][2]['tensor'],
            [{'name': 'op1:0', 'weight': 2, 'num_bytes': 2}])
        edge_ids.add(fused_op_graph[2][3]['id'])
        self.assertEqual(fused_op_graph[2][3]['weight'], 4)
        self.assertListEqual(
            fused_op_graph[2][3]['tensor'],
            [{'name': 'op3:0', 'weight': 4, 'num_bytes': 4}])
        self.assertSetEqual(edge_ids, set(list(range(3))))

    def test_generate_fused_op_graph6(self):
        op_graph = nx.DiGraph()
        #    -> op2 -> op3
        #   /           /
        # op0 -> op1 ->/
        op0 = {'id': 0, 'name': 'op0', 'weight': 2, 'temporary_memory': 0,
               'persistent_memory': 5, 'colocation_group': 'group0',
               'output_memory': [1, 2]}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3, 'temporary_memory': 5,
               'persistent_memory': 3, 'colocation_group': 'group0',
               'output_memory': [3]}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3, 'temporary_memory': 2,
               'persistent_memory': 2, 'colocation_group': 'group1',
               'output_memory': [4]}
        op3 = {'id': 3, 'name': 'op4', 'weight': 1, 'temporary_memory': 7,
               'persistent_memory': 0, 'colocation_group': 'group1',
               'output_memory': []}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_node(3, **op3)
        op_graph.add_edge(
            0, 1, id=0, weight=1,
            tensor=[{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        op_graph.add_edge(
            0, 2, id=1, weight=2,
            tensor=[{'name': 'op0:1', 'weight': 2, 'num_bytes': 2}])
        op_graph.add_edge(
            1, 3, id=2, weight=3,
            tensor=[{'name': 'op1:0', 'weight': 3, 'num_bytes': 3}])
        op_graph.add_edge(
            2, 3, id=3, weight=4,
            tensor=[{'name': 'op2:0', 'weight': 4, 'num_bytes': 4}])

        # pylint: disable=protected-access
        fused_op_graph = placer_lib.FusedOpPlacer._generate_fused_op_graph(
            op_graph, False)

        self.assertEqual(fused_op_graph.number_of_nodes(), 2)
        fused_op0 = fused_op_graph.nodes[0]  # op0, op1
        self.assertEqual(fused_op0['weight'], 5)
        self.assertEqual(fused_op0['temporary_memory'], 5)
        self.assertEqual(fused_op0['persistent_memory'], 8)
        self.assertEqual(sum(fused_op0['output_memory']), 5)
        fused_op1 = fused_op_graph.nodes[1]  # op2, op3
        self.assertEqual(fused_op1['weight'], 4)
        self.assertEqual(fused_op1['temporary_memory'], 7)
        self.assertEqual(fused_op1['persistent_memory'], 2)
        self.assertListEqual(fused_op1['output_memory'], [])
        self.assertEqual(fused_op_graph.number_of_edges(), 1)
        fused_edge = fused_op_graph[0][1]
        self.assertEqual(fused_edge['id'], 0)
        self.assertEqual(fused_edge['weight'], 5)
        self.assertListEqual(
            fused_edge['tensor'],
            [{'name': 'op0:1', 'weight': 2, 'num_bytes': 2},
             {'name': 'op1:0', 'weight': 3, 'num_bytes': 3}])

    def test_generate_fused_op_graph7(self):
        op_graph = nx.DiGraph()
        #    -> op2 -> op3
        #   /           /
        # op0 -> op1 <-/
        op0 = {'id': 0, 'name': 'op0', 'weight': 2, 'temporary_memory': 0,
               'persistent_memory': 5, 'colocation_group': 'group0',
               'output_memory': [1, 2]}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3, 'temporary_memory': 5,
               'persistent_memory': 3, 'colocation_group': 'group0',
               'output_memory': []}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3, 'temporary_memory': 2,
               'persistent_memory': 2, 'colocation_group': 'group1',
               'output_memory': [4]}
        op3 = {'id': 3, 'name': 'op4', 'weight': 1, 'temporary_memory': 7,
               'persistent_memory': 0, 'colocation_group': 'group1',
               'output_memory': [3]}
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_node(3, **op3)
        op_graph.add_edge(
            0, 1, id=0, weight=1,
            tensor=[{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        op_graph.add_edge(
            0, 2, id=1, weight=2,
            tensor=[{'name': 'op0:1', 'weight': 2, 'num_bytes': 2}])
        op_graph.add_edge(
            3, 1, id=2, weight=3,
            tensor=[{'name': 'op3:0', 'weight': 3, 'num_bytes': 3}])
        op_graph.add_edge(
            2, 3, id=3, weight=4,
            tensor=[{'name': 'op2:0', 'weight': 4, 'num_bytes': 4}])

        # pylint: disable=protected-access
        fused_op_graph = placer_lib.FusedOpPlacer._generate_fused_op_graph(
            op_graph, False)

        self.assertEqual(fused_op_graph.number_of_nodes(), 3)
        self.assertEqual(fused_op_graph.nodes[0], {**op0, 'old_id': 0})
        self.assertEqual(fused_op_graph.nodes[1], {**op1, 'old_id': 1})
        fused_op = fused_op_graph.nodes[2]  # op2, op3
        self.assertEqual(fused_op['weight'], 4)
        self.assertEqual(fused_op['temporary_memory'], 7)
        self.assertEqual(fused_op['persistent_memory'], 2)
        self.assertEqual(sum(fused_op['output_memory']), 3)
        self.assertEqual(fused_op_graph.number_of_edges(), 3)
        self.assertEqual(fused_op_graph[0][1]['weight'], 1)
        self.assertListEqual(
            fused_op_graph[0][1]['tensor'],
            [{'name': 'op0:0', 'weight': 1, 'num_bytes': 1}])
        self.assertEqual(fused_op_graph[0][2]['weight'], 2)
        self.assertListEqual(
            fused_op_graph[0][2]['tensor'],
            [{'name': 'op0:1', 'weight': 2, 'num_bytes': 2}])
        self.assertEqual(fused_op_graph[2][1]['weight'], 3)
        self.assertListEqual(
            fused_op_graph[2][1]['tensor'],
            [{'name': 'op3:0', 'weight': 3, 'num_bytes': 3}])
        self.assertSetEqual(
            {edge[-1] for edge in fused_op_graph.edges(data='id')},
            set(range(3)))


if __name__ == "__main__":
    unittest.main()
