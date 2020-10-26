# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Device test module."""
# pylint: disable=missing-class-docstring, missing-function-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import networkx as nx

from placer.device import DeviceWrapper, DeviceWrapperAllocator


class MemoryTest(unittest.TestCase):

    def test_simple(self):
        op_graph = nx.DiGraph()
        op_graph.add_node(0, temporary_memory=10, persistent_memory=0,
                          output_memory=[5])
        device_graph = nx.DiGraph()
        device_graph.add_node(0, size=0, memory_limit=12)
        op_data = op_graph.nodes[0]

        device = DeviceWrapper(0, device_graph, op_graph)
        self.assertEqual(device.get_op_memory(op_data), 10)
        self.assertTrue(device.is_placeable(op_data))

        device = DeviceWrapperAllocator(0, device_graph, op_graph)
        self.assertEqual(device.get_op_memory(op_data), 15)
        self.assertFalse(device.is_placeable(op_data))


if __name__ == "__main__":
    unittest.main()
