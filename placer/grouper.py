# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Grouper module."""
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tensorflow as tf

from placer import placer_utils as utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum('grouper', 'tf', ['tf', 'coplace'],
                         'Grouping algorithm')


class Grouper(object):
    """Default grouper that does nothing."""

    def __call__(self, op_graph):
        raise NotImplementedError()


def _len_and_str(string):
    """Returns a tuple of string length and string."""
    return (len(string), string)


def _update_colocation_group(op_graph, colocation_group_map):
    """Updates colocation groups of operators in op_graph with new mapping."""
    # pick the shortest group name among groups as new group name
    group_dict = {group: min(group_set, key=_len_and_str)
                  for group, group_set in colocation_group_map.items()}

    # print merged groups
    reverse_mapping = {}
    for prev_name, new_name in group_dict.items():
        if new_name in reverse_mapping:
            reverse_mapping[new_name].append(prev_name)
        else:
            reverse_mapping[new_name] = [prev_name]
    for new_name, prev_names in reverse_mapping.items():
        _LOGGER.debug('Change group: %s -> %s', sorted(prev_names), new_name)

    # update colocation group
    for _, op_data in op_graph.nodes.items():
        if isinstance(op_data['colocation_group'], list):
            new_group = None
            for colocation_group in op_data['colocation_group']:
                ret = group_dict.get(colocation_group, colocation_group)
                if new_group is None:
                    new_group = ret
                else:
                    assert new_group == ret, 'node=%s, cur=%s, new=%s' % (
                        op_data['name'], new_group, ret)
        else:
            prev_group_name = op_data['colocation_group']
            new_group = group_dict.get(prev_group_name, prev_group_name)

        op_data['colocation_group'] = new_group


def process_colocation_group(op_graph):
    """Process a list of colocations groups into a single colocation group."""

    # This maps a colocation group name to a set of other group names
    colocation_group_map = utils.ColocationGroupMap()

    for _, op_data in op_graph.nodes.items():
        colocation_group = op_data['colocation_group']
        for op1, op2 in itertools.combinations(colocation_group, 2):
            colocation_group_map.colocate(op1, op2)

    _update_colocation_group(op_graph, colocation_group_map)


class TFColocationGrouper(Grouper):
    """Generate a new graph by using TensorFlow colocation group information."""

    def __call__(self, op_graph):
        # use the existing colocation group information
        process_colocation_group(op_graph)
        return op_graph


class CoplacementGrouper(Grouper):
    """Generate a new graph by using heuristic at the paper."""

    def __init__(self, log_colocation_group=False, ignore_control_edges=False):
        super(CoplacementGrouper, self).__init__()
        self._log_colocation_group = log_colocation_group
        self._ignore_control_edges = ignore_control_edges

    @staticmethod
    def _run_colocation_step(op_graph, ignore_control_edges):
        """Check whether there are operators that can be co-located.

        When the output of an operator is consumed only by another operator,
        assign the same colocation group for them

        Returns:
          True if there are opeartors that can be co-located.
          False, otherwise.
        """
        colocation_candidates = utils.ColocationGroupMap()

        for op_id, op_data in op_graph.nodes.items():
            # TODO: should we consider tensor-wise? not operator wise?
            out_edges = list(op_graph.out_edges(op_id))
            if len(out_edges) != 1:
                continue

            next_op_id = out_edges[0][1]

            # pass control edges because this does not have data transfer
            edge_data = op_graph.get_edge_data(op_id, next_op_id)
            if ignore_control_edges and edge_data["is_control"]:
                continue

            next_op_data = op_graph.nodes[next_op_id]

            op_group = op_data['colocation_group']
            next_op_group = next_op_data['colocation_group']

            if op_group != next_op_group:
                # these two can be colocated
                _LOGGER.debug('Possible colocation ops. %s[%s] -> %s[%s]',
                              op_data['name'],
                              op_group,
                              next_op_data['name'],
                              next_op_group)
                colocation_candidates.colocate(op_group, next_op_group)

        if len(colocation_candidates) > 0:
            _update_colocation_group(op_graph, colocation_candidates)
            return True

        return False

    def __call__(self, op_graph):
        process_colocation_group(op_graph)
        # first use default colocation group information
        if self._log_colocation_group:
            with open('tf_colocation_groups.log', 'w') as f:
                utils.print_colocation_group(
                    op_graph, print_cb=lambda v: f.write(v + '\n'))
        while self._run_colocation_step(op_graph, self._ignore_control_edges):
            pass
        if self._log_colocation_group:
            with open('coplaced_groups.log', 'w') as f:
                utils.print_colocation_group(
                    op_graph, print_cb=lambda v: f.write(v + '\n'))
        return op_graph


_GROUPER_CLASS_MAP = {
    'none': Grouper,
    'tf': TFColocationGrouper,
    'coplace': CoplacementGrouper,
}


def get_grouper(grouper=None):
    """Generates and returns a grouper instance."""
    grouper = grouper or FLAGS.grouper
    _LOGGER.info('Grouper: %s', grouper)
    grouper_class = _GROUPER_CLASS_MAP[grouper]
    return grouper_class()
