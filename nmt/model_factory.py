# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
"""Model factory for nmt models."""
from __future__ import absolute_import, division, print_function

import functools

from nmt import attention_nmt, gnmt, nmt

_MODEL_NAME = {
    'nmt': nmt.nmt,
    'separate_cell_nmt': nmt.separate_cell_nmt,
    'keras_nmt': nmt.keras_nmt,
    'attention_nmt': attention_nmt.attention_nmt,
    'gnmt': gnmt.gnmt,
    'gnmt_v2': gnmt.gnmt_v2,
}


def get_model_fn(name, **kwargs):
    """Returns a model function with the given spec."""
    if name not in _MODEL_NAME:
        raise ValueError('Unknown model name: %s' % name)
    func = _MODEL_NAME[name]
    return functools.partial(func, **kwargs)
