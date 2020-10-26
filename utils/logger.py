# Copyright 2020 University of Illinois Board of Trustees. All Rights Reserved.
# Author: Beomyeol Jeon, DPRG (https://dprg.cs.uiuc.edu)
# This file is part of Baechi, which is released under specific terms. See file License.txt file for full license details.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

CRITICAL=logging.CRITICAL
ERROR=logging.ERROR
WARNING=logging.WARNING
INFO=logging.INFO
DEBUG=logging.DEBUG

def get_logger(filepath, level=INFO):
    name = os.path.splitext(os.path.basename(filepath))[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger
