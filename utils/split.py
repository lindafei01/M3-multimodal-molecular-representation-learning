# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
from typing import List, Optional
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import logging

class Splitter(object):
    def __init__(self, split_method='5fold_random', seed=42, fold=5):
        self.split_method = split_method
        #self.split_method = '5fold_scaffold'
        self.seed = seed
        self.n_splits = fold
        self.splitter = self._init_split(self.split_method, self.seed)
        self.skf = None

    def _init_split(self, split_method, seed=42):
        if split_method == 'random':
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
        elif split_method == 'scaffold_balanced':
            splitter = GroupKFold(n_splits=self.n_splits)
        else:
            raise ValueError('Unknown splitter method: {}'.format(split_method))

        return splitter

    def split(self, data, target=None, group=None):
        if self.split_method in ['random']:
            self.skf = self.splitter.split(data)
            logging.info(f'random')
        elif self.split_method in ['scaffold_balanced']:
            self.skf = self.splitter.split(data, target, group)
            logging.info(f'scaffold_balanced')
        else:
            raise ValueError('Unknown splitter method: {}'.format(self.split_method))
        return self.skf