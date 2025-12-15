# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .TwoWikiDataset import TwoWikiDataset 

class HotPotDataset(TwoWikiDataset):
    sentence_column_name = 'sentences'