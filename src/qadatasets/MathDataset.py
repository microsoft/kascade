# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, Union, List
from .QADataset import QADataset 

class MathDataset(QADataset):
    question_column_name = 'problem'
    answer_column_name = 'answer'
    
    def get_context(self, key: Union[int, slice, Iterable[int]]) -> List[str]:
        if isinstance(key, int):
            return [""]
        else:
            return [""] * len(self.hf_dataset[key]['problem'])  