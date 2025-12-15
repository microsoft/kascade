# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datasets import Dataset
from typing import Iterable, List, Union

class QADataset:
    def __init__(self, hf_dataset: Dataset):
        self.hf_dataset = hf_dataset    
    def __len__(self):
        return len(self.hf_dataset)
    def get_question(self, key: Union[int, slice, Iterable[int]]) -> List[str]:
        return self.hf_dataset[key][self.question_column_name]
    def get_context(self, key: Union[int, slice, Iterable[int]]) -> List[str]:
        raise NotImplementedError
    def get_answer(self, key: Union[int, slice, Iterable[int]]) -> List[str]:
        return self.hf_dataset[key][self.answer_column_name]

    