# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, Union, List
from .QADataset import QADataset
import re

class LongBenchMultihopDataset(QADataset):
    context_column_name = 'context'
    question_column_name = 'input'
    answer_column_name = 'answers'
    def get_answer(self, key: int | slice | Iterable[int]) -> List[str]:
        answers = self.hf_dataset[key][self.answer_column_name]
        all_classes = self.hf_dataset[key]["all_classes"]
        if isinstance(key, int):
            answers = [answers]
            all_classes = [all_classes]
        return (answers, all_classes)

    def get_context(self, key: Union[int, slice, Iterable[int]]) -> List[str]:
        contexts = self.hf_dataset[key][self.context_column_name]
        if isinstance(key, int):
            return [contexts]
        return contexts