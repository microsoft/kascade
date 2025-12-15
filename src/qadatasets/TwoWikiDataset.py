# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, Union, List
from .QADataset import QADataset

class TwoWikiDataset(QADataset):
    context_column_name = 'context'
    question_column_name = 'question'
    answer_column_name = 'answer'
    sentence_column_name = 'sentences'
    def get_context(self, key: Union[int, slice, Iterable[int]]) -> List[str]:
        if isinstance(key, int):
            key = [key]
        contexts = []
        for k in key:
            context = []
            for doc in self.hf_dataset[k][self.context_column_name][self.sentence_column_name]:
                doc = " ".join(doc)
                context.append(f"\n\n{doc}\n\n")
            contexts.append("".join(context))
        return contexts
        

