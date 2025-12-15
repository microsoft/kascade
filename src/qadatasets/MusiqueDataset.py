# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, Union, List
from .TwoWikiDataset import TwoWikiDataset   

class MusiqueDataset(TwoWikiDataset):
    context_column_name = 'paragraphs'
    question_column_name = 'question'
    answer_column_name = 'answer'
    paragraph_column_name = 'paragraph_text'

    def get_context(self, key: Union[int, slice, Iterable[int]]) -> List[str]:
        if isinstance(key, int):
            key = [key]
        contexts = []
        for k in key:
            context = []
            for doc in self.hf_dataset[k][self.context_column_name]:
                context.append(f"\n\n{doc[self.paragraph_column_name]}\n\n")
            contexts.append("".join(context))
        return contexts

