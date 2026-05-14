# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, Union, List

from .QADataset import QADataset


class GSMLongHorizonDataset(QADataset):
    question_column_name = "question"
    answer_column_name = "final_answer"

    @classmethod
    def parse_horizon_subset(cls, subset: str) -> int:
        try:
            horizon_length = int(subset)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid horizon subset '{subset}'. For {cls.dataset_name}, "
                "subsets must be integers in [1, 5]."
            ) from None

        if horizon_length < 1 or horizon_length > 5:
            raise ValueError(
                f"Invalid horizon subset '{subset}'. For {cls.dataset_name}, "
                "subsets must be integers in [1, 5]."
            )
        return horizon_length

    @classmethod
    def filter_dataset_for_subset(cls, dataset, subset: str):
        horizon_length = cls.parse_horizon_subset(subset)
        return dataset.filter(lambda row, h=horizon_length: row["horizon_length"] == h)

    @classmethod
    def subset_label(cls, subset: str | None, default_split: str) -> str:
        if subset is None:
            return default_split
        return f"horizon_{subset}"

    def get_context(self, key: Union[int, slice, Iterable[int]]) -> List[str]:
        if isinstance(key, int):
            return [""]
        return [""] * len(self.hf_dataset[key][self.question_column_name])
