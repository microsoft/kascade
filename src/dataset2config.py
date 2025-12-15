# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from kascade.metrics import *
from kascade.qadatasets import *

dataset2class = {
    "framolfese/2WikiMultihopQA": TwoWikiDataset,
    "hotpotqa/hotpot_qa": HotPotDataset,
    "bdsaglam/musique": MusiqueDataset,
    "THUDM/LongBench": LongBenchMultihopDataset,
    "HuggingFaceH4/aime_2024": MathDataset,
    "HuggingFaceH4/MATH-500": MathDataset,
}

dataset2split = {
    "framolfese/2WikiMultihopQA": 'validation',
    "hotpotqa/hotpot_qa": 'validation',
    "bdsaglam/musique": 'validation',
    "THUDM/LongBench": 'test',
    "HuggingFaceH4/aime_2024": 'train',
    "HuggingFaceH4/MATH-500": 'test',
}

dataset2max_batch_size = {
    "framolfese/2WikiMultihopQA": 4,
    "hotpotqa/hotpot_qa": 4,
    "bdsaglam/musique": 4,
    "THUDM/LongBench": 1,
    "HuggingFaceH4/aime_2024": 4,
    "HuggingFaceH4/MATH-500": 32,
}

dataset2max_new_tokens = {
        "narrativeqa": 128,
        "qasper": 128,
        "multifieldqa_en": 64,
        "multifieldqa_zh": 64,
        "hotpotqa": 32,
        "2wikimqa": 32,
        "musique": 32,
        "dureader": 128,
        "gov_report": 512,
        "qmsum": 512,
        "multi_news": 512,
        "vcsum": 512,
        "trec": 64,
        "triviaqa": 32,
        "samsum": 128,
        "lsht": 64,
        "passage_count": 32,
        "passage_retrieval_en": 32,
        "passage_retrieval_zh": 32,
        "lcc": 64,
        "repobench-p": 64,
        "framolfese/2WikiMultihopQA": 32,
        "hotpotqa/hotpot_qa": 32,
        "bdsaglam/musique": 32,
        "HuggingFaceH4/aime_2024": 32768,
        "HuggingFaceH4/MATH-500": 32768,
}

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "framolfese/2WikiMultihopQA": qa_f1_score,
    "hotpotqa/hotpot_qa": qa_f1_score,
    "bdsaglam/musique": qa_f1_score,
    "HuggingFaceH4/aime_2024": math_score,
    "HuggingFaceH4/MATH-500": math_score,
}

dataset2prompt_format = {
    "narrativeqa": 1,
    "qasper": 2,
    "multifieldqa_en": 3,
    "multifieldqa_zh": 4,
    "hotpotqa": 0,
    "2wikimqa": 0,
    "musique": 0,
    "dureader": 5,
    "gov_report": 6,
    "qmsum": 7,
    "multi_news": 8,
    "vcsum": 9,
    "trec": 10,
    "triviaqa": 11,
    "samsum": 12,
    "lsht": 13,
    "passage_retrieval_en": 14,
    "passage_count": 15,
    "passage_retrieval_zh": 16,
    "lcc": 17,
    "repobench-p": 18,
    "framolfese/2WikiMultihopQA": 0,
    "hotpotqa/hotpot_qa": 0,
    "bdsaglam/musique": 0,
    "HuggingFaceH4/aime_2024": 20,
    "HuggingFaceH4/MATH-500": 20,
}
