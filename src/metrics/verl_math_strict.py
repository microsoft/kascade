"""
This logic is largely taken from the verl math evaluation code and modified.
"""

import re


_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def _normalize_answer(answer):
    return str(answer).replace(",", "").replace("$", "").replace(" ", "").strip()


def _strip_think_tags(solution_str):
    return _THINK_TAG_PATTERN.sub("", solution_str)


def _remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]

    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def _extract_boxed_answer(solution_str):
    visible_solution = _strip_think_tags(solution_str)
    boxed_answer = _last_boxed_only_string(visible_solution)
    if boxed_answer is None:
        return None

    try:
        answer = _normalize_answer(_remove_boxed(boxed_answer))
    except (AssertionError, Exception):
        return None

    return answer if answer else None


def compute_chained_score(
    data_source,
    solution_str,
    ground_truth,
    format_score=0.0,
    score=1.0,
    **kwargs,
):
    if ground_truth is None:
        return 0.0

    answer = _extract_boxed_answer(solution_str)
    if answer is None:
        return 0.0

    normalized_gt = _normalize_answer(ground_truth)
    if answer == normalized_gt:
        return float(score)

    numeric_matches = re.findall(r"-?\d+\.?\d*", answer)
    if numeric_matches and numeric_matches[-1] == normalized_gt:
        return float(score)

    return float(format_score)


def math_score_verl_strict(prediction, ground_truth, **kwargs):
    return int(
        compute_chained_score(
            data_source="math",
            solution_str=prediction,
            ground_truth=ground_truth,
            **kwargs,
        )
    )

