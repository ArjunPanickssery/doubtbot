import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class Answer:
    numeric: float
    proof: str


@dataclass
class QuestionAnswerPair:
    question: str
    answer_numeric: float
    answer_proof: str
    correct: bool = None


@dataclass
class DatasetItem:
    question: str
    answer_correct: Answer
    answer_incorrect: Answer


def save_to_json(dictionary, file_name):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def transform_to_dataset_item(data: List[dict]) -> List[DatasetItem]:
    """Transforms a list of dicts into a list of DatasetItems"""
    return [
        DatasetItem(
            question=item["question"],
            answer_correct=Answer(
                proof=item["answer_correct"]["proof"],
                numeric=item["answer_correct"]["numeric"],
            ),
            answer_incorrect=Answer(
                proof=item["answer_incorrect"]["proof"],
                numeric=item["answer_incorrect"]["numeric"],
            ),
        )
        for item in data
        if isinstance(item["answer_incorrect"], dict)
    ]


def transform_to_question_answer_pair(
    data: List[DatasetItem],
    *,
    include_correct: bool = True,
    include_incorrect: bool = True,
) -> List[QuestionAnswerPair]:
    """Transforms a list of DatasetItems into a list of QuestionAnswerPairs"""
    result = []

    for item in data:
        if include_correct:
            result.append(
                QuestionAnswerPair(
                    question=item.question,
                    answer_numeric=item.answer_correct.numeric,
                    answer_proof=item.answer_correct.proof,
                    correct=True,
                )
            )

        if include_incorrect:
            result.append(
                QuestionAnswerPair(
                    question=item.question,
                    answer_numeric=item.answer_incorrect.numeric,
                    answer_proof=item.answer_incorrect.proof,
                    correct=False,
                )
            )

    return result


def load_data() -> tuple[List[DatasetItem], List[DatasetItem]]:
    train_data_raw = load_from_json("data/train_data.json")
    test_data_raw = load_from_json("data/test_data.json")
    train_data = transform_to_dataset_item(train_data_raw)
    test_data = transform_to_dataset_item(test_data_raw)
    return train_data, test_data
