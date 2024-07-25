import dataclasses
from typing import List
from model_wrappers import (
    HuggingFaceWrapper,
    Llama2Wrapper,
    Llama3Wrapper,
    ModelWrapper,
    WizardMathWrapper,
)
from tqdm import tqdm
import json
from typing import List
from tqdm import tqdm
from model_wrappers import Llama3Wrapper
from data import (
    QuestionAnswerPair,
    load_data,
    save_to_json,
    transform_to_question_answer_pair,
)


def evaluate_baseline(
    model: Llama3Wrapper, dataset: List[QuestionAnswerPair], output_file: str
):
    results = []
    for i, item in enumerate(tqdm(dataset)):
        print(f"Evaluating item {i} / {len(dataset)}: {item.question}")
        # predicted_answer, predicted_proof = model.answer_question(item.question)
        resp = model.answer_question(item.question)

        result = {
            "question": item.question,
            "correct_answer": item.answer_numeric,
            "correct_proof": item.answer_proof,
            "predicted_answer": resp,
            # "predicted_answer": predicted_answer,
            # "predicted_proof": predicted_proof,
            # "is_correct": abs(predicted_answer - item.answer_numeric) < 1e-6,
        }
        results.append(result)

    save_to_json(results, output_file)

    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)

    return accuracy


if __name__ == "__main__":
    train_data, test_data = load_data()

    model = Llama2Wrapper("llama2_7b", "meta-llama/Llama-2-7b-chat-hf")
    question_answer_pairs = transform_to_question_answer_pair(train_data)
    # model = WizardMathWrapper("wizard_7b", "WizardLMTeam/WizardMath-7B-V1.1")

    print("Evaluating on a subset of training data...")
    train_accuracy = evaluate_baseline(
        model, question_answer_pairs[:10], "results/answerer_baseline_train_subset.json"
    )
    print(f"Baseline accuracy on training subset: {train_accuracy:.2%}")
