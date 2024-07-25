from typing import List

from tqdm import tqdm

from data import (
    QuestionAnswerPair,
    load_data,
    save_to_json,
    transform_to_question_answer_pair,
)
from model_wrappers import (
    Llama2Wrapper,
    Llama3Wrapper,
)


def evaluate_baseline(
    model: Llama3Wrapper, dataset: List[QuestionAnswerPair], output_file: str
):
    results = []
    for i, item in enumerate(tqdm(dataset)):
        print(f"Evaluating item {i} / {len(dataset)}: {item.question}")
        try:
            predicted_answer, predicted_proof = model.answer_question(item.question)
            is_correct = abs(predicted_answer - item.answer_numeric) < 1e-6
        except Exception as e:
            print(f"Error: {e}")
            predicted_answer = None
            predicted_proof = None
            is_correct = False

        result = {
            "question": item.question,
            "correct_answer": item.answer_numeric,
            "correct_proof": item.answer_proof,
            "predicted_answer": predicted_answer,
            "predicted_proof": predicted_proof,
            "is_correct": is_correct,
        }
        results.append(result)

    save_to_json(results, output_file)

    accuracy = sum(1 for r in results if r["is_correct"]) / len(results)

    return accuracy


if __name__ == "__main__":
    train_data, test_data = load_data()

    # model = Llama2Wrapper("llama2_7b", "meta-llama/Llama-2-7b-chat-hf")
    model = Llama3Wrapper("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct")
    question_answer_pairs = transform_to_question_answer_pair(
        train_data, include_incorrect=False
    )
    # model = WizardMathWrapper("wizard_7b", "WizardLMTeam/WizardMath-7B-V1.1")

    print("Evaluating on a subset of training data...")
    train_accuracy = evaluate_baseline(
        model, question_answer_pairs, "results/answerer_baseline_train_subset.json"
    )
    print(f"Baseline accuracy on training subset: {train_accuracy:.2%}")
