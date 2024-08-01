import dataclasses
from typing import List
import os

from model_wrappers import (
    ModelWrapper,
    Llama2Wrapper,
    Llama3Wrapper,
    GPTWrapper
)
from tqdm import tqdm

from data import QuestionAnswerPair, transform_to_question_answer_pair,load_data, save_to_json


def run_critic_eval(
    critic: ModelWrapper,
    judge: ModelWrapper,
    dataset: List[QuestionAnswerPair],
    output_path: str,
):
    results = []
    for item in tqdm(dataset):
        critique = critic.critique(item)
        judge_prob_pre_critique = judge.get_judge_confidence(item)
        judge_prob_post_critique = judge.get_judge_confidence(item, critique=critique)
        results.append(
            {
                "question_answer_pair": dataclasses.asdict(item),
                "critic": critic.model_id,
                "judge": judge.model_id,
                "critique": critique,
                "judge_confidence_pre_critique": judge_prob_pre_critique,
                "judge_confidence_post_critique": judge_prob_post_critique,
            }
        )
        save_to_json(results, output_path)


if __name__ == "__main__":
    train_data, test_data = load_data()
    sample = transform_to_question_answer_pair(train_data[:100])

    gpt4o = GPTWrapper("gpt4o", "gpt-4o-2024-05-13")
    gpt35_turbo = GPTWrapper("gpt35_turbo", "gpt-3.5-turbo-0125")
    llama2_7b = Llama2Wrapper("llama2_7b", "meta-llama/Llama-2-7b-chat-hf")
    llama3_8b = Llama3Wrapper("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    models = [gpt4o, gpt35_turbo, llama2_7b, llama3_8b]
    for i in range(len(models)):
        critic = models[i]
        for judge in models[i + 1 :]:
            output_file = f"no_solver_results/{critic.model_id}_{judge.model_id}.json"
            if not os.path.exists(output_file):
                run_critic_eval(critic, judge, sample, output_file)