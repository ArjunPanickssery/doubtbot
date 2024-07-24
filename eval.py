import dataclasses
from typing import List

from model_wrappers import (
    HuggingFaceWrapper,
    Llama2Wrapper,
    Llama3Wrapper,
    ModelWrapper,
)
from tqdm import tqdm

from data import QuestionAnswerPair, load_data, save_to_json


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
    # print(train_data[1])
    # raise Exception("stop")
    # critic = Llama2Wrapper("llama2_7b", "meta-llama/Llama-2-7b-chat-hf")
    critic = judge = Llama3Wrapper("llama3_8b", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    run_critic_eval(critic, judge, train_data, "results_v2.json")
