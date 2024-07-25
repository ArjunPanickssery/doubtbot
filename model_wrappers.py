import os
from typing import Tuple
from timeit import default_timer as timer
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import requests
from data import DatasetItem
import re

from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_LENGTH = 1024


def parse_answer_and_proof(content: str) -> Tuple[float, str]:
    proof_match = re.search(r"<proof>(.*?)</proof>", content, re.DOTALL)
    if not proof_match:
        raise RuntimeError("Could not find proof in the response")
    # Sometimes the LLM misses off the closing tag, so we take the answer even if it ends in whitespace then the end of the string
    answer_match = re.search(
        r"<numeric_answer>(.*?)(?:</numeric_answer>|\s*$)", content, re.DOTALL
    )
    if not answer_match:
        raise RuntimeError("Could not find numeric answer in the response")

    proof = proof_match.group(1).strip()
    try:
        numeric_answer = float(answer_match.group(1).strip())
    except ValueError:
        raise RuntimeError(f"Could not parse numeric answer: {answer_match.group(1)}")

    return numeric_answer, proof


class ModelWrapper:
    def __init__(self, model_id: str, model_name):
        """
        Initializes the LLM debater with the specified model.

        Args:
            model_id (str): A short identifier for the model (e.g., "llama2_7b")
            model_name (str): The name of the model to load from HF/API
        """
        self.model_id = model_id
        self.model_name = model_name

    def critique(
        self,
        item: DatasetItem,
    ) -> str:
        """
        Generates a critique for the given DatasetItem.

        Args:
            item (DatasetItem): The dataset item to critique

        Returns:
            str: The generated critique

        Raises:
            NotImplementedError: This method should be implemented by subclasses
            ValueError: If the item is invalid or missing required fields
        """
        raise NotImplementedError

    def get_judge_confidence(
        self,
        item: DatasetItem,
        critique: str = None,
    ) -> str:
        """
        Determines the judge's confidence in the correctness of the given DatasetItem's solution.

        Args:
            item (DatasetItem): The dataset item to judge
            critique (str, optional): A critique of the solution. Defaults to None.

        Returns:
            str: The judge's confidence level

        Raises:
            NotImplementedError: This method should be implemented by subclasses
            ValueError: If the item is invalid or missing required fields
        """
        raise NotImplementedError

    def answer_question(self, question: str) -> Tuple[float, str]:
        """
        Generates an answer to the given question.

        Args:
            question (str): The question to answer

        Returns:
            Tuple[float, str]: A tuple containing the numeric answer and the proof

        Raises:
            NotImplementedError: This method should be implemented by subclasses
            ValueError: If the question is empty or invalid
            RuntimeError: If the answer or proof could not be extracted from the response
        """
        raise NotImplementedError


class HuggingFaceWrapper(ModelWrapper):
    JUDGE_PROMPT_WITHOUT_CRITIQUE = """Here's a math question: {question}

Here's a proposed solution: {answer_numeric}

Proof: {answer_proof}

Do you think the solution is correct? (Yes/No)"""

    JUDGE_PROMPT_WITH_CRITIQUE = """Here's a math question: {question}

Here's a proposed solution: {answer_numeric}

Proof: {answer_proof}

Here's a critique of the proposed solution: {critique}

Do you think the solution is correct? (Yes/No)"""

    CRITIC_PROMPT = """You are given the following question and an incorrect answer.

Question: {question}

Incorrect Answer: {answer_numeric}
Invalid Proof: {answer_proof}

Write an effective critique of the answer, identifying relevant mistakes or omissions in the proof. It's essential that we clearly show the wrongness of the answer."""

    ANSWER_PROMPT = """Here's a math question: {question}

Write a solution to the question. Your solution should be a number and a proof that explains how you arrived at that number.

Respond with a proof in <proof> </proof> tags, and a numeric answer in <numeric_answer> </numeric_answer> tags. The numeric_answer should be one number, without units or currency.

For example:
```
Sure, here's my answer:

<proof>
Let's call the number of clips Natalia sold in April "A" and the number of clips she sold in May "B". Since she sold half as many clips in May as she did in April, we can write:

B = A/2

We know that Natalia sold 48 clips in April, so A = 48. Substituting this value into the equation above, we get:

B = 48/2 = 24

Therefore, Natalia sold a total of 48 clips in April and 24 clips in May.

48 + 24 = 72
</proof>
<numeric_answer>72</numeric_answer>
```
"""

    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=HF_TOKEN,
            # torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _format_critic_prompt(self, unformatted_prompt: str):
        raise NotImplementedError

    def _format_judge_prompt(self, unformatted_prompt: str) -> str:
        raise NotImplementedError

    def _format_answer_prompt(self, unformatted_prompt: str) -> str:
        raise NotImplementedError

    def _extract_critique_from_response(self, response: str) -> str:
        raise NotImplementedError

    def _extract_answer_and_proof_from_response(self, response: str) -> str:
        raise NotImplementedError

    def get_judge_confidence(
        self,
        item: DatasetItem,
        critique: str = None,
        letters=["Yes", "No"],
    ) -> str:
        if critique:
            unformatted_prompt: str = self.JUDGE_PROMPT_WITH_CRITIQUE.format(
                question=item.question,
                answer_numeric=item.answer_correct.numeric,
                answer_proof=item.answer_correct.proof,
                critique=critique,
            )
        else:
            unformatted_prompt: str = self.JUDGE_PROMPT_WITHOUT_CRITIQUE.format(
                question=item.question,
                answer_numeric=item.answer_correct.numeric,
                answer_proof=item.answer_correct.proof,
                critique=critique,
            )

        full_prompt = self._format_judge_prompt(unformatted_prompt)
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model(input_ids).logits[0, -1, :]
        probs = output.softmax(dim=0)

        yes_prob = probs[self.tokenizer.encode(letters[0])[-1]].item()
        no_prob = probs[self.tokenizer.encode(letters[1])[-1]].item()
        return yes_prob / (yes_prob + no_prob)

    def critique(
        self,
        item: DatasetItem,
    ) -> str:
        unformatted_prompt = self.CRITIC_PROMPT.format(
            question=item.question,
            answer_numeric=item.answer_correct.numeric,
            answer_proof=item.answer_correct.proof,
        )
        full_prompt = self._format_critic_prompt(unformatted_prompt)
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model.generate(input_ids, max_length=MAX_LENGTH)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self._extract_critique_from_response(decoded)
        return response

    def answer_question(self, question: str) -> str:
        unfomatted_prompt = self.ANSWER_PROMPT.format(question=question)
        full_prompt = self._format_answer_prompt(unfomatted_prompt)
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model.generate(input_ids, max_length=MAX_LENGTH)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self._extract_answer_and_proof_from_response(decoded)


class WizardMathWrapper(HuggingFaceWrapper):
    def _format_critique_prompt(self, unformatted_prompt: str):
        """
        This comes from Huggingface
        https://huggingface.co/WizardLM/WizardMath-70B-V1.0
        """
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{unformatted_prompt}\n\n### Response:"

    def _format_judge_prompt(self, unformatted_prompt: str) -> str:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{unformatted_prompt}\n\n### Response: ("

    def _format_answer_prompt(self, unformatted_prompt: str) -> str:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{unformatted_prompt}\n\n### Response:"

    def _extract_critique_from_response(self, response: str) -> str:
        return response.split("Response:", 1)[1].strip()

    def _extract_answer_and_proof_from_response(self, response: str) -> str:
        return response.split("Response:", 1)[1].strip()


# meta-llama/Llama-2-7b-chat-hf, etc
class Llama2Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = "Sure, here's my critique:\n\n"
    CRITIC_SYSTEM_PROMPT = "You're a math expert who critiques math problems."
    JUDGE_SYSTEM_PROMPT = "You're a judge who evaluates math problems."
    ANSWERER_WORDS_IN_MOUTH = "Sure, here's my answer:\n\n"
    ANSWERER_SYSTEM_PROMPT = "You're a math expert who answers math problems."

    def _format_critic_prompt(self, unformatted_prompt: str):
        return f"""<s>[INST] <<SYS>>
        {self.CRITIC_SYSTEM_PROMPT}
        <</SYS>>
        {unformatted_prompt} [/INST] {self.CRITIC_WORDS_IN_MOUTH}""".strip()

    def _format_judge_prompt(self, unformatted_prompt: str):
        return f"""<s>[INST] <<SYS>>
        {self.JUDGE_SYSTEM_PROMPT}
        <</SYS>>
        {unformatted_prompt} [/INST] (""".strip()

    def _format_answer_prompt(self, unformatted_prompt: str):
        return f"""<s>[INST] <<SYS>>
        {self.ANSWERER_SYSTEM_PROMPT}
        <</SYS>>
        {unformatted_prompt} [/INST] {self.ANSWERER_WORDS_IN_MOUTH}""".strip()

    def _extract_critique_from_response(self, response: str) -> str:
        return response.split("critique:\n\n")[1].strip()

    def _extract_answer_and_proof_from_response(self, response: str) -> str:
        return response.split("answer:\n\n", 2)[2]


# meta-llama/Meta-Llama-3-8B-Instruct, etc
class Llama3Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = "Sure, here's my critique:\n\n"
    CRITIC_SYSTEM_PROMPT = "You're a math expert who critiques math problems."
    JUDGE_SYSTEM_PROMPT = "You're a judge who evaluates math problems."
    ANSWERER_WORDS_IN_MOUTH = "Sure, here's my answer:\n\n"
    ANSWERER_SYSTEM_PROMPT = "You're a math expert who answers math problems."

    def _format_critic_prompt(self, unformatted_prompt: str):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.CRITIC_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{unformatted_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{self.CRITIC_WORDS_IN_MOUTH}"""

    def _format_judge_prompt(self, unformatted_prompt: str):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.JUDGE_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{unformatted_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

("""

    def _format_answer_prompt(self, unformatted_prompt: str):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.ANSWERER_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{unformatted_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{self.ANSWERER_WORDS_IN_MOUTH}
        """

    def _extract_critique_from_response(self, response: str) -> str:
        return response.split("critique:\n\n")[1].strip()

    def _extract_answer_and_proof_from_response(self, response: str) -> str:
        return response.split("answer:\n\n", 2)[2]


# google/gemma-2-9b, google/gemma-2-27b
class Gemma2Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = "Sure, here's my critique:"
    ANSWERER_WORDS_IN_MOUTH = "Sure, here's my answer:"

    def _format_critic_prompt(self, unformatted_prompt: str):
        return f"""<start_of_turn>user\n{unformatted_prompt}<end_of_turn>\n<start_of_turn>model\n{self.CRITIC_WORDS_IN_MOUTH}"""

    def _format_judge_prompt(self, unformatted_prompt: str):
        return f"""<start_of_turn>user\n{unformatted_prompt}<end_of_turn>\n<start_of_turn>model\n("""

    def _format_answer_prompt(self, unformatted_prompt: str):
        return f"""<start_of_turn>user\n{unformatted_prompt}<end_of_turn>\n<start_of_turn>model\n{self.ANSWERER_WORDS_IN_MOUTH}"""

    def _extract_critique_from_response(self, response: str) -> str:
        return response.split(" critique:")[1].strip()

    def _extract_answer_and_proof_from_response(self, response: str) -> str:
        return response.split("answer:\n\n", 2)[2]
