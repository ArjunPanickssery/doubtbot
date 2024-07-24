import os
from typing import List

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import DatasetItem

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_LENGTH = 1024


# Wrapper for both debaters and judges
class ModelWrapper:
    def __init__(self, model_id: str, model_name):
        """
        Initializes the LLM debater with the specified model.

        Args:
            model_id (str): A short identifier for the model ("llama2_7b")
            model_name (str): The name of the model to load from HF/API
        """
        self.model_id = model_id
        self.model_name = model_name

    # For critics
    def critique(
        self,
        item: DatasetItem,
    ) -> str:
        raise NotImplementedError

    # For judges
    def get_judge_confidence(
        self,
        item: DatasetItem,
        critique: str = None,
    ) -> str:
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

    CRITIC_PROMPT = """You are provided with a mathematical question, a proof, and an answer. Your task is to carefully analyze the proof and the answer, identify any errors or flaws, and explain why the proof is invalid. Additionally, you must convince me that the answer is incorrect. Structure your response in the following manner:

Restate the Question: Briefly summarize the mathematical question for clarity.
Proof Analysis:
Identify Errors: Point out specific steps or logical inconsistencies in the proof that make it invalid.
Explain Errors: Provide detailed explanations of why these errors occur and how they impact the validity of the proof.
Answer Evaluation:
Identify Incorrect Answer: Clearly state why the provided answer is wrong.
Provide Correct Answer (if possible): If applicable, give the correct answer with a brief explanation or correct steps.
Example Input:

Math Question: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
Proof: "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10"
Answer: 10.0

Example Output:

Restate the Question: The question asks how much Weng earned for babysitting 50 minutes at a rate of $12 per hour.

Proof Analysis:

Identify Errors: The proof incorrectly calculates the earnings per minute and multiply instead of add to find the total earnings.
Explain Errors:
The correct approach should first determine the per-minute rate by dividing the hourly rate by 60 minutes.
The proof mistakenly states the per-minute rate correctly as $0.2 but then multiply this rate directly to the number of minutes worked instead of adding.
The error is in the step: "Working 50 minutes, she earned 0.2 x 50 = $10" Here, the correct operation is to add the per-minute rate by the total minutes worked, not add.

Answer Evaluation:

Identify Incorrect Answer: The provided answer, $10.0, is incorrect due to the flawed calculation method.
Provide Correct Answer: The correct calculation should be:
Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 + 50 = $50.2
Therefore, the correct answer is $50.2, not $10.0.

Question: {question}

Incorrect Answer: {answer_numeric}
Invalid Proof: {answer_proof}

Use this structured approach to provide a detailed and convincing explanation of why the proof is invalid and why the answer is wrong.
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

    def _format_critic_prompt(self, unformatted_prompt: str):
        raise NotImplementedError

    def _format_judge_prompt(self, unformatted_prompt: str) -> str:
        raise NotImplementedError

    def _extract_critique_from_response(self, response: str) -> str:
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


class WizardMathWrapper(HuggingFaceWrapper):
    def _format_critique_prompt(self, unformatted_prompt: str):
        """
        This comes from Huggingface
        https://huggingface.co/WizardLM/WizardMath-70B-V1.0
        """
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{unformatted_prompt}\n\n### Response:"

    def _format_judge_prompt(self, unformatted_prompt: str) -> str:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{unformatted_prompt}\n\n### Response: ("

    def _extract_critique_from_response(self, response: str) -> str:
        return response.split("Response:")[1].strip()


# meta-llama/Llama-2-7b-chat-hf, etc
class Llama2Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = "Sure, here's my critique:\n\n"
    CRITIC_SYSTEM_PROMPT = "You're a math expert who critiques math problems."
    JUDGE_SYSTEM_PROMPT = "You're a judge who evaluates math problems."

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

    def _extract_critique_from_response(self, response: str) -> str:
        return response.split("critique:\n\n")[1].strip()


# meta-llama/Meta-Llama-3-8B-Instruct, etc
class Llama3Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = "Sure, here's my critique:\n\n"
    CRITIC_SYSTEM_PROMPT = "You're a math expert who critiques math problems."
    JUDGE_SYSTEM_PROMPT = "You're a judge who evaluates math problems."

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

    def _extract_critique_from_response(self, response: str) -> str:
        return response.split("critique:\n\n")[1].strip()


# google/gemma-2-9b, google/gemma-2-27b
class Gemma2Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = "Sure, here's my critique:"

    def _format_critic_prompt(self, unformatted_prompt: str):
        return f"""<start_of_turn>user\n{unformatted_prompt}<end_of_turn>\n<start_of_turn>model\n{self.CRITIC_WORDS_IN_MOUTH}"""

    def _format_judge_prompt(self, unformatted_prompt: str):
        return f"""<start_of_turn>user\n{unformatted_prompt}<end_of_turn>\n<start_of_turn>model\n("""

    def _extract_argument_from_response(self, response: str) -> str:
        return response.split(" critique:")[1].strip()
