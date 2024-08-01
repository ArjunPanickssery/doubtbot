import math
import os
from typing import List, Dict

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import anthropic
from openai import OpenAI

from data import DatasetItem, QuestionAnswerPair

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_LENGTH = 1024


# Wrapper for both debaters and judges
class ModelWrapper:
    JUDGE_PROMPT_WITHOUT_CRITIQUE = """Here's a math question: {question}

Here's a proposed solution: {answer_numeric}

Proof: {answer_proof}

Do you think the solution is correct (Yes/No)? Answer with only yes or no."""

    JUDGE_PROMPT_WITH_CRITIQUE = """Here's a math question: {question}

Here's a proposed solution: {answer_numeric}

Proof: {answer_proof}

Here's a critique of the proposed solution: {critique}

Do you think the solution is correct (Yes/No)? Answer with only yes or no."""

    CRITIC_PROMPT = """We're conducting an educational exercise. You are given the following question and an answer.
     
Question: {question}

Answer: {answer_numeric}
Proof: {answer_proof}

Write an effective critique of the answer, identifying relevant mistakes or omissions in the proof. It's essential that we clearly show the wrongness of the answer."""
    
    JUDGE_SYSTEM_PROMPT = """You are a math expert who evaluates math problems."""
    JUDGE_WORDS_IN_MOUTH = ""

    CRITIC_SYSTEM_PROMPT = """You are a math expert who critiques math problems as part of an educational exercise."""
    CRITIC_WORDS_IN_MOUTH = ""

    def __init__(self, model_id: str, model_name: str):
        """
        Initializes the LLM debater with the specified model.

        Args:
            model_id (str): A short identifier for the model ("llama2_7b")
            model_name (str): The name of the model to load from HF/API
        """
        self.model_id = model_id
        self.model_name = model_name

    def _get_token_probs(self, tokens: List[str], system_prompt: str, user_prompt: str, words_in_mouth: str):
        raise NotImplementedError

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth: str):
        raise NotImplementedError

    # For critics
    def critique(
        self,
        item: QuestionAnswerPair,
    ) -> str:
        unformatted_prompt = self.CRITIC_PROMPT.format(
            question=item.question,
            answer_numeric=item.answer_numeric,
            answer_proof=item.answer_proof,
        )
        return self._response(self.CRITIC_SYSTEM_PROMPT, unformatted_prompt, self.CRITIC_WORDS_IN_MOUTH)

    # For judges
    def get_judge_confidence(
        self,
        item: QuestionAnswerPair,
        critique: str = None,
    ) -> float:
        if critique:
            unformatted_prompt: str = self.JUDGE_PROMPT_WITH_CRITIQUE.format(
                question=item.question,
                answer_numeric=item.answer_numeric,
                answer_proof=item.answer_proof,
                critique=critique,
            )
        else:
            unformatted_prompt: str = self.JUDGE_PROMPT_WITHOUT_CRITIQUE.format(
                question=item.question,
                answer_numeric=item.answer_numeric,
                answer_proof=item.answer_proof,
            )

        token_probs = self._get_token_probs(
            ['Yes', 'No'],
            self.JUDGE_SYSTEM_PROMPT,
            unformatted_prompt,
            self.JUDGE_WORDS_IN_MOUTH,
        )
        print(token_probs)
        return token_probs['Yes'] / (token_probs['Yes'] + token_probs['No'])

class HuggingFaceWrapper(ModelWrapper):
    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=HF_TOKEN,
            # torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def _format_prompt(
        self, system_prompt: str, user_prompt: str, words_in_mouth=""
    ) -> str:
        raise NotImplementedError

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        formatted_prompt = self._format_prompt(
            system_prompt, user_prompt, words_in_mouth
        )
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model.generate(
            input_ids,
            max_new_tokens=MAX_LENGTH,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,  # Set pad_token_id to EOS token ID to avoid padding
        ).to("cpu")
        decoded = self.tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )  # Decode only the generated tokens
        return decoded

    def _get_token_probs(
        self,
        response_tokens: List[str],
        system_prompt: str,
        user_prompt: str,
        words_in_mouth="",
    ) -> Dict[str, float]:
        formatted_prompt = self._format_prompt(
            system_prompt, user_prompt, words_in_mouth
        )
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model(input_ids).logits[0, -1, :].to("cpu")
        probs = output.softmax(dim=0)

        return {
            token: probs[self.tokenizer.encode(token)[-1]].item()
            for token in response_tokens
        }


class WizardMathWrapper(HuggingFaceWrapper):
    def _format_prompt(
        self, system_prompt: str, user_prompt: str, words_in_mouth=""
    ) -> str:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{user_prompt}\n\n### Response:"


# meta-llama/Llama-2-7b-chat-hf, etc
class Llama2Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = "Sure, here's my response:\n\n"

    def _format_prompt(self, system_prompt: str, user_prompt: str, words_in_mouth=""):
        return f"""<s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        {user_prompt} [/INST] {words_in_mouth}""".strip()


# meta-llama/Meta-Llama-3-8B-Instruct, etc
class Llama3Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = (
        "Sure, here's my response:\n\n"  # Start with a leading space
    )

    def _format_prompt(
        self, system_prompt: str, user_prompt: str, words_in_mouth=""
    ) -> str:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{words_in_mouth}"""


# google/gemma-2-9b, google/gemma-2-27b
class Gemma2Wrapper(HuggingFaceWrapper):
    CRITIC_WORDS_IN_MOUTH = "Sure, here's my response:" 

    def _format_prompt(
        self, system_prompt: str, user_prompt: str, words_in_mouth=""
    ) -> str:
        return f"""<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{words_in_mouth}"""


class GPTWrapper(ModelWrapper):
    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        self.client = OpenAI()

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        """Generates model output using OpenAI's API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=MAX_LENGTH,
            n=1,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def _get_token_probs(
        self,
        response_tokens: List[str],
        system_prompt: str,
        user_prompt: str,
        words_in_mouth="",
    ) -> Dict[str, float]:
        """Generates token probabilities using OpenAI's API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt
                + f"\n\nResponse with just {response_tokens[0]} or {response_tokens[1]}, nothing else.",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=MAX_LENGTH,
            n=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )

        token_probs = {token: 0 for token in response_tokens}
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        for item in logprobs:
            if item.token in token_probs:
                token_probs[item.token] = math.exp(item.logprob)

        total_prob = sum(token_probs.values())
        return {k: v / total_prob for k, v in token_probs.items()}


# Claude's API doesn't support logprobs so we can't use it as a judge
class ClaudeWrapper(ModelWrapper):
    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        self.client = anthropic.Anthropic()

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        """Generates model output using Anthropic's API"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=MAX_LENGTH,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ],
        )

        return message.content[0].text

    def _get_token_probs(
        self,
        response_tokens: List[str],
        system_prompt: str,
        user_prompt: str,
        words_in_mouth="",
    ) -> Dict[str, float]:
        """Generates token probabilities using Anthropic's API"""
        raise NotImplementedError("Anthropic does not provide token probabilities")