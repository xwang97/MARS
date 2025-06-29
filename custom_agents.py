from openai import OpenAI
import random
from utils import get_api_key
from sagemaker.predictor import retrieve_default
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient


# ======== API key Setup ==============
openai_api_key = get_api_key("../openai_api_key_4_XiaoWang.txt")
hf_api_key = get_api_key("../hf_api_key.txt")


# ======= Definition of the OpenAI agent class ================
class OpenAIAgent:
    def __init__(self, name, model="gpt-3.5-turbo", api_key=None):
        self.name = name
        self.model = model
        # self.client = OpenAI(api_key=api_key)
        self.client = InferenceClient(model=model, api_key=api_key)
        self.total_tokens = 0
        self.token_log = []

    def run(self, prompt: str | list[dict]) -> str | dict:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            return self._call_openai(messages)["content"]  # Return string for backward compatibility
        elif isinstance(prompt, list):
            return self._call_openai(prompt)  # Return dict for debate/chat-style use

    def _call_openai(self, messages: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        # Track token usage
        usage = response.usage
        if usage:
            self.total_tokens += usage.total_tokens
            self.token_log.append({
                "agent": self.name,
                "model": self.model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "messages_preview": messages[-1]["content"][:100]  # optional
            })
        return {
            "role": "assistant",
            "content": response.choices[0].message.content.strip()
        }


# ======= Definition of the AWS agent class ================
class AWSAgent:
    def __init__(self, name, endpoint_name, model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=4096):
        self.name = name
        self.endpoint_name = endpoint_name
        self.model = model
        self.predictor = retrieve_default(endpoint_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_new_tokens = max_new_tokens
        self.total_tokens = 0
        self.token_log = []

    def run(self, prompt: str | list[dict]) -> str | dict:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            return self._call_aws(messages)["content"]  # for compatibility
        elif isinstance(prompt, list):
            return self._call_aws(prompt)  # chat-style list[dict]

    def _call_aws(self, messages: list[dict]) -> dict:
        formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        payload = {
            "inputs": formatted_input,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "details": True,
                "decoder_input_details": True
            }
        }
        response = self.predictor.predict(payload)

        input_tokens = len(response['details']['prefill'])
        output_tokens = response["details"]["generated_tokens"]
        self.total_tokens += (input_tokens + output_tokens)
        self.token_log.append({
            "agent": self.name,
            "model": self.model,
            "endpoint": self.endpoint_name,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": self.total_tokens,
            "messages_preview": messages[-1]["content"][:100]
        })
        return {
            "role": "assistant",
            "content": response["generated_text"].strip()
        }


# ======= Definition of the agent class using HuggingFace ================
class HFAgent:
    def __init__(self, name, model="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=4096):
        self.name = name
        self.model = model
        self.client = InferenceClient(api_key=hf_api_key)
        self.max_new_tokens = max_new_tokens
        self.total_tokens = 0
        self.token_log = []

    def run(self, prompt: str | list[dict]) -> str | dict:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            return self._call_hf(messages)["content"]  # Return string for backward compatibility
        elif isinstance(prompt, list):
            return self._call_hf(prompt)  # Return dict for debate/chat-style use

    def _call_hf(self, messages: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        # Track token usage
        usage = response.usage
        if usage:
            self.total_tokens += usage.total_tokens
            self.token_log.append({
                "agent": self.name,
                "model": self.model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "messages_preview": messages[-1]["content"][:100]  # optional
            })
        return {
            "role": "assistant",
            "content": response.choices[0].message.content.strip()
        }


# ======== Define the review related agents ==================
def create_author_agent(name="Author", model="gpt-3.5-turbo", endpoint=None):
    if model == "gpt-3.5-turbo":
        return OpenAIAgent(
            name=name,
            model=model,
            api_key=openai_api_key
        )
    else:
        if endpoint is None:
            raise ValueError("Endpoint can't be None when using AWS models")
        return AWSAgent(
            name=name,
            endpoint_name=endpoint,
            model=model,
        )


def create_reviewer_agents(num_reviewers: int = 3, model="gpt-3.5-turbo", endpoint=None):
    reviewers = []
    if model == "gpt-3.5-turbo":
        for i in range(num_reviewers):
            llm = random.choice(reviewer_llms)
            reviewer = OpenAIAgent(
                name=f"Reviewer_{i+1}",
                model=llm,
                api_key=openai_api_key
            )
            reviewers.append(reviewer)
    else:
        if endpoint is None:
            raise ValueError("Endpoint can't be None when using AWS models")
        for i in range(num_reviewers):
            reviewer = AWSAgent(
                name=f"Reviewer_{i+1}",
                endpoint_name=endpoint,
                model=model,
            )
            reviewers.append(reviewer)
    return reviewers


def create_meta_reviewer_agent(model="gpt-3.5-turbo", endpoint=None):
    if model == "gpt-3.5-turbo":
        return OpenAIAgent(
            name="MetaReviewer",
            model=meta_llm,
            api_key=openai_api_key
        )
    else:
        if endpoint is None:
            raise ValueError("Endpoint can't be None when using AWS models")
        return AWSAgent(
            name="MetaReviewer",
            endpoint_name=endpoint,
            model=model,
        )
