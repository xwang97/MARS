from openai import OpenAI
import random
from utils import get_api_key
from cerebras.cloud.sdk import Cerebras
from mistralai import Mistral
from huggingface_hub import InferenceClient


# ======== API key Setup ==============
openai_api_key = get_api_key("../openai_api_key_4_XiaoWang.txt")
nvidia_api_key = get_api_key("../nvidia_api_key.txt")
deepseek_api_key = get_api_key("../deepseek_api_key.txt")
cerebras_api_key = get_api_key("../cerebras_api_key.txt")
mistral_api_key = get_api_key("../mistral_api_key.txt")
hf_api_key = get_api_key("../hf_api_key.txt")

# ======== LLM Setup ==============
author_llm = "microsoft/phi-3.5-moe-instruct"
reviewer_llms = [
    # "gpt-3.5-turbo",
    # "gpt-4o-mini",
    # "mistral-small-latest",
    # "ministral-8b-latest",
    # "mistralai/mistral-7b-instruct-v0.3",
    # "google/gemma-2-9b-it",
    # "qwen/qwq-32b",
    # "qwen/qwen2-7b-instruct",
    # "qwen/qwen2.5-7b-instruct",
    # "gpt-4o-mini",
    # "llama-3.3-70b",
    # "meta/llama-3.3-70b-instruct",
    # "llama-3.1-8b",
    # "mistralai/mixtral-8x7b-instruct-v0.1",
    # "qwen-3-32b"
    "mistralai/mixtral-8x22b-instruct-v0.1",
]
# meta_llm = "deepseek-chat"
meta_llm = "gpt-4o-mini"


# ======= Definition of the OpenAI agent class ================
class OpenAIAgent:
    def __init__(self, name, model="gpt-3.5-turbo"):
        self.name = name
        self.model = model
        self.openai_sdk = True
        if model[:3] == "gpt":
            self.client = OpenAI(api_key=openai_api_key)
        elif "deepseek" in model:
            self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
        elif "llama" in model:
            # self.client = Cerebras(api_key=cerebras_api_key)
            self.client = OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1")
        elif "qwen" in model or "gemma" in model or "mistral" in model:
            self.client = OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1")
        # elif "mistral" in model or "ministral" in model:
        #     self.client = Mistral(api_key=mistral_api_key)
        #     self.openai_sdk = False
        self.total_tokens = 0
        self.token_log = []

    def run(self, prompt: str | list[dict]) -> str | dict:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            return self._call_openai(messages)["content"]  # Return string for backward compatibility
        elif isinstance(prompt, list):
            return self._call_openai(prompt)  # Return dict for debate/chat-style use

    def _call_openai(self, messages: list[dict]) -> dict:
        if not self.openai_sdk:
            response = self.client.chat.complete(
                model=self.model, messages=messages
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=False
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
def create_author_agent(name="Author"):
    return OpenAIAgent(
        name=name,
        model=author_llm,
    )


def create_reviewer_agents(num_reviewers: int = 3):
    reviewers = []
    for i in range(num_reviewers):
        # llm = random.choice(reviewer_llms)
        llm = reviewer_llms[i]
        reviewer = OpenAIAgent(
            name=f"Reviewer_{i+1}",
            model=llm,
        )
        reviewers.append(reviewer)
    return reviewers


def create_meta_reviewer_agent():
    return OpenAIAgent(
        name="MetaReviewer",
        model=meta_llm,
    )